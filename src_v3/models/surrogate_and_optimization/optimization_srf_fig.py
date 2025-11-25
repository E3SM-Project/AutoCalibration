##########################
# load various packages: see requirements.txt
##########################
print("Importing packages and source code")
# various utility
import numpy as np
import xarray as xr
import yaml
import os, time, sys,copy,pdb
import glob
import joblib
from datetime import datetime
import pytz
from tesuract.preprocessing import DomainScaler

# modules in this folder for pre/postprocessing input data and results
import preprocessing
import postprocessing
from postprocessing import paste_nonempty

# optimization functions
from scipy.optimize import minimize
import multiprocessing as mp

# plots and presentation
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import csv


#############################
# load config file and prep for job 
#############################
# cfg = yaml.safe_load(open('config_optimization.yaml'))
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

# Specify cost function, save details, and surrogate
sys.path.insert(1, "../cost_functions") # Add file to the Python path
cost_function_file = cfg["cost_function"]
cost_function_name = cost_function_file[:-3]
exec(f'import {cost_function_name}') # import cost_function_file
exec(f'from {cost_function_name} import cost_function') # import the cost_function itself

savedir = cfg["savedir"]
savename = cfg["savename"]
if savename is None:
    savename = ""
surrogate_dir = cfg["surrogate"] 
if savedir is None:
    savedir = surrogate_dir
surrogate_cfg = yaml.safe_load(open(os.path.join(surrogate_dir, 'surrogate_fit/specs/config_surrogate.yaml')))


# Specifications for Optimization and Validation
datadir = cfg["datadir"]
responses = surrogate_cfg["responses"]
response_names =  [res for rf in responses.values() for res in rf]
param_bounds_dict = cfg["param_bounds"]

optimization = cfg["optimization"]  
validation = cfg["validation"]

# Machine details
n_cores = cfg["n_cores"]
if cfg["n_cores"] == "n_cores_half":
    n_cores = int(mp.cpu_count() / 2)
else:
    n_cores = cfg["n_cores"]
machine = cfg["machine"] 

# get datetime to use in filenames
current_datetime = datetime.now(pytz.timezone("US/Mountain"))
formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")

# use cost function for a single sim to define cost function for an ensemble of sims
def get_ensemble_cost(n, response_names, Y, obs, area_weights, lat=None, lon=None):
    cost = []; component_cost = []
    for j in range(n):
        sim_output_j = {}
        for i, res in enumerate(response_names):
            sim_output_j[res] = Y[i][j]

        cost_j, component_cost_j, component_weights = cost_function(obs, sim_output_j, area_weights, lat, lon, return_components=True, mask=mask)
        cost.append(cost_j)
        component_cost.append(list(component_cost_j.values()))

    return cost, component_cost, component_weights


#############################
# Load surrogate
#############################
print("Loading model from {0}".format(surrogate_dir))
surrogate_path = glob.glob(os.path.join(surrogate_dir, "surrogate_fit/output/model*.joblib"))[0]
surrogate = joblib.load(surrogate_path)


#############################
# Load Data 
#############################
X, x_labels, X_bounds, n_inputs, Y_raw, Y_obs_raw, n_responses, normalized_area_weights, area_weights, workdirs, lat_list, lon_list = preprocessing.load_data(datadir, responses, param_bounds_dict, enforce_bounds=validation["enforce_bounds"], return_lat_lon = True)

# transform to interval [-1,1]
feature_transform = DomainScaler(dim=X.shape[1],input_range=list(X_bounds),output_range=(-1, 1))
X_s = feature_transform.fit_transform(X)
X_s_bounds = [(pb[0], pb[1]) for pb in feature_transform.fit_transform(X_bounds.T).T]

if optimization['param_start'] is not None:
    optimization['param_start'] = feature_transform.fit_transform(np.array(optimization['param_start']))


# Transform target and obs data
Y, Y_obs, joinT, Y_joined, response_transform, Y_obs_joined, scalingT, Y_s, W, M, mask = preprocessing.transform_data(Y_raw, Y_obs_raw, n_responses, normalized_area_weights, response_transforms=None, response_names=response_names)

nsim = len(Y_s)

obs = {}; area_weights = {}; lat = {}; lon = {}
for i, res in enumerate(response_names):
    obs[res] = Y_obs[i]
    area_weights[res] = W[i] / np.sum(W[i])
    if lat_list[i] is None:
        lat[res] = None
    else:
        lat[res] = np.array(lat_list[i])[M[i]]
    if lon_list[i] is None:
        lon[res] = None
    else:
        lon[res] = np.array(lon_list[i])[M[i]]


#############################
# define optimization functions 
#############################
idx_lambda_low = np.where(x_labels == "lambda_low")
idx_lambda_high = np.where(x_labels == "lambda_high")

def params_to_cost(input_params):
    # Enforce parameter constraints (hard-coded for now)
    # Note: generally, we would need to transform back to original scale before
    # comparing params, but we're ok here b/c lambda_low and lambda_high have
    # the same bounds
    if input_params[idx_lambda_low] > input_params[idx_lambda_high]:
        cost = 1e100 
    else:
        preds_joined = surrogate.predict(input_params)
        preds_list = joinT.inverse_transform(preds_joined)
        preds = {}
        for i,res in enumerate(response_names):
            preds[res] = preds_list[i] 

        cost = cost_function(obs, preds, area_weights, lat, lon, mask=mask)

    return cost

def parallel_optimization(xstart):
    res = minimize(
        params_to_cost,
        xstart,
        method="L-BFGS-B",
        jac=None,
        bounds=X_s_bounds,
        options={"ftol": 1e-10, "maxiter": 70000, "disp": False}
    )
    return res.fun, res.x, res.nfev, res.nit

def optimize_params():
    rn = np.random.RandomState(optimization["seed"])
    R = int(mp.cpu_count())
    xstarts = 2 * rn.rand(R, n_inputs) - 1
    
    # Make sure we aren't starting out-of-bounds
    if optimization['param_start'] is None:
        idx_oob = np.where(xstarts[:,idx_lambda_low] > xstarts[:,idx_lambda_high])[0]
        while(len(idx_oob) > 0):
            xstarts[idx_oob,:] = 2 * rn.rand(len(idx_oob), n_inputs) - 1
            idx_oob = np.where(xstarts[:,idx_lambda_low] > xstarts[:,idx_lambda_high])[0]
    else:
        xstarts[:] = optimization['param_start']
   
    # optimize
    with mp.get_context("fork").Pool() as pool:
        results = pool.map(parallel_optimization, xstarts)

    # extract results
    fevals = [soln[0] for soln in results]
    xopts = [soln[1] for soln in results]
    optarg = np.argmin(fevals)
    xopt_s = xopts[optarg]
    xopt_ = feature_transform.inverse_transform(xopt_s)[0]
    
    return xopt_, xopt_s
   

#############################
# Run Optimization 
#############################
if optimization['run']:    
    print("\nOptimizing")

    # Optimize
    xopt_, xopt_s = optimize_params()

    # print and save optimization results
    optimization_dir = os.path.join(savedir, "optimization", paste_nonempty(["optimization", savename, formatted_datetime], "_"))
    optimization_output_dir = os.path.join(optimization_dir, "output")
    os.makedirs(optimization_output_dir)
    
    optimization_path = os.path.join(optimization_output_dir, paste_nonempty(["opt_params", savename], "_") + ".txt")
    np.savetxt(optimization_path, xopt_)

    postprocessing.print_opt_params(
        xopt_,
        x_labels,
        cost_function_name,
        2*n_cores,
        n_inputs,
        solver="L-BFGS-B",
        filename=optimization_path
   )
            
    # save specs
    postprocessing.save_specs(os.path.join(optimization_dir, 'specs'), workdirs, save_optimization_specs=True)

    # save environment files
    postprocessing.save_environment(os.path.join(optimization_dir, 'environment'), machine)
        

###################
# validation
###################
# Load in simulation runs and observations
if validation['modsim']['run'] or validation['surrogate']['run']:
    print("\nValidating")
    
    # set up validation directories
    validation_dir = os.path.join(savedir, "validation", paste_nonempty(["validation", savename, formatted_datetime], "_"))
    validation_output_dir = os.path.join(validation_dir, "output")
    
    # Get validation data
    X_val, Y_val_raw, workdirs_val = preprocessing.load_val_data(datadir, responses)
        
        
# modsim validation
if validation['modsim']['run']:
    # calculate rmse of ensemble members
    cost_ens, component_cost_ens, component_weights = get_ensemble_cost(nsim, response_names, Y, obs, area_weights, lat, lon)
    cost_ens_dat = np.column_stack((workdirs, X, component_cost_ens, cost_ens))
   
    # calculate rmse at validation locations
    Y_val = preprocessing.transform_val_data(Y_val_raw, n_responses, M)
    nval = len(Y_val[0])

    cost_val, component_cost_val, component_weights = get_ensemble_cost(nval, response_names, Y_val, obs, area_weights, lat, lon)
    cost_val_dat = np.column_stack((workdirs_val, X_val, component_cost_val, cost_val))
   
    # combine all info into one array
    component_names = list(component_weights.keys())
    colnames = ['id'] + list(x_labels) + component_names + ['total_cost']
    cost_dat = np.vstack([colnames, cost_ens_dat, cost_val_dat])
    
    # Write the data to a CSV file
    os.makedirs(validation_output_dir)
    val_modsim_path = os.path.join(validation_output_dir, paste_nonempty(["val_modsim", savename], "_") + ".csv")
    with open(val_modsim_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(cost_dat)
   
    # save weights
    weights_path = os.path.join(validation_output_dir, paste_nonempty(["cost_weights", savename], "_") + ".csv")
    with open(weights_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(np.vstack([list(component_weights.keys()), list(component_weights.values())]))

    # create variables needed for plots 
    component_arr = np.vstack([component_cost_ens,
                               component_cost_val])
    total_cost = np.concatenate([cost_ens, cost_val])
    modsim_cost = np.column_stack([component_arr, total_cost])
    subsets = {'ens': list(range(len(X)))}
    idx = len(X)
    for wd in workdirs_val:
        subsets[wd] = [idx]
        idx+=1
    all_cost_labels = component_names + ['total_cost']
    input_arr = np.vstack([X, X_val])
    
    # make plots
    postprocessing.plot_cost_boxplots(validation_output_dir, savename + "_modsim",
                                      modsim_cost, all_cost_labels, subsets) 
    postprocessing.plot_inputs_v_cost(validation_output_dir, savename + "_modsim",
                                      input_arr, modsim_cost, x_labels, all_cost_labels, subsets)
    postprocessing.plot_components_v_total_cost(validation_output_dir, savename + "_modsim",
                                                component_arr, total_cost, component_names)
    postprocessing.plot_cost_corr(validation_output_dir, savename + "_modsim",
                                  modsim_cost, all_cost_labels)

    # save specs
    postprocessing.save_specs(os.path.join(validation_dir, 'specs'), workdirs, save_optimization_specs=True)

    # save environment files
    postprocessing.save_environment(os.path.join(validation_dir, 'environment'), machine)
    
# Surrogate validation
if validation['surrogate']['run']:
    param_paths = cfg["validation"]["surrogate"]['param_paths']
    if param_paths is not None: 
        for path in param_paths:
            xval = postprocessing.load_params(path, x_labels)
            X_val = np.concatenate([X_val, np.expand_dims(xval, axis = 0)])
            workdirs_val.append(path)
        
    if optimization['run']:
        X_val = np.concatenate([X_val, np.expand_dims(xopt_, axis = 0)])
        workdirs_val.append(paste_nonempty(["opt_params", savename, formatted_datetime], "_"))
    
    # predict Y using surrogate

    ############################
    # Begin fig data generation
    ############################
    # For surface plot, modify the input parameters such that all are at their optimum except for 2. Sample a grid of the 2. 
    optimal = X_val.sel(workdir='optmar26f') #<== data aray (16,)
    
    # Index of parameters of interest:
    i_p3 = 8 
    i_lf = 2
    
    # Create a data array of shape nsample x 16 with all to sample for the figure. .
    nsample = 1000
    X_fig = np.tile( optimal, (nsample,1))

    # Create arrays of new samples for parameters of interest.
    # Sample randomly. 
    sample_coord = (np.linspace( 1 , nsample, nsample )).astype(int)
    p3_array = np.random.uniform( X_bounds[i_p3][0], X_bounds[i_p3][1], nsample)
    lf_array = np.random.uniform( X_bounds[i_lf][0], X_bounds[i_lf][1], nsample )
    X_fig[:,i_p3] = p3_array
    X_fig[:,i_lf] = lf_array
    
    # Scale parameters to between -1 and 1
    X_s_fig = feature_transform.fit_transform(X_fig)

    Y_pred_fig_joined = surrogate.predict(X_s_fig)
    Y_pred_fig = joinT.inverse_transform(Y_pred_fig_joined)

    # Calculate the cost for Y_pred_fig
    # calculate rmse for Y_val
    cost_val_fig, component_cost_val_fig, component_weights_fig = get_ensemble_cost(nsample, response_names, Y_pred_fig, obs, area_weights, lat, lon)

    #fig_data_out = xr.Dataset( data_vars=dict(params=(['sample_coord','input_params'],X_fig),cost=(['sample_coord'],cost_val_fig),),coords = dict(sample_coord=sample_coord,input_params=optimal.input_params.values))
    fig_data_out = xr.Dataset( data_vars=dict(params=(['sample_coord','input_params'],X_fig),cost=(['sample_coord'],cost_val_fig),),coords = dict(input_params=optimal.input_params.values))
    outpath = os.path.join(validation_dir, 'fig_data.nc')
    fig_data_out.to_netcdf( outpath )
    print( f" saved figure data to {outpath}")

    
    
    ############################
    # End fig data generation
    ############################
    
    
    Y_pred_joined = surrogate.predict(X_s)
    Y_pred = joinT.inverse_transform(Y_pred_joined)
    
    # calculate cost for Y_pred
    cost_ens, component_cost_ens, component_weights = get_ensemble_cost(nsim, response_names, Y_pred, obs, area_weights, lat, lon)
    cost_ens_dat = np.column_stack((workdirs, X, component_cost_ens, cost_ens))
    
    # predict Y_val using surrogate
    X_val_s = feature_transform.fit_transform(X_val)
    Y_val_pred_joined = surrogate.predict(X_val_s)
    Y_val_pred = joinT.inverse_transform(Y_val_pred_joined)

    # calculate rmse for Y_val
    nval = len(Y_val_pred[0])
    cost_val, component_cost_val, component_weights = get_ensemble_cost(nval, response_names, Y_val_pred, obs, area_weights, lat, lon)
    cost_val_dat = np.column_stack((workdirs_val, X_val, component_cost_val, cost_val))

    # combine all info into one array
    component_names = list(component_weights.keys())
    colnames = ['id'] + list(x_labels) + component_names + ['total_cost']
    cost_dat = np.vstack([colnames, cost_ens_dat, cost_val_dat])
    
    if not validation['modsim']['run']:
        os.makedirs(validation_output_dir)
        
        weights_path = os.path.join(validation_output_dir, paste_nonempty(["cost_weights", savename], "_") + ".csv")
        with open(weights_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(np.vstack([list(component_weights.keys()), list(component_weights.values())]))
        
        # save specs
        postprocessing.save_specs(os.path.join(validation_dir, 'specs'), workdirs, save_optimization_specs=True)
        # save environment files
        postprocessing.save_environment(os.path.join(validation_dir, 'environment'), machine)
    
    # save output
    val_surrogate_path = os.path.join(validation_output_dir, paste_nonempty(["val_surrogate", savename], "_") + ".csv")
    with open(val_surrogate_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(cost_dat)

    # create variables needed for plots 
    component_arr = np.vstack([component_cost_ens,
                               component_cost_val])
    total_cost = np.concatenate([cost_ens, cost_val])
    surr_cost = np.column_stack([component_arr, total_cost])

    # make plots
    if validation['modsim']['run']:
        if optimization['run']: # can't validate optimized solution against modsim results yet
            surr_cost_no_opt = np.delete(surr_cost, -1, axis=0)
        else:
            surr_cost_no_opt = surr_cost.copy()
        postprocessing.plot_surrogate_v_actual_cost(validation_output_dir, savename,
                                                    surr_cost_no_opt, modsim_cost, all_cost_labels, subsets)

    subsets = {'ens': list(range(len(X)))}
    idx = len(X)
    for wd in workdirs_val:
        subsets[wd] = [idx]
        idx+=1
    all_cost_labels = component_names + ['total_cost']
    input_arr = np.vstack([X, X_val])
        
    postprocessing.plot_cost_boxplots(validation_output_dir, savename + "_surrogate",
                                      surr_cost, all_cost_labels, subsets) 
    postprocessing.plot_inputs_v_cost(validation_output_dir, savename + "_surrogate",
                                      input_arr, surr_cost, x_labels, all_cost_labels, subsets)
