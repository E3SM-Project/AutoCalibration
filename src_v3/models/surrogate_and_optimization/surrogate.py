##########################
# load various packages: see requirements.txt
##########################
print("Importing packages and source code")
# various utility
import numpy as np
import yaml
import os, time, sys,copy
import joblib
from datetime import datetime
import pytz

# surrogate construction
from sklearn.decomposition import PCA
import tesuract
from tesuract.preprocessing import DomainScaler
import sklearn
from sklearn.model_selection import KFold
import multiprocessing as mp

# modules in this folder for pre/postprocessing input data and results
import preprocessing
import postprocessing
from postprocessing import paste_nonempty
import regression_params

# plots and presentation
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import csv

#############################
# load config file and data
#############################
# cfg = yaml.safe_load(open('config_surrogate.yaml'))
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

machine = cfg["machine"]
data = cfg["data"]
responses = cfg["responses"]
response_names =  [res for rf in responses.values() for res in rf]
provenance_repo = cfg["provenance_repo"]
savename = cfg["savename"]
if savename is None:
    savename = ""

if cfg["n_cores"] == "n_cores_half":
    n_cores = int(mp.cpu_count() / 2)
else:
    n_cores = cfg["n_cores"]
for i in range(len(regression_params.params["pce"])):
    regression_params.params["pce"][i]["fit_params"][0]["n_jobs"] = n_cores

surrogate_params = cfg["surrogate"]
cv_params = cfg["cv"]
sensitivity = cfg["sensitivity"]
n_components = surrogate_params['n_components']
    

##############################
# Load in simulation runs and observations
##############################
print("\nPre-processing data")
X, x_labels, X_bounds, n_inputs, Y_raw, Y_obs_raw, n_responses, normalized_area_weights, area_weights, workdirs = preprocessing.load_data(data["obs_path"], data["ens_path"], responses, surrogate_params["param_bounds"], surrogate_params["enforce_bounds"])


# Print input summary
table_input = PrettyTable()
table_input.title = f"Inputs Summary: {len(X)} Ensemble Members, {n_inputs} Inputs"
table_input.field_names = ["Input", "Min", "Max"]
for i in range(n_inputs):
    table_input.add_row([x_labels[i], X_bounds[i][0], X_bounds[i][1]])
table_input.float_format = ".6"
print(table_input)


# transform to interval [-1,1]
feature_transform = DomainScaler(dim=X.shape[1],input_range=list(X_bounds),output_range=(-1, 1))
X_s = feature_transform.fit_transform(X)


#############################
# Transform target and obs data
#############################
Y, Y_obs, joinT, Y_joined, response_transform, Y_obs_joined, scalingT, Y_s, S, W, M, mask = preprocessing.transform_data(Y_raw, Y_obs_raw, n_responses, normalized_area_weights, surrogate_params["response_transforms"], response_names, stdz_by_ncol = surrogate_params["stdz_by_ncol"], norm_method = surrogate_params["norm_method"], response_weights = surrogate_params["response_weights"])


# Print output summary
table_output = PrettyTable()
table_output.title = f"Output Summary: {len(Y_joined)} Simulations of {len(Y)} Fields"
table_output.field_names = ["Output", "ncol"]
idx_response = 0
for src_name in list(responses):
    table_output.add_row([src_name, ""])
    for rsn in list(responses[src_name]):
        table_output.add_row([rsn, Y[idx_response].shape[1]])
        idx_response += 1
    table_output.add_row(["", ""])
table_output.add_row(["Total", Y_joined.shape[1]])
print(table_output)
    

#############################
# Fit or load surrogate
#############################
reg_model = surrogate_params["method"]
if surrogate_params["fit_new"]:
    print("\nFitting new surrogate")
    
    # Create target transform pipeline
    target_transform = sklearn.pipeline.Pipeline([("response_transform", response_transform), ("scale", scalingT), ("pca", PCA(n_components=n_components, whiten=True)),])
    
    # fit surrogate model
    print(f"Selecting best {reg_model} regression parameters...")
    surrogate = tesuract.MRegressionWrapperCV(
        n_jobs=n_cores,
        regressor=[reg_model],
        reg_params=regression_params.params[reg_model],
        scorer=surrogate_params["surrogate_scorer"],
        target_transform=target_transform,
        target_transform_params={},
        verbose=0,
    )
    start = time.time()
    fit_start = time.time()
    surrogate.fit(X_s, Y_joined)

    # change auto target transform to manual for cloning
    if n_components == "auto":
        n_pc = len(surrogate.best_estimators_)
        if "pca" in target_transform.named_steps.keys():
            target_transform.set_params(pca__n_components=n_pc)
    
    # Specify best params for surrogate
    print(f"\nFitting {reg_model} regression with best parameter set")
    best_params = surrogate.best_params_
    surrogate = tesuract.MRegressionWrapperCV(
        n_jobs=-1,
        regressor=[reg_model for i in range(len(best_params))],
        reg_params=best_params,
        custom_params=True,
        target_transform=target_transform,
        target_transform_params={},
        scorer=surrogate_params["surrogate_scorer"],
        verbose=0,
    )
    
    # Fit final model with best params
    surrogate.fit(X_s, Y_joined)
    print("Total fitting time is {0:.3f} seconds".format(time.time() - fit_start))
    
    # Create new surrogate_provenance_yyyymmddhhmmss folder
    current_datetime = datetime.now(pytz.timezone("US/Mountain"))
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    
    savedir = os.path.join(provenance_repo,
                           paste_nonempty(["surrogate_provenance", savename, formatted_datetime], "_"))
        
    # save model
    print("Saving model to {0}\n".format(savedir))
    surrogate_output_dir = os.path.join(savedir, "surrogate_fit", "output")
    os.makedirs(surrogate_output_dir)
    joblib.dump(surrogate, os.path.join(surrogate_output_dir,
                                        paste_nonempty(["model", savename], "_") + ".joblib"))
    
    # save PC information
    postprocessing.save_pc_info(surrogate_output_dir, savename, X_s, surrogate, surrogate.target_transform, n_components, Y_joined, Y_obs_joined)
    
    # save surrogate predictions
    postprocessing.save_predictions(surrogate_output_dir, savename, X, Y_joined, feature_transform, surrogate, M, workdirs)
    
    # save specs
    surrogate_specs_dir = os.path.join(savedir, 'surrogate_fit', 'specs')
    postprocessing.save_specs(surrogate_specs_dir, workdirs, save_surrogate_specs=True)
        
    # save pertinent environment files
    surrogate_environment_dir = os.path.join(savedir, 'surrogate_fit', 'environment')
    #postprocessing.save_environment(surrogate_environment_dir, machine)
else:
    savedir = os.path.join(provenance_repo, surrogate_params["provenance_dir"])
    print("Loading model from {0}".format(savedir))
    surrogate_model_file = os.path.join(savedir, surrogate_params["model_load_file"])
    surrogate = joblib.load(surrogate_model_file)
    
    # get datetime to use in filenames
    current_datetime = datetime.now(pytz.timezone("US/Mountain"))
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    
    
#############################
# Run cross-validation
#############################
if cv_params["run"]:
    print("\nComputing CV scores")

    # set up cross-validation folds
    nsplits = cv_params["nsplits"]
    nsim_all = len(Y_s)
    if cv_params["nsim"] == "all":
        nsim = nsim_all
    else:
        nsim = cv_params["nsim"]
        
    folds = []
    cv_method = cv_params["method"]
    if cv_method == "kfold":
        if nsim < nsim_all:
            print("\nWARNING! Some data will be ignored across all folds since method='kfold' and nsim!='all'")
        kf = KFold(n_splits=nsplits)
        for train_idx, test_idx in kf.split(range(nsim)):
            folds.append((train_idx, test_idx))
    elif cv_method == "random":
        # find sims that meet the condition to be in the test set 
        X_condition = X_bounds.copy()
        if cv_params["test_condition"] is not None:
            for xlab in cv_params["test_condition"]:
                X_condition[list(x_labels).index(xlab),:] = cv_params["test_condition"][xlab]
        idx_condition = preprocessing.get_idx_keep(X, X_condition) 
            
        ntest = cv_params["ntest"]
        if ntest is None:
            print("WARNING! ntest is None. Must specify ntest to use method=random")
        for k in range(nsplits):
            include_idx = np.random.choice(nsim_all, size=nsim, replace=False)
            pool_idx = np.array(list(set(include_idx) and set(idx_condition)))
            npool = len(pool_idx) 
            if npool >= ntest: 
                test_idx = pool_idx[np.random.choice(npool, size=ntest, replace=False)]
            else:
                print(f"WARNING! Less than ntest samples meet the test_condition. Using {npool} samples instead")
            train_idx = np.array([i for i in include_idx if i not in test_idx])
            folds.append((train_idx, test_idx))
            
    full_cv = cv_params["full_cv"]
    if full_cv: # reset pca so that we can re-compute for every training set
        # (re)-create target transform pipeline
        pass # not implemented yet

    # Run cv on transformed and standardized responses Y_s
    cv_scores_all = postprocessing.compute_cv_score(surrogate, X=X_s, y=Y_s, regressor=reg_model, folds=folds, prefit_params=not full_cv, n_jobs = n_cores)

    # save output
    cv_dir = os.path.join(savedir, "cv", paste_nonempty(["cv", savename, formatted_datetime], "_"))
    cv_output_dir = os.path.join(cv_dir, "output")
    os.makedirs(cv_output_dir)

    cv_scores_zipped = list(zip(*[cv_scores_all[key] for key in cv_scores_all]))
    cv_score_output_file = os.path.join(cv_output_dir, paste_nonempty(["cv", savename], "_") + ".csv")
    with open(cv_score_output_file, 'w', newline='') as f:
        score_type = cv_scores_all.keys()
        writer = csv.DictWriter(f, fieldnames=score_type)
        writer.writeheader()
        for score in cv_scores_zipped:
            writer.writerow(dict(zip(score_type, score)))

    # save specs
    postprocessing.save_specs(os.path.join(cv_dir, 'specs'), workdirs, save_surrogate_specs=True)

    # save environment files
    #postprocessing.save_environment(os.path.join(cv_dir, 'environment'), machine)
        
    cv_score_mean = {k: np.mean(v) for k, v in cv_scores_all.items()}
        
        
# Print model summary - TODO: include additional surrogate info 
table_model = PrettyTable()
table_model.title = "Model Summary"
table_model.field_names = ["Type", "Value"]
if surrogate_params["fit_new"] or cv_params["run"] or sensitivity['run'] or history_matching['run'] or calibration['run'] or cfg['validation']['surrogate']:
    table_model.add_row(["# Components", n_components])
    prop_var_explained = surrogate.target_transform["pca"].explained_variance_ratio_.sum()
    table_model.add_row(["Prop Var Explained", prop_var_explained])
    table_model.add_row(["Regression method", reg_model])
    if reg_model == "pce":
        table_model.add_row(["Max Order", regression_params.params['pce'][0]['order'][-1]])
        table_model.add_row(["Order", [surrogate.best_params_[k]['order'] for k in range(len(surrogate.best_params_))]])
if cv_params["run"]:
    table_model.add_row(["Full CV", full_cv])
    table_model.add_row(["CV Method", cv_method])
    table_model.add_row(["nsim", nsim])
    table_model.add_row(["nsplits", nsplits])
    if cv_method=="random":
        table_model.add_row(["ntest", ntest])
    for key in cv_score_mean:
        table_model.add_row([key, cv_score_mean[key]])
table_model.float_format = ".6"
print(table_model)


    
#############################
# Sensitivity Analysis
#do we need this?
#############################
if sensitivity['run'] and (sensitivity['plot'] or sensitivity['save']):
    print("\nRunning Sensitivity Analysis")
    
    # for each output variable, compute global Sobol' indices using surrogate fit
    pc_sobol = surrogate.feature_importances_ # must compute before next line can run
    total_sobol = surrogate.sobol_weighted_ # combined sobol indicies across pcs (weighted by eigenvalue)
    
    # save output
    sensitivity_dir = os.path.join(savedir, "sensitivity", paste_nonempty(["sensitivity", savename, formatted_datetime], "_"))
    sensitivity_output_dir = os.path.join(sensitivity_dir, "output")
    os.makedirs(sensitivity_output_dir)

    if sensitivity['plot']:
        # for each input parameter (x-axis), plot total Sobol' index.
        plt.figure(figsize=(8, 6))
        plt.bar(x_labels, total_sobol)

        plt.xticks(rotation=90)
        plt.xlabel('Input Parameter')
        plt.ylabel("Total Sobol' Index")
        plt.title("Combined Sensitivity Across Output Fields")
        
        plot_savename = paste_nonempty(["sensitivity", savename], "_") + ".png"
        plt.savefig(os.path.join(sensitivity_output_dir, plot_savename), bbox_inches='tight')
        plt.close()

    if sensitivity['save']:
        # save a small .csv file with a single column containing total sobol' index
        sobol_dat = np.hstack((np.vstack(("input_parameter", np.reshape(x_labels, (-1, 1)))),
                               np.vstack(("total_sobol", np.reshape(total_sobol, (-1, 1))))))
        
        sensitivity_output_file = os.path.join(sensitivity_output_dir, paste_nonempty(["sensitivity", savename], "_") + ".csv")
        with open(sensitivity_output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(sobol_dat)
    
    # save specs
    postprocessing.save_specs(os.path.join(sensitivity_dir, 'specs'), workdirs, save_surrogate_specs=True)

    # save environment files
    #postprocessing.save_environment(os.path.join(sensitivity_dir, 'environment'), machine)
    
    
