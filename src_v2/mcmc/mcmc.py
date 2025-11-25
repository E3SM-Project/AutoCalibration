"""

Purpose: Implement emcee mcmc method

"""
import sys
import time
import scipy
import emcee
import tesuract
import numpy as np
import os, pickle, datetime

import matplotlib.pyplot as plt
from functools import partial
from hashlib import blake2b
from joblib import dump, load
import multiprocessing
from multiprocessing import Pool
from sklearn.metrics import make_scorer
from joblib import Parallel, delayed
from scipy.optimize import minimize
from multiprocessing import cpu_count

from functions import my_scorer, log_prob, function_2d, take_mean_of_walkers, log_like, log_prior_x, log_prior_tau
from plots import plot_bayes_corner_plot_range, plot_bayes_univar_dist, plot_bayes_corner_plot, plot_multi_scatter, plot_trace_for_param

import xarray as xr

import sys
sys.path.insert(0, '../models/')
import preprocessing, postprocessing

'''
Notes:
------

* Do NOT set the random seed, i.e. np.random.seed(seed) as it will mess up the emcee chains (I think)

'''

# global vars
thin = 10
nwalkers = 200
estimate_type = 'mle'
mle_sigma = .5

################################################
# setup variables for extraction
################################################
# command line args
parallel_mode = 1 #int(sys.argv[3])   # serial=0 or parallel=1
################################################
# extract data
################################################
import yaml
from prettytable import PrettyTable
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

n_burn= cfg["n_burn_in"]
n_sample = cfg["n_samples"]

mcmc_sampler = cfg["calibration"]["Bayesian"]['MCMC'] #int(sys.argv[2])    # re-use existing=0 or fit new=1
season = cfg["season"]
nyear = cfg["nyear"]
datadir = cfg["datadir"]
savedir = cfg["savedir"]
lat_lon_fields = cfg["fields"]
nlat_lon_fields = len(lat_lon_fields)

lat_plev_fields = cfg["lat_plev_fields"]
nlat_plev_fields = len(lat_plev_fields)

global_fields = cfg["global_fields"]
nglobal_fields = len(global_fields)
RESTOM_target = cfg["RESTOM_target"]

target_type = cfg["target_type"]
surrogate_params = cfg["surrogate"]["options"]
opt_params = cfg["calibration"]["optimize"]
resolution = cfg["resolution"]
n_components = surrogate_params['n_components']

target_source = cfg['target_source']
if target_source is None:
    target_source = 'obs'

if cfg['subtract_ens_mean']:
    target_centering = 'ensmean'
else:
    target_centering = 'raw'

Y_raw, Y_obs_raw, nlat_lon_fields, normalized_area_weights, area_weights = preprocessing.load_lat_lon_data(season, nlat_lon_fields, lat_lon_fields, nyear, resolution, target_source, datadir)

Y_raw_plev, Y_obs_raw_plev, nlat_plev_fields, plev_mask, normalized_area_weights_plev, area_weights_plev = preprocessing.load_lat_plev_data(season, nlat_plev_fields, lat_plev_fields, nyear, resolution, target_source, datadir)

Y_raw_global, Y_obs_raw_global = preprocessing.load_global_data(global_fields, RESTOM_target, nyear, datadir)

if nlat_lon_fields == 0:
    field_str_list = "_".join(lat_plev_fields)
elif nlat_plev_fields == 0:
    field_str_list = "_".join(lat_lon_fields)
else:
    field_str_list = "_".join(["_".join(lat_lon_fields), "_".join(lat_plev_fields)]) 
if len(global_fields) > 0:
    if 'RESTOM' in global_fields:
        field_str_list = "_".join([field_str_list, 'RESTOM', str(float(RESTOM_target))])

filename = f"{season}_{nyear}yr_{target_source}_{target_type}sqrtnonly_{resolution}_{n_components}_{field_str_list}_{target_centering}_{n_burn}_{n_sample}"
model_filename = "model_" + filename + ".joblib"
if os.path.exists( os.path.join(savedir, filename)) == False:
    os.mkdir(filename)
full_save_path = os.path.join(savedir, filename, model_filename)
filename_save = filename
# printing summary of experiment...
table_setup = PrettyTable()
table_setup.field_names = ["Season", "Sim. length", "lat/lon fields",  "lat/plev fields", "global fields",  "Target"]
table_setup.add_row([season, nyear, lat_lon_fields, lat_plev_fields, global_fields, cfg["target_type"]])
table_setup.title = "E3SM Calibration Experiment"
print(table_setup)
from tesuract.preprocessing import DomainScaler

#############################
# Load feature data (input)
#############################
X_xr, X_bounds = preprocessing.load_input_parameters(datadir)
X = X_xr.values
x_labels = X_xr["x"].values
X_bounds_label =[tuple(X_bounds[0,:]),
        tuple(X_bounds[1,:]),
        tuple(X_bounds[2,:]),
        tuple(X_bounds[3,:]),
        tuple(X_bounds[4,:])]

feature_transform = DomainScaler(dim=X.shape[1],input_range=list(X_bounds),output_range=(-1, 1))
X_s = feature_transform.fit_transform(X)
feature_dim = X_s.shape[1]

default_params = np.array([500.0, 2.40, 0.12, 3600, -0.0007])
x_s_default = feature_transform.fit_transform(default_params)

#############################
# Transform target and obs data
#############################

Y, Y_obs, S, W, W_plev, scalar_function, scalar_function_plev, surrogate_scorer, plev_mask_flatten = preprocessing.transform_data(Y_raw, Y_raw_plev, Y_raw_global, Y_obs_raw, Y_obs_raw_plev, Y_obs_raw_global,
        nlat_lon_fields, nlat_plev_fields, global_fields, area_weights, area_weights_plev, normalized_area_weights, normalized_area_weights_plev, cfg)


#############################
# Join transform on the data
#############################
joinT = preprocessing.JoinTransform()
Y_joined = joinT.fit_transform(Y)
Y_obs_joined = joinT.transform(Y_obs)
sigmas_joined = joinT.transform(S)

scalingT = preprocessing.SigmaScaling(sigma=sigmas_joined)

Y_s = scalingT.fit_transform(Y_joined)
Y_obs_s = scalingT.fit_transform(Y_joined)

#############################
# Create target transform pipeline
#############################
import sklearn
n_components = cfg["surrogate"]["options"]["n_components"]
from sklearn.decomposition import PCA

if cfg["target_type"] == "full" or cfg["target_type"] == "zonal":
    target_transform = sklearn.pipeline.Pipeline([("scale", preprocessing.SigmaScaling(sigma=sigmas_joined)),("pca", PCA(n_components=n_components, whiten=True)),])
elif cfg["target_type"] == "scalar":
    target_transform = sklearn.pipeline.Pipeline(
        [
            ("scale", preprocessing.SigmaScaling(sigma=sigmas_joined)),
        ]
    )

import multiprocessing as mp

#############################
# Fit surrogate with ROM + PCE
#############################
import joblib

if cfg["surrogate"]["fit"]:
    # select method from config
    reg_model = surrogate_params["method"]
    print(f"Fitting with {reg_model}-based regression...")
    # fit surrogate model
    surrogate = tesuract.MRegressionWrapperCV(
        n_jobs=preprocessing.n_cores_half,
        regressor=[reg_model],
        reg_params=[preprocessing.model_grid_dict[reg_model]],
        scorer=surrogate_scorer,
        target_transform=target_transform,
        target_transform_params={},
        verbose=0,
    )
    start = time.time()
    print("\nFitting surrogate...")
    fit_start = time.time()
    surrogate.fit(X_s, Y_joined)
    print("Total fitting time is {0:.3f} seconds".format(time.time() - fit_start))

    # change auto target transform to manual for cloning
    if n_components == "auto":
        n_pc = len(surrogate.best_estimators_)
        if "pca" in target_transform.named_steps.keys():
            target_transform.set_params(pca__n_components=n_pc)
    # Clone and compute the cv score of full model
    print("\nComputing CV score and final fit...")

    cv_score, surrogate = preprocessing.compute_cv_score_multiple(surrogate,
        X=X_s, y=Y_joined, scoring=["r2", "neg_root_mean_squared_error", "neg_median_absolute_error"], regressor = reg_model,  target_transform=target_transform
    )

    # Fit final, simplified model
    surrogate.fit(X_s, Y_joined)
    # save model
    joblib.dump(surrogate, os.path.join(savedir, filename, 'surrogate.joblib'))
    print("Saving model to {0}...\n".format(os.path.join(savedir, filename, 'surrogate.joblib')))

else:
    print("Loading model from {0}...".format(os.path.join(savedir, filename, 'surrogate.joblib')))
    surrogate = joblib.load(os.path.join(savedir, filename, 'surrogate.joblib'))
    cv_score = ""
    if surrogate_params["compute_cv_score"]:
        print("\nComputing CV score...")
        preprocessing.compute_cv_score_multiple(self = surrogate,X=X_s, y=Y_joined, scoring={"r2", "neg_root_mean_squared_error", "neg_median_absolute_error"}, regressor = reg_model)
    if surrogate_params["compute_raw_cv_scores"]:
        surrogate_clone = surrogate.clone()
        kf = KFold(n_splits=5)
        all_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X_s[train_index], X_s[test_index]
            Y_train, Y_test = Y_joined[train_index], Y_joined[test_index]
            surrogate_clone.fit(X_train, Y_train)
            Y_pred = surrogate_clone.predict(X_test)
            scores = sklearn.metrics.r2_score(Y_test, Y_pred, multioutput="raw_values")
            scores[scores < 0] = 0.0
            all_scores.append(scores)
        raw_cv_scores = np.mean(all_scores, axis=0)

## define functions
import optimization as opt

#default_rmses = surrogate.predict(x_s_default)
#S_new = default_rmses[0]
#S_old = S
#S = S_new

sigma_train = opt.compute_prior_sigmas(Y, Y_obs, S, nlat_lon_fields, nlat_plev_fields, W=W, W_plev = W_plev)
logsigma_train = np.log(sigma_train)
raw_sigmas = opt.compute_prior_sigmas(Y, Y_obs, S, nlat_lon_fields, nlat_plev_fields, W=W, W_plev = W_plev, return_raw=True)
raw_logsigma_train = [np.log(r) for r in raw_sigmas]


#############################
# custom weights for MLE calculation
#############################
sigma_train, logsigma_train = opt.process_custom_weights(opt_params, season, nlat_lon_fields, nlat_plev_fields, nglobal_fields, lat_lon_fields, lat_plev_fields, global_fields, sigma_train, logsigma_train)


params = {'Y_obs' : Y_obs, 'ntot_fields' : nlat_lon_fields + nlat_plev_fields + nglobal_fields,
        'nll_fields': nlat_lon_fields, 'nlp_fields': nlat_plev_fields, 'ng_fields': nglobal_fields,
        'W': W, 'W_plev': W_plev, 'feature_transform': feature_transform, 'surrogate': surrogate,
        'feature_dim': 5, 'joinT': joinT, 'S': S, 'raw_logsigma_train': raw_logsigma_train, 'logsigma_train': logsigma_train}



###################
if opt_params["do_MAP"]:
    opt_type, xopt_, xopt_s, sigma_opt, mse_func = opt.optimize_MAP(params, opt_params)
    xopt_map = xopt_

    postprocessing.print_opt_results(
        xopt_,
        x_labels,
        opt_type,
        2*preprocessing.n_cores_half,
        feature_dim,
        cv_score,
        solver="L-BFGS-B",
        filename=os.path.join(savedir, filename, "opt_MAP.txt"),
    )
    np.save(os.path.join(savedir, filename, 'opt_MAP.npy'), xopt_map)
else:
    xopt_map = np.load(os.path.join(savedir, filename, 'opt_MAP.npy'))

if opt_params["do_MLE"]:
    opt_type, xopt_, xopt_s, sigma_opt, mse_func = opt.optimize_MLE_msigma(params, opt_params)
    xopt_mle = xopt_

    postprocessing.print_opt_results(
        xopt_,
        x_labels,
        "MLE",
        2*preprocessing.n_cores_half,
        feature_dim,
        cv_score,
        solver="L-BFGS-B",
        filename=os.path.join(savedir, filename, "opt_MLE.txt"),
    )
    np.save(os.path.join(savedir, filename, 'opt_MLE.npy'), xopt_mle)
else:
    xopt_mle = np.load(os.path.join(savedir, filename, 'opt_MLE.npy'))

################################################
# MLE test
################################################
def logprob_single(xin, params):
    x, logsigma = xin[:params['feature_dim']], xin[params['feature_dim'] : params['feature_dim'] + params['ntot_fields']]
    return opt.logprob(x, logsigma, params)

rn = np.random.RandomState(42389)
filename_sampler = 'emcee_sampler.h5'

ndim = 5
nfields = sigma_train.size
if mcmc_sampler:
    # prepare h5 backend
    backend = emcee.backends.HDFBackend(os.path.join(savedir, filename, filename_sampler))
    backend.reset(nwalkers=nwalkers, ndim=ndim+nfields)

    # start with 50/50 lhs and random samples
    randperm = rn.permutation(X_s.shape[0])[:int(nwalkers/2)]
    p0_1 = np.array([np.append(Xi_s,np.log(sigma_train*rn.rand())) for Xi_s in X_s[randperm]])

    # intiailization p0
    p0_2 = rn.uniform(-1, 1, (int(nwalkers/2), ndim+nfields))
    p0_2[:,ndim:] = np.log(sigma_train*.5*(1 + p0_2[:,ndim:]))

    p0 = np.vstack([p0_1,p0_2])

    log_prob_clone = logprob_single #partial(log_prob, y=Y_ref, surrogate=surrogate)
    if parallel_mode:
        # with Pool() as pool:
        with multiprocessing.get_context('fork').Pool() as pool:
            # instantiate sampler class with pool
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim+nfields,
                log_prob_clone,
                args = [params],
                pool=pool,
                backend=backend,
                moves=emcee.moves.StretchMove(a=2.0))

            # burn-in
            state = sampler.run_mcmc(p0, n_burn, progress=True)
            sampler.reset()

            # sampling
            start = time.time()
            sampler.run_mcmc(state, n_sample, progress=True)
            end = time.time()
            time_diff = end - start
            print("Parallel processing took {0:.1f} seconds".format(time_diff))

    else:
        # instantiate sampler class
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim+1,
            log_prob_clone,
            backend=backend)

        # burn-in
        state = sampler.run_mcmc(p0, n_burn, progress=True)
        sampler.reset()

        # sampling
        start = time.time()
        sampler.run_mcmc(state, n_sample, progress=True)
        end = time.time()
        time_diff = end - start
        print("Serial processing took {0:.1f} seconds".format(time_diff))

else:
    # retrieve samper object
    sampler = emcee.backends.HDFBackend(os.path.join(savedir, filename, filename_sampler))

################################################
# query convergence of emcee sampler
################################################
# extract samples
samples = sampler.get_chain(flat=True,thin=thin)
# extract log probabilities from sampler
log_probs = sampler.get_log_prob(flat=True,thin=thin)
samples_x = feature_transform.inverse_transform(samples[:,:-nfields])
samples_thinned = samples.copy()
samples_thinned[:,:-nfields] = samples_x
mcmc = 'emcee'
np.save(#"{}samples.npy".format(
    #filename_save
    os.path.join(savedir, filename, 'samples.npy'), samples_thinned)

# mean acceptance rate
if mcmc_sampler:
    mean_acceptance_fraction = np.mean(sampler.acceptance_fraction)
else:
    mean_acceptance_fraction = np.mean(sampler.accepted/float(sampler.iteration))
print(
    "Mean acceptance fraction: {0:.3f}".format(
        mean_acceptance_fraction
    )
)

# mean autocorrelation time
mean_autocorrelation_time = np.mean(sampler.get_autocorr_time(tol=0))
print(
    "Mean autocorrelation time: {0:.3f} steps".format(
        mean_autocorrelation_time
    )
)

# autocorrelation progression over sampling
samples_averaged = take_mean_of_walkers(samples=sampler.get_chain(flat=True,thin=1), nwalkers=nwalkers)
autocorr = function_2d(samples=samples_averaged)
plot_multi_scatter(data=autocorr, plot_dir=os.path.join(savedir, filename))

if mcmc_sampler:
    with open(os.path.join(savedir, filename, 'convergence_statistics.text'), 'w') as f:
        f.write('total sampling time: {0:.1f} seconds, time per sample {0:.1f} seconds\n'.format(time_diff, time_diff/n_sample))
        f.write('mean acceptance fraction: {0:.3f}\n'.format(mean_acceptance_fraction))
        f.write('mean autocorrelation time: {0:.3f}\n'.format(mean_autocorrelation_time))

################################################
# max a posteriori estimation (MAP)
################################################

# find index of maximum
max_idx = np.argmax(log_probs)

# max a posterior estimation
bayes_map = samples_thinned[max_idx, :]
print("Maximum a posteriori estimation (thinned): \n{}".format(
    np.array_str(bayes_map,precision=4)
    )
)

# get map with unthinned samples
def get_unthinned_map():
    samples = sampler.get_chain(flat=True,thin=1)
    samples_x = feature_transform.inverse_transform(samples[:,:-nfields])
    log_probs = sampler.get_log_prob(flat=True,thin=1)
    samples_unthinned = samples.copy()
    samples_unthinned[:,:-nfields] = samples_x
    map_argmax = np.argmax(log_probs)
    bayes_map = samples_unthinned[map_argmax,:]
    print("Maximum a posteriori estimation (unthinned): \n{}".format(
        np.array_str(bayes_map,precision=4)
        )
    )
    return bayes_map
bayes_map = get_unthinned_map()

################################################
# plot samples for each param
################################################
# matplotlib setup
font = {'size': 16}
plt.rc('font', **font)

# univariate distributions
for i, x in enumerate(X_xr.coords['x'].values):
    plot_bayes_univar_dist(
        xindex=i,
        xlabel=x,
        plot_dir=os.path.join(savedir, filename),
        samples=samples_thinned[:, i],
        bayes_map=bayes_map[i],
        log_prob_mle=xopt_mle[i]
    )

with open(os.path.join(savedir,filename, 'param_calibration.txt'), 'w') as f:
    f.write('maximum likelihood estimation (mle): {}\n'.format(xopt_mle))
    f.write('maximum a posteriori estimation (map): {}\n'.format(bayes_map))



# corner plot (covariance)
plot_bayes_corner_plot(
        samples=samples_thinned[:,0:5],
    plot_dir = os.path.join(savedir, filename),
    truths = bayes_map[0:5],
    control = default_params,
    labels=X_xr.coords['x'].values
)

plot_bayes_corner_plot_range(
        samples=samples_thinned[:,0:5],
    plot_dir=os.path.join(savedir, filename),
    ranges = X_bounds_label,
    truths = bayes_map[0:5],
    control = default_params,
    labels=X_xr.coords['x'].values
)


################################################
# plot trace for each param
################################################
# univariate distributions
params = X_xr.coords['x'].values
for i, x in enumerate(params):
    samples_x = samples_thinned[:, i]
    samples_reshaped_x = samples_x.reshape(nwalkers, -1).T
    plot_trace_for_param(
        xindex=i,
        xlabel=x,
        plot_dir=os.path.join(savedir, filename),
        samples_x=samples_reshaped_x
    )

