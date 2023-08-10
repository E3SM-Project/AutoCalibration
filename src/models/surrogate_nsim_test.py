import numpy as np
from sklearn.metrics import make_scorer
import xarray as xr
import yaml
import os, time, sys
import clif.preprocessing as cpp
import joblib

import tesuract
import preprocessing
import postprocessing
from tesuract.preprocessing import DomainScaler
from tesuract.preprocessing import PCATargetTransform
import sklearn

from scipy.optimize import minimize
from functools import partial
import multiprocessing as mp

import matplotlib.pyplot as plt

from prettytable import PrettyTable
from tqdm import tqdm

from sklearn.model_selection import KFold
import pdb

#############################
# load config file and data
#############################
# make config2.yaml a system argument
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

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
RESTOM_target = cfg['RESTOM_target']

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
nsim_test_grid = [25, 50, 75, 100, 125, 150, 175, 200, 225]
import os
ntest_grid = len(nsim_test_grid)
job_array = int(os.environ['SLURM_ARRAY_TASK_ID'])

rseed = job_array
ntrain = nsim_test_grid[job_array % ntest_grid]
filename = f"{season}_{nyear}yr_{target_type}_{resolution}_{n_components}_{field_str_list}_{target_centering}_" + str(ntrain) + '_' + str(int( (job_array - job_array % ntest_grid) / ntest_grid))


# printing summary of experiment...
table_setup = PrettyTable()
table_setup.field_names = ["Season", "Sim. length", "lat/lon Fields","lat/plev Fields", "Target"]
table_setup.add_row([season, nyear, lat_lon_fields, lat_plev_fields, cfg["target_type"]])
table_setup.title = "E3SM Calibration Experiment"
print(table_setup)

#############################
# Load feature data (input)
#############################
X_xr, X_bounds = preprocessing.load_input_parameters(datadir)
X = X_xr.values
x_labels = X_xr["x"].values

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

import preprocessing

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

#############################
# Fit surrogate with ROM + PCE
#############################
import random
random.seed(rseed)
train_index = random.sample(range(250), ntrain)
X_train = X_s[train_index,:]
X_test = X_s[list(set(range(250)).difference(train_index)),:]
Y_train = Y_joined[train_index,:]
Y_test = Y_joined[list(set(range(250)).difference(train_index)),:]

# fit surrogate model
surrogate = tesuract.MRegressionWrapperCV(
    n_jobs=preprocessing.n_cores_half,
    regressor=["pce"],
    reg_params=[preprocessing.pce_grid],
    scorer=surrogate_scorer,
    target_transform=target_transform,
    target_transform_params={},
    verbose=0,
)
start = time.time()
print("\nFitting surrogate...")
fit_start = time.time()
surrogate.fit(X_train, Y_train)
print("Total fitting time is {0:.3f} seconds".format(time.time() - fit_start))

# change auto target transform to manual for cloning
if n_components == "auto":
    n_pc = len(surrogate.best_estimators_)
    if "pca" in target_transform.named_steps.keys():
        target_transform.set_params(pca__n_components=n_pc)
# Clone and compute the cv score of full model
print("\nComputing CV score and final fit...")

cv_score, surrogate = preprocessing.compute_cv_score_multiple(surrogate,
    X=X_train, y=Y_train, scoring={"r2", "neg_root_mean_squared_error", "neg_median_absolute_error"}, target_transform=target_transform
)

# Fit final, simplified model
surrogate.fit(X_train, Y_train)

cv_score = dict(sorted(cv_score.items()))
# printing model summary...
table_model = PrettyTable()
table_model.title = "ROM-based ML Model Summary"
table_model.field_names = ["Type", "Value"]
table_model.add_row(["Input size", f"({feature_dim},)"])
table_model.add_row(["# Components", f"({n_components},)"])
table_model.add_row(["Output size", f"({Y_joined.shape[1]},)"])

for key in cv_score:
    table_model.add_row([key, cv_score[key]])
table_model.float_format = ".6"
print(table_model)
#############################
# Generating starting point for log likelihood sigmas
#############################
import optimization as opt
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
# MAP optimization
###################
if opt_params["method"] == "MAP":
    opt_type, xopt_, xopt_s, sigma_opt, mse_func = opt.optimize_MAP(params, opt_params)

###################
# MLE optimization
###################
if opt_params["method"] == "MLE":
    opt_type, xopt_, xopt_s, sigma_opt, mse_func = opt.optimize_MLE_msigma(params, opt_params)

###################
# MLE spatial var optimization
###################
if opt_params["method"] == "MLE_msigma":
    opt_type, xopt_, xopt_s, sigma_opt, mse_func = opt.optimize_MLE_msigma(params, opt_params)

###################
# save and plot calibration results
###################

# save scaled optimal solution
if opt_params["method"] is not None:
    print("Saving optimal in {0}...".format(f"opt_{filename}_{opt_type}.npy"))
    np.save(os.path.join(savedir, f"opt_{filename}_{opt_type}.npy"), xopt_)

    postprocessing.print_opt_results(
        xopt_,
        x_labels,
        opt_type,
        2*preprocessing.n_cores_half,
        feature_dim,
        cv_score,
        solver="L-BFGS-B",
        filename=os.path.join(savedir, f"opt_{filename}_{opt_type}.txt"),
    )
    with(open(f"{savedir}/opt_{filename}_{opt_type}_sigmas.txt", 'w') as f):
        f.write(str(sigma_opt))

    if opt_params["options"]["plot_diagnostics"]:
        postprocessing.plot_results(
            mse_func,
            X_s,
            xopt_s,
            x_s_default,
            opt_type,
            filename=os.path.join(savedir, f"plots/diagnostics_{filename}_{opt_type}.png"),
        )

###################
# validating the optimal estimates
###################
if cfg["validation"]["surrogate"]:
    xopt_sur_errors, default_sur_errors, seasons, fields = postprocessing.compute_validation(filename, savedir, opt_type, season, lat_lon_fields, lat_plev_fields, global_fields, xopt_s, x_s_default, Y_obs_raw, Y_obs_raw_plev, params, target_type, surrogate, plev_mask)

    postprocessing.plot_validation(xopt_sur_errors, default_sur_errors, seasons, fields, params, season, target_type, opt_type,
        field_str_list, savedir)

