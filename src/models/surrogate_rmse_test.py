import numpy as np
from sklearn.metrics import make_scorer
import xarray as xr
import yaml
import os, time, sys
import clif.preprocessing as cpp
import joblib

import tesuract
import postprocessing
from tesuract.preprocessing import DomainScaler
from tesuract.preprocessing import PCATargetTransform
import sklearn

from scipy.optimize import minimize
from functools import partial
import multiprocessing as mp

import matplotlib.pyplot as plt

from prettytable import PrettyTable

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
fields = cfg["fields"]
nfields = len(fields)
target_type = cfg["target_type"]
surrogate_params = cfg["surrogate"]["options"]
opt_params = cfg["calibration"]["optimize"]
resolution = cfg["resolution"]

dataset = xr.open_dataset(
    os.path.join(datadir, f"lat_lon_{nyear}yr_{resolution}_{season}.nc")
)
dataset_ref = xr.open_dataset(
    os.path.join(datadir, f"lat_lon_{resolution}_{season}_obs.nc")
)

field_str_list = "_".join(fields)
filename = f"{season}_{nyear}yr_{target_type}_{field_str_list}"
model_filename = "model_" + filename + ".joblib"
full_save_path = os.path.join(savedir, model_filename)

# printing summary of experiment...
table_setup = PrettyTable()
table_setup.field_names = ["Season", "Sim. length", "Field(s)", "Target"]
table_setup.add_row([season, nyear, fields, cfg["target_type"]])
table_setup.title = "E3SM Calibration Experiment"
print(table_setup)

#############################
# Load feature data (input)
#############################
X_xr = dataset["lhs"]
X = X_xr.values
x_labels = X_xr["x"].values
X_bounds = dataset["lhs_bnds"].values

feature_transform = DomainScaler(
    dim=X.shape[1],
    input_range=list(X_bounds),
    output_range=(-1, 1),
)
X_s = feature_transform.fit_transform(X)
feature_dim = X_s.shape[1]

default_params = np.array([500.0, 2.40, 0.12, 3600, -0.0007])
x_s_default = feature_transform.fit_transform(default_params)

#############################
# Transform target and obs data
#############################
Y_raw = [dataset[f] for f in fields]  # simulation data
Y_obs_raw = [dataset_ref[f] for f in fields]  # obs data
area_weights = dataset.area[0]  # lat lon area weights
normalized_area_weights = area_weights / area_weights.sum()

# define transforms for full field, zonal and scalar outputs
flatten_transform = cpp.FlattenData(dims=["lat", "lon"])
zonal_mean_transform = cpp.MarginalizeOutTransform(
    dims=["lon"], lat_lon_weights=area_weights
)
mse_transform = cpp.MarginalizeOutTransform(
    dims=["lat", "lon"], lat_lon_weights=area_weights
)

# compute an xarray data array of scalers
Sigmas = [np.sqrt(Yi.var()) + 0 * Yi for Yi in Y_obs_raw]

if cfg["target_type"] == "full":
    Y_obs = [flatten_transform.fit_transform(Yi) for Yi in Y_obs_raw]
    Y = [flatten_transform.fit_transform(Yi) for Yi in Y_raw]
    S = [flatten_transform.fit_transform(s) for s in Sigmas]
    W = normalized_area_weights.values.flatten()

if cfg["target_type"] == "zonal":
    Y_obs = [zonal_mean_transform.fit_transform(Yi) for Yi in Y_obs_raw]
    Y = [zonal_mean_transform.fit_transform(Yi) for Yi in Y_raw]
    S = [zonal_mean_transform.fit_transform(s) for s in Sigmas]
    weights = zonal_mean_transform.fit_transform(area_weights).values
    W = weights / weights.sum()


# define scalar error
def scalar_function(Y, Y_obs):
    # Y and Y_obs should be the same shape
    return np.sqrt(mse_transform.fit_transform((Y - Y_obs) ** 2))


# load custom scalar function
# will overwrite scalar_function above
if cfg["custom_scalar_function"]["file"] is not None:
    print("\nLoading scalar function...")
    cwd = os.path.dirname(__file__)
    full_path = os.path.join(cwd, "custom_scalar_function.py")
    exec(open(full_path).read())
    # overwrite function above with custom
    scalar_function = custom_scalar_function

if cfg["target_type"] == "scalar":
    # computes the root mean square error per simulation (skill score), not to be confused with RMSE per grid point
    Ytemp = [
        scalar_function(Yi, Y_obs_raw[i]).expand_dims(dim="new", axis=1)
        for i, Yi in enumerate(Y_raw)
    ]
    Y = [Yi for Yi in Ytemp]  # compute Root mean square error
    Y_obs = [xr.DataArray(0)] * nfields  # Y_obs = 0
    S = [s.mean(dim=["lat", "lon"]) ** 1 for s in Sigmas]
    W = np.ones(1)


# convert xarray to numpy arrays
Y = [Yi.values for Yi in Y]
Y_obs = [Yi.values for Yi in Y_obs]
S = [s.values for s in S]

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

if cfg["target_type"] == "full" or cfg["target_type"] == "zonal":
    target_transform = sklearn.pipeline.Pipeline(
        [
            ("scale", preprocessing.SigmaScaling(sigma=sigmas_joined)),
            ("pca", PCATargetTransform(n_components=n_components, whiten=True)),
        ]
    )
elif cfg["target_type"] == "scalar":
    target_transform = sklearn.pipeline.Pipeline(
        [
            ("scale", preprocessing.SigmaScaling(sigma=sigmas_joined)),
        ]
    )

#############################
# Fit surrogate with ROM + PCE
#############################
n_cores_half = 1
pce_grid = [
    {
        "order": list(range(1, 12)),
        "mindex_type": ["total_order", "hyperbolic"],
        "fit_type": ["linear", "ElasticNetCV"],
        "fit_params": [
            {
                "alphas": np.logspace(-8, 4, 20),
                "max_iter": 100000,
                "tol": 2.5e-2,
                "n_jobs": n_cores_half,
            }
        ],
    },
    {
        "order": list(range(1, 12)),
        "mindex_type": ["total_order", "hyperbolic"],
        "fit_type": ["LassoCV"],
        "fit_params": [
            {
                "alphas": np.logspace(-8, 4, 20),
                "max_iter": 500000,
                "tol": 2.5e-2,
                "n_jobs": n_cores_half,
            }
        ],
    },
]

if cfg["surrogate"]["fit"]:

    # fit surrogate model
    surrogate = tesuract.MRegressionWrapperCV(
        n_jobs=-1,
        regressor=["pce"],
        reg_params=[pce_grid],
        target_transform=target_transform,
        target_transform_params={},
        verbose=0,
    )
    start = time.time()
    print("\nFitting surrogate...")
    fit_start = time.time()
    surrogate.fit(X_s, Y_joined)
    print("Total fitting time is {0:.3f} seconds".format(time.time() - fit_start))

    def score_func(y, y_pred):
        return np.linalg.norm(y - y_pred) / np.linalg.norm(y)

    # change auto target transform to manual for cloning
    if n_components == "auto":
        n_pc = len(surrogate.best_estimators_)
        if "pca" in target_transform.named_steps.keys():
            target_transform.set_params(pca__n_components=n_pc)
    # Clone and compute the cv score of full model
    print("\nComputing CV score and final fit...")

    cv_score, surrogate = surrogate.compute_cv_score(
        X=X_s, y=Y_joined, scoring="r2", target_transform=target_transform
    )

    # Fit final, simplified model
    surrogate.fit(X_s, Y_joined)
    # save model
    joblib.dump(surrogate, full_save_path)
    print("Saving model to {0}...\n".format(full_save_path))

else:
    print("Loading model from {0}...".format(full_save_path))
    surrogate = joblib.load(full_save_path)
    cv_score = ""
    if surrogate_params["compute_cv_score"]:
        print("\nComputing CV score...")
        cv_score, _ = surrogate.compute_cv_score(X=X_s, y=Y_joined, scoring="r2")

# printing model summary...
table_model = PrettyTable()
table_model.title = "ROM-based ML Model Summary"
table_model.field_names = [
    "Input size",
    "# Components",
    "Output size",
    "CV score",
]
table_model.add_row(
    [
        f"({feature_dim},)",
        f"({n_components},)",
        f"({Y_joined.shape[1]},)",
        cv_score,
    ]
)
table_model.float_format = ".4"
print(table_model)

#############################
# Generating starting point for log likelihood sigmas
#############################


def compute_prior_sigmas(Y, Y_obs, S, W=None, return_raw=False):
    """Compute per grid point sigmas from training data

    Parameters
    ----------
    Y : _type_
        _description_
    Y_obs : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    avg_sigmas = np.zeros(nfields)
    raw_sigmas = []
    for i in range(nfields):
        # W = normalized_area_weights.values.flatten()
        var_per_grid_point = (Y[i] - Y_obs[i]) ** 2 / S[i] ** 2
        prior_sigmas_all = np.sqrt(var_per_grid_point.mean(axis=0))
        raw_sigmas.append(prior_sigmas_all)
        if W is None:
            # compute uniform mean (each grid point equal weights)
            avg_sigmas[i] = np.mean(prior_sigmas_all)
        else:
            # compute weighted mean
            avg_sigmas[i] = np.sum(W * prior_sigmas_all)
    if return_raw:
        return raw_sigmas
    return avg_sigmas


sigma_train = compute_prior_sigmas(Y, Y_obs, S, W=W)
logsigma_train = np.log(sigma_train)
raw_sigmas = compute_prior_sigmas(Y, Y_obs, S, W=W, return_raw=True)
raw_logsigma_train = [np.log(r) for r in raw_sigmas]

#############################
# custom weights for MLE calculation
#############################
custom_weights = opt_params["options"]["weights"]
if custom_weights is not None:
    # only affects the MLE estimate (not MAP or MLE_msigma)
    assert (
        len(custom_weights) == nfields
    ), "weights must be the same length as the number of fields"
    weight_factor = 10.0  # to accentuate the difference in weights (mostly for the Bayesian MAP and MCMC)
    xi_temp = weight_factor * 1.0 * np.array(custom_weights)
    xi_temp[xi_temp < 1e-8] = 1e-8  # avoid divide by zero errors
    sigma_train = 1.0 / np.sqrt(xi_temp)
    logsigma_train = np.log(sigma_train)

#############################
# Define log likelihood, prior and posterior
#############################
def logLike(x, logsigma, Y_obs=Y_obs, return_mse_only=False):
    Y_pred = surrogate.predict(x)
    Y_pred_unjoined = [yi[0] for yi in joinT.inverse_transform(Y_pred)]
    nsamples = Y_pred_unjoined[0].shape[0]

    sigma_sq = np.exp(logsigma) ** 2
    xi = 1.0 / sigma_sq

    # Compute sum of square errors of the MVN
    weighted_mses = np.zeros(nfields)
    for i, Y_i in enumerate(Y_pred_unjoined):
        squared_error = (Y_i - Y_obs[i]) ** 2 / S[i] ** 2
        weighted_sse = np.sum(squared_error * W)
        weighted_mses[i] = weighted_sse

    logLs = np.zeros(nfields)
    MSEs_only = np.zeros(nfields)
    for i, wsse in enumerate(weighted_mses):
        logLs[i] = (
            -0.5 * xi[i] * nsamples * wsse
            + 0.5 * nsamples * np.log(xi[i])
            # - 0.5 * nsamples * np.log(2 * np.pi)  # constant
        )
        MSEs_only[i] = -0.5 * xi[i] * wsse
    if return_mse_only:
        return MSEs_only.sum()
    return logLs.sum()


def logLike_msigma(
    x,
    raw_logsigmas,
    Y_obs=Y_obs,
    return_mse_only=False,
):
    Y_pred = surrogate.predict(x)
    Y_pred_unjoined = [yi[0] for yi in joinT.inverse_transform(Y_pred)]
    nsamples = Y_pred_unjoined[0].shape[0]

    sigma_sqs = [np.exp(r) ** 2 for r in raw_logsigmas]
    xi = [1.0 / s for s in sigma_sqs]

    # Compute sum of square errors of the MVN
    weighted_mses = np.zeros(nfields)
    for i, Y_i in enumerate(Y_pred_unjoined):
        squared_error = (Y_i - Y_obs[i]) ** 2 / S[i] ** 2
        weighted_sse = np.sum(xi[i] * squared_error * W)
        weighted_mses[i] = weighted_sse

    logLs = np.zeros(nfields)
    MSEs_only = np.zeros(nfields)
    for i, wsse in enumerate(weighted_mses):
        logLs[i] = (
            -0.5 * nsamples * wsse
            + 0.5 * np.log(xi[i]).sum()
            # - 0.5 * nsamples * np.log(2 * np.pi)  # constant
        )
        MSEs_only[i] = -0.5 * wsse
    if return_mse_only:
        return MSEs_only.sum()
    return logLs.sum()


def logPrior(logsigma, alpha_prior=3, beta_prior=0.5):
    sigma_sq = np.exp(logsigma) ** 2
    logps = (-alpha_prior - 1) * np.log(sigma_sq) - beta_prior / sigma_sq
    return np.sum(logps)


def logprob(x, logsigma):
    if (x < -1.0).any() or (x > 1.0).any():  # any is faster than min/max
        return -1e12
    return logLike(x, logsigma, Y_obs) + logPrior(logsigma)


###################
# Define objective functions for MLE and MAP estimates
###################
def neg_logLike(x, logsigma=logsigma_train, params={}):
    if (x < -1.0).any() or (x > 1.0).any():  # any is faster than min/max
        return 1e12
    return -1 * logLike(x, logsigma, **params)


def neg_logLike_msigma(x, raw_logsigma=raw_logsigma_train, params={}):
    if (x < -1.0).any() or (x > 1.0).any():  # any is faster than min/max
        return 1e12
    return -1 * logLike_msigma(x, raw_logsigma, **params)


def neg_logprob(xin):
    x, logsigma = xin[:feature_dim], xin[feature_dim : feature_dim + nfields]
    return -1 * logprob(x, logsigma)


###################
# MAP optimization
###################
if opt_params["MAP"] == True:
    print("\nOptimizing...")
    opt_type = "MAP"
    rn = np.random.RandomState(777)

    def parallel_optimization(i):
        xstart = 2 * rn.rand(feature_dim) - 1
        logsigma_start = np.log(0.5 * rn.rand(nfields))
        xinstart = np.append(xstart, logsigma_start)
        logprob_bounds = [(-1.0, 1.0) for i in range(feature_dim)] + [
            (-10, 5)
        ] * nfields
        res = minimize(
            neg_logprob,
            xinstart,
            method="L-BFGS-B",
            jac=None,
            bounds=logprob_bounds,
            options={"ftol": 1e-16, "maxiter": 70000, "disp": False},
        )
        return res.fun, res.x, res.nfev, res.nit

    R = int(mp.cpu_count())
    results = joblib.Parallel(n_jobs=R)(
        joblib.delayed(parallel_optimization)(i) for i in range(R)
    )
    fevals = [soln[0] for soln in results]
    xinopts = [soln[1] for soln in results]
    optarg = np.argmin(fevals)
    xopt_s = xinopts[optarg][:feature_dim]
    sigma_opt = np.exp(xinopts[optarg][feature_dim : feature_dim + nfields])
    xopt_ = feature_transform.inverse_transform(xopt_s)[0]

    np.save(xinopts, os.path.join(savedir, f"opt_{filename}_{opt_type}.npy"))

    postprocessing.print_opt_results(
        xopt_,
        x_labels,
        opt_type,
        R,
        feature_dim,
        solver="L-BFGS-B",
        filename=os.path.join(savedir, f"opt_{filename}_{opt_type}.txt"),
    )

    mse_func_map = partial(
        neg_logLike, logsigma=np.exp(sigma_opt), params={"return_mse_only": True}
    )
    if opt_params["options"]["plot_diagnostics"]:
        postprocessing.plot_results(
            mse_func_map,
            X_s,
            xopt_s,
            x_s_default,
            opt_type,
            filename=os.path.join(savedir, f"plot_{filename}_{opt_type}.png"),
        )

###################
# MLE optimization
###################
mse_func = partial(
    neg_logLike, logsigma=logsigma_train, params={"return_mse_only": True}
)

if opt_params["MLE"] == True:
    opt_type = "MLE"
    print("\nOptimizing...")
    rn = np.random.RandomState(777)

    def parallel_optimization(i):
        xstart = 2 * rn.rand(feature_dim) - 1
        res = minimize(
            neg_logLike,
            xstart,
            method="L-BFGS-B",
            jac=None,
            bounds=[(-1.0, 1.0) for i in range(feature_dim)],
            options={"ftol": 1e-16, "maxiter": 70000, "disp": False},
        )
        return res.fun, res.x, res.nfev, res.nit

    R = int(mp.cpu_count())
    # print(f"Running {R} LBFGS solves...")
    results = joblib.Parallel(n_jobs=R)(
        joblib.delayed(parallel_optimization)(i) for i in range(R)
    )
    fevals = [soln[0] for soln in results]
    xopts = [soln[1] for soln in results]
    optarg = np.argmin(fevals)
    xopt_s = xopts[optarg]
    xopt_ = feature_transform.inverse_transform(xopt_s)[0]

    postprocessing.print_opt_results(
        xopt_,
        x_labels,
        opt_type,
        R,
        feature_dim,
        solver="L-BFGS-B",
        filename=os.path.join(savedir, f"opt_{filename}_{opt_type}.txt"),
    )

    if opt_params["options"]["plot_diagnostics"]:
        postprocessing.plot_results(
            mse_func,
            X_s,
            xopt_s,
            x_s_default,
            opt_type,
            filename=os.path.join(savedir, f"plot_{filename}_{opt_type}.png"),
        )

###################
# MLE spatial var optimization
###################
mse_func2 = partial(
    neg_logLike_msigma,
    raw_logsigma=raw_logsigma_train,
    params={"return_mse_only": True},
)

if opt_params["MLE_msigma"] is True:
    opt_type = "MLE w/ mult. sigmas"
    print("\nOptimizing...")
    rn = np.random.RandomState(777)

    def parallel_optimization(i):
        xstart = 2 * rn.rand(feature_dim) - 1
        res = minimize(
            neg_logLike_msigma,
            xstart,
            method="L-BFGS-B",
            jac=None,
            bounds=[(-1.0, 1.0) for i in range(feature_dim)],
            options={"ftol": 1e-16, "maxiter": 70000, "disp": False},
        )
        return res.fun, res.x, res.nfev, res.nit

    R = int(mp.cpu_count())
    results = joblib.Parallel(n_jobs=R)(
        joblib.delayed(parallel_optimization)(i) for i in range(R)
    )
    fevals = [soln[0] for soln in results]
    xopts = [soln[1] for soln in results]
    optarg = np.argmin(fevals)
    xopt_s = xopts[optarg]
    xopt_ = feature_transform.inverse_transform(xopt_s)[0]

    postprocessing.print_opt_results(
        xopt_,
        x_labels,
        opt_type,
        R,
        feature_dim,
        solver="L-BFGS-B",
        filename=os.path.join(savedir, f"opt_{filename}_MLE_msigma.txt"),
    )

    if opt_params["options"]["plot_diagnostics"]:
        postprocessing.plot_results(
            mse_func2,
            X_s,
            xopt_s,
            x_s_default,
            opt_type,
            filename=os.path.join(savedir, f"plot_{filename}_MLE_msigma.png"),
        )

###################
# validating the optimal estimates
###################
opt_type = cfg["validation"]["optimization"]
if opt_type is not None:
    # load true e3sm control/ default run on xopt_s
    e3sm_default = xr.open_dataset(
        os.path.join(datadir, f"control_{season}_lat_lon.nc")
    )
    # load the optimal solution
    xopt_e3sm_file = os.path.join(
        datadir,
        f"xopt_{field_str_list}_{target_type}_{opt_type}_{season}_lat_lon.nc",
    )
    e3sm_xopt = xr.open_dataset(xopt_e3sm_file)

    # Extract fields and rescale the PRECT
    e3sm_xopt_fields = {}
    for f in fields:
        ytemp = e3sm_xopt[f][0]  # [0] to remove time dimension
        if f == "PRECT":
            ytemp *= 3600 * 24 * 1000  # change to mm/day from m/sec
        e3sm_xopt_fields[f] = ytemp

    # Extract default fields
    e3sm_default_fields = {}
    for f in fields:
        ytemp2 = e3sm_default[f][0]  # [0] to remove time dimension
        if f == "PRECT":
            ytemp2 *= 3600 * 24 * 1000  # change to mm/day from m/sec
        e3sm_default_fields[f] = ytemp2

    # now compare to Y_obs_raw
    default_errors = {}
    xopt_errors = {}
    for i, f in enumerate(fields):
        z = mse_transform.fit_transform(
            (Y_obs_raw[i] - Y_obs_raw[i].mean()) ** 2
        ).values
        default_errors[f] = np.sqrt(
            mse_transform.fit_transform(
                (e3sm_default_fields[f] - Y_obs_raw[i]) ** 2
            ).values
            / z
        )
        xopt_errors[f] = np.sqrt(
            mse_transform.fit_transform(
                (e3sm_xopt_fields[f] - Y_obs_raw[i]) ** 2
            ).values
            / z
        )

    print("Default errors:\n", default_errors)
    print("Opt soln errors:\n", xopt_errors)

    # plot paired bar chart
    labels = fields
    default_means = [np.around(default_errors[f], 3) for f in fields]
    at_means = [np.around(xopt_errors[f], 3) for f in fields]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, default_means, width, label="default")
    rects2 = ax.bar(x + width / 2, at_means, width, label="autotuned")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("normalized RMSE")
    ax.set_title(f"{season}, {target_type} field, {opt_type} optimization")
    ax.set_xticks(x, labels)
    ax.set_ylim([0, 1])
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fancybox=True)

    ax.bar_label(rects1, padding=2)
    ax.bar_label(rects2, padding=2)

    plot_savename = (
        f"validation_plot_{field_str_list}_{target_type}_{opt_type}_{season}.png"
    )
    fig.savefig(os.path.join(savedir, plot_savename))
