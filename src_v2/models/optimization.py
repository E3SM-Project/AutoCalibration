import numpy as np
import multiprocessing as mp
import xarray as xr
import os, time, sys
from scipy.optimize import minimize
from functools import partial
import copy
#############################
# Generating starting point for log likelihood sigmas
#############################
def compute_prior_sigmas(Y, Y_obs, S, nlat_lon_fields, nlat_plev_fields, W=None, W_plev=None, return_raw=False):
    """Compute per grid point sigmas from training data

    Parameters
    ----------
    Y : list
        fields from the PPE
    Y_obs : list
        observational fields
    S : standard deviation

    Returns
    -------
    _type_
        _description_
    """
    avg_sigmas = np.zeros(len(Y))
    raw_sigmas = []
    for i in range(len(Y)):
        var_per_grid_point = (Y[i] - Y_obs[i]) ** 2 / S[i] ** 2
        prior_sigmas_all = np.sqrt(var_per_grid_point.mean(axis=0))
        raw_sigmas.append(prior_sigmas_all)
        if (W is None and i < nlat_lon_fields) or (W_plev is None and i >= nlat_lon_fields):
            avg_sigmas[i] = np.mean(prior_sigmas_all)
        elif i >= nlat_lon_fields + nlat_plev_fields:
            avg_sigmas[i] = prior_sigmas_all
        elif i >= nlat_lon_fields: # compute uniform mean (each grid point equal weights)
            avg_sigmas[i] = np.sum(W_plev[i-nlat_lon_fields] * prior_sigmas_all)
        else:
            # compute weighted mean
            avg_sigmas[i] = np.sum(W * prior_sigmas_all)
    if return_raw:
        return raw_sigmas
    return avg_sigmas

def process_custom_weights(opt_params, season, nlat_lon_fields, nlat_plev_fields, nglobal_fields, lat_lon_fields, lat_plev_fields, global_fields, sigma_train, logsigma_train):
    if opt_params["options"]["use_weights"] == 'season_field' and season == 'ALL':
        custom_weights = np.zeros(nlat_lon_fields + nlat_plev_fields + nglobal_fields)
        season_weights = opt_params["options"]["season_weights"]
        variable_weights = opt_params["options"]["variable_weights"]
        assert (
            len(season_weights) == 4
        ), "season weights must have length 4"
        assert (
            len(variable_weights) == (len(lat_lon_fields) + len(lat_plev_fields) + len(global_fields))
        ), "variable weights must have length the number of output variables"

        season_weights_dict = dict(zip(['DJF', 'MAM', 'JJA', 'SON'], season_weights))
        variable_weights_dict = dict(zip(lat_lon_fields + lat_plev_fields + global_fields, variable_weights))

        fields = list(np.tile(lat_lon_fields, 4)) + list(np.tile(lat_plev_fields, 4)) + global_fields
        seasons = list(np.repeat(['DJF', 'MAM', 'JJA', 'SON'], nlat_lon_fields/4)) + list(np.repeat(['DJF', 'MAM', 'JJA', 'SON'], nlat_plev_fields/4)) + list(np.repeat('NA',nglobal_fields))
        for i in range(len(custom_weights)):
            if i < nlat_lon_fields + nlat_plev_fields:
                custom_weights[i] = season_weights_dict[seasons[i]]  *  variable_weights_dict[fields[i]]
            else:
                custom_weights[i] = variable_weights_dict[fields[i]]
    elif opt_params["options"]["use_weights"] == 'manual':
        custom_weights = opt_params["options"]["weights"]
        # only affects the MLE estimate (not MAP or MLE_msigma)
        assert (
            len(custom_weights) == nfields
        ), "weights must be the same length as the number of fields"
    else:
        return sigma_train, logsigma_train
    weight_factor = 10.0  # to accentuate the difference in weights (mostly for the Bayesian MAP and MCMC)
    xi_temp = weight_factor * 1.0 * np.array(custom_weights)
    xi_temp[xi_temp < 1e-8] = 1e-12  # avoid divide by zero errors
    sigma_train = 1.0 / np.sqrt(xi_temp)
    logsigma_train = np.log(sigma_train)
    return sigma_train, logsigma_train


# Define functions
def logLike(x, logsigma, params, return_mse_only=False):

    Y_pred = params['surrogate'].predict(x) # time-wise, the largest step in evaluating each step of the optimization, line 337 of pca_multitarget_regression.py
    Y_pred_unjoined = [yi[0] for yi in params['joinT'].inverse_transform(Y_pred)]
    nsamples = Y_pred_unjoined[0].shape[0]

    sigma_sq = np.exp(logsigma) ** 2
    xi = 1.0 / sigma_sq
    # Compute sum of square errors of the MVN
    weighted_mses = np.zeros(params['ntot_fields'])
    for i, Y_i in enumerate(Y_pred_unjoined):
        squared_error = (Y_i - params['Y_obs'][i]) ** 2 / params['S'][i] ** 2
        if i < params['nll_fields']:
            weighted_sse = np.sum(squared_error * params['W'])
        elif i < params['nll_fields'] + params['nlp_fields']:
            weighted_sse = np.sum(squared_error * params['W_plev'][i  - params['nll_fields']])
        else:
            weighted_sse = squared_error
        weighted_mses[i] = weighted_sse

    logLs = np.zeros(params['ntot_fields'])
    MSEs_only = np.zeros(params['ntot_fields'])
    for i, wsse in enumerate(weighted_mses):
        if i < params['nll_fields'] + params['nlp_fields']:
            nsamples = Y_pred_unjoined[i].shape[0]
        else:
            nsamples = Y_pred_unjoined[0].shape[0]
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
    params,
    return_mse_only=False,
):
    Y_pred = params['surrogate'].predict(x)
    Y_pred_unjoined = [yi[0] for yi in params['joinT'].inverse_transform(Y_pred)]

    sigma_sqs = [np.exp(r) ** 2 for r in raw_logsigmas]
    xi = [1.0 / s for s in sigma_sqs]

    # Compute sum of square errors of the MVN
    weighted_mses = np.zeros(params['ntot_fields'])
    for i, Y_i in enumerate(Y_pred_unjoined):
        squared_error = (Y_i - params['Y_obs'][i]) ** 2 / params['S'][i] ** 2
        if i < params['nll_fields']:
            weighted_sse = np.sum(squared_error * params['W'])
        elif i < params['nll_fields'] + params['nlp_fields']:
            weighted_sse = np.sum(squared_error * params['W_plev'][i  - params['nll_fields']])
        else:
            weighted_sse = squared_error
        weighted_mses[i] = weighted_sse

    logLs = np.zeros(params['ntot_fields'])
    MSEs_only = np.zeros(params['ntot_fields'])
    for i, wsse in enumerate(weighted_mses):
        if i < params['nll_fields'] + params['nlp_fields']:
            nsamples = Y_pred_unjoined[i].shape[0]
        else:
            nsamples = Y_pred_unjoined[0].shape[0]
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


def logprob(x, logsigma, params={}):
    if (x < -1.0).any() or (x > 1.0).any():  # any is faster than min/max
        return -1e12
    return logLike(x, logsigma, params) + logPrior(logsigma)

###################
# Define objective functions for MLE and MAP estimates
###################
def neg_logLike(x, logsigma, params={}):
    if (x < -1.0).any() or (x > 1.0).any():  # any is faster than min/max
        return 1e12
    return -1 * logLike(x, logsigma, params)


def neg_logLike_msigma(x, raw_logsigma, params={}):
    if (x < -1.0).any() or (x > 1.0).any():  # any is faster than min/max
        return 1e12
    return -1 * logLike_msigma(x, raw_logsigma, params)

def neg_logprob(xin, params):
    x, logsigma = xin[:params['feature_dim']], xin[params['feature_dim'] : params['feature_dim'] + params['ntot_fields']]
    return -1 * logprob(x, logsigma, params)

####################
# carrying out the optimization
####################
def parallel_optimization_MAP(fun_input):
    i, xstarts, logsigma_starts, params = fun_input
    xstart = xstarts[i]
    logsigma_start = logsigma_starts[i]
    xinstart = np.append(xstart, logsigma_start)
    logprob_bounds = [(-1.0, 1.0) for i in range(params['feature_dim'])] + [
        (-10, 5)
    ] * params['ntot_fields']
    res = minimize(
        neg_logprob,
        xinstart,
        args=(params),
        method="L-BFGS-B",
        jac=None,
        bounds=logprob_bounds,
        options={"ftol": 1e-16, "maxiter": 70000, "disp": False},
    )
    return res.fun, res.x, res.nfev, res.nit


def optimize_MAP(params, opt_params):
    print("\nOptimizing...")
    opt_type = opt_params["method"]
    rn = np.random.RandomState(777)

    R = int(mp.cpu_count())
    xstarts = 2 * rn.rand(R, params['feature_dim']) - 1
    logsigma_starts = np.log(0.5 * rn.rand(R, params['ntot_fields']))
    
    with mp.get_context("fork").Pool() as pool:
        results = pool.map(parallel_optimization_MAP, [[i, xstarts, logsigma_starts, params] for i in range(R)])

    fevals = [soln[0] for soln in results]
    xopts = [soln[1] for soln in results]
    optarg = np.argmin(fevals)
    xopt_s = xopts[optarg][:params['feature_dim']]
    sigma_opt = np.exp(xopts[optarg][params['feature_dim'] : params['feature_dim'] + params['ntot_fields']])
    xopt_ = params['feature_transform'].inverse_transform(xopt_s)[0]
    
    params_mse = copy.deepcopy(params)
    params_mse['return_mse_only'] = True
    mse_func = partial(
        neg_logLike, logsigma=np.exp(sigma_opt), params = params_mse)
    return opt_type, xopt_, xopt_s, sigma_opt, mse_func


def parallel_optimization_MLE(fun_input):
    i, xstarts, params = fun_input
    xstart = xstarts[i]
    res = minimize(
        neg_logLike,
        xstart,
        args=(params['logsigma_train'], params),
        method="L-BFGS-B",
        jac=None,
        bounds=[(-1.0, 1.0) for i in range(params['feature_dim'])],
        options={"ftol": 1e-14, "maxiter": 70000, "disp": False},
    )
    return res.fun, res.x, res.nfev, res.nit

def optimize_MLE(params, opt_params):
    print("\nOptimizing...")
    opt_type = opt_params["method"]
    rn = np.random.RandomState(777)

    R = int(mp.cpu_count())
    xstarts = 2 * rn.rand(R, params['feature_dim']) - 1
    with mp.get_context("fork").Pool() as pool:    
        results = pool.map(parallel_optimization_MLE, [[i, xstarts, params] for i in range(R)])

    fevals = [soln[0] for soln in results]
    xopts = [soln[1] for soln in results]
    optarg = np.argmin(fevals)
    xopt_s = xopts[optarg]
    sigma_opt = np.exp(xopts[optarg][params['feature_dim'] : params['feature_dim'] + params['ntot_fields']])
    xopt_ = params['feature_transform'].inverse_transform(xopt_s)[0]

    params_mse = copy.deepcopy(params)
    params_mse['return_mse_only'] = True
    mse_func = partial(
        neg_logLike, logsigma=np.log(sigma_opt), params=params_mse
    )
    return opt_type, xopt_, xopt_s, sigma_opt, mse_func

def parallel_optimization_MLE_msigma(fun_input):
    i, xstarts, params = fun_input
    xstart = xstarts[i]
    res = minimize(
        neg_logLike_msigma,
        xstart,
        args = (params['raw_logsigma_train'], params),
        method="L-BFGS-B",
        jac=None,
        bounds=[(-1.0, 1.0) for i in range(params['feature_dim'])],
        options={"ftol": 1e-16, "maxiter": 70000, "disp": False},
    )
    return res.fun, res.x, res.nfev, res.nit


def optimize_MLE_msigma(params, opt_params):
    opt_type = opt_params["method"]
    print("\nOptimizing...")
    rn = np.random.RandomState(777)

    R = int(mp.cpu_count())
    xstarts = 2 * rn.rand(R, params['feature_dim']) - 1

    with mp.get_context("fork").Pool() as pool:
        results = pool.map(parallel_optimization_MLE_msigma, [[i, xstarts, params] for i in range(R)])

    fevals = [soln[0] for soln in results]
    xopts = [soln[1] for soln in results]
    optarg = np.argmin(fevals)
    xopt_s = xopts[optarg]
    xopt_ = params['feature_transform'].inverse_transform(xopt_s)[0]

    params_mse = copy.deepcopy(params)
    params_mse['return_mse_only'] = True
    mse_func = partial(
        neg_logLike_msigma,
        raw_logsigma=params_mse['raw_logsigma_train'],
        params={"return_mse_only": True},
    )
    raw_sigma_train = [np.exp(Yi) for Yi in params['raw_logsigma_train']]
    return opt_type, xopt_, xopt_s, raw_sigma_train, mse_func


