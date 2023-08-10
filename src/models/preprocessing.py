import numpy as np
import tesuract
import sklearn
import xarray as xr
import os
import clif.preprocessing as cpp
import multiprocessing as mp

def load_lat_lon_data(season, nlat_lon_fields, lat_lon_fields, nyear, resolution, target_source, datadir):
    """Load fields defined by lon/lat

    Parameters
    ----------
    season : str
        either ANN, DJF, MAM, JJA, SON, or ALL
    nlat_lon_fields : int
        number of output lat/lon variables
    lat_lon_fields: list
        list of strings of lenth nlat_lon_fields specifying which fields to use
    nyear: int
        number of years in simulation run; either 5 or 10 currently
    resolution: str
        whether to use "24x48" or "180x360" grid in lat/lon
    target_source: str
        either "obs" or "ctrl"; default "obs"; we want to match to observational fields
    datadir: str
        where to find the data; have used "../../data"

    Returns
    -------
    Y_raw : list 
        list of length nlat_lon_fields, each with first dimension governing 250 simulation runs, xarray time-averaged-simulation results
    Y_obs_raw : list
        list of length nlat_lon_fields, with one fewer dimension compared to Y_raw, that are xarray time-averaged observations (or control)
    nlat_lon_fields : int
        number of lat/lon fields, which has been multiplied by 4 compared to its input value if season == 'ALL'
    normalized_area_weights : xarray.DataArray
        xarray object with dimensions "resolution" that describe area taken up by each grid point, with normalized_area_weights.sum() = 1
    area_weights : xarray.DataArray
        xarray object with dimensions "resolution" that describe area taken up by each grid point, with area_weights.sum() != 1
    """

    if season == 'ALL':# if all seasons chosen, we take one field for each season, increasing number of fields x 4
        Y_raw = []
        Y_obs_raw = []
        nlat_lon_fields = nlat_lon_fields * 4
        for season_i in ['DJF', 'MAM', 'JJA', 'SON']:
            dataset = xr.open_dataset(os.path.join(datadir, f"lat_lon_{nyear}yr_{resolution}_{season_i}.nc"))
            if target_source == 'obs':
                dataset_ref = xr.open_dataset(os.path.join(datadir, f"lat_lon_{resolution}_{season_i}_obs.nc"))
            elif target_source == 'ctrl':
                dataset_ref = xr.open_dataset(os.path.join(datadir, f"lat_lon_{nyear}yr_{resolution}_{season_i}_ctrl.nc"))
            for f in lat_lon_fields:
                Y_raw.append(dataset[f])
                Y_obs_raw.append(dataset_ref[f])
    else: #otherwise, just need one season
        dataset = xr.open_dataset( os.path.join(datadir, f"lat_lon_{nyear}yr_{resolution}_{season}.nc"))
        if target_source == 'obs':
            dataset_ref = xr.open_dataset(os.path.join(datadir, f"lat_lon_{resolution}_{season}_obs.nc"))
        elif target_source == 'ctrl':
            dataset_ref = xr.open_dataset(os.path.join(datadir, f"lat_lon_{nyear}yr_{resolution}_{season}_ctrl.nc"))
        Y_raw = [dataset[f] for f in lat_lon_fields]  # simulation data
        Y_obs_raw = [dataset_ref[f] for f in lat_lon_fields]  # obs data
    area_weights = dataset.area[0]  # lat lon area weights
    normalized_area_weights = area_weights / area_weights.sum()
    return Y_raw, Y_obs_raw, nlat_lon_fields, normalized_area_weights, area_weights

def load_lat_plev_data(season, nlat_plev_fields, lat_plev_fields, nyear, resolution, target_source, datadir):
    """Load fields defined by lat/plev

    Parameters
    ----------
    season : str
        either ANN, DJF, MAM, JJA, SON, or ALL
    nlat_plev_fields : int
        number of output lat/plev variables
    lat_plev_fields: list
        list of strings of lenth nlat_plev_fields specifying which fields to use
    nyear: int
        number of years in simulation run; either 5 or 10 currently
    resolution: str
        whether to use "24x37" or "180x37" grid in lat/plev
    target_source: str
        either "obs" or "ctrl"; default "obs"; we want to match to observational fields
    datadir: str
        where to find the data; have used "../../data"

    Returns
    -------
    Y_raw_plev : list 
        list of length nlat_plev_fields, each with first dimension governing 250 simulation runs, xarray time-averaged-simulation results
    Y_obs_raw_plev : list
        list of length nlat_plev_fields, with one fewer dimension compared to Y_raw, that are xarray time-averaged observations (or control)
    nlat_plev_fields : int
        number of lat/plev fields, which has been multiplied by 4 compared to its input value if season == 'ALL'
    normalized_area_weights_plev : xarray.DataArray
        xarray object with dimension latitude that describe areas taken up by each latitude grid point, with normalized_area_weights_plev.sum() = 1
    area_weights_plev : xarray.DataArray
        xarray object with dimensions latitude that describe area taken up by each latitude grid point, with area_weights_plev.sum() != 1
    """

    if season == 'ALL':
        Y_raw_plev = []
        Y_obs_raw_plev = []
        plev_mask = []
        nlat_plev_fields = nlat_plev_fields * 4
        for season_i in ['DJF', 'MAM', 'JJA', 'SON']:
            dataset_plev = xr.open_dataset(os.path.join(datadir, f"lat_plev_{nyear}yr_{resolution}_{season_i}.nc"))
            if target_source == 'obs':
                dataset_plev_ref = xr.open_dataset(os.path.join(datadir, f"lat_plev_{resolution}_{season_i}_obs.nc"))
            elif target_source == 'ctrl':
                dataset_plev_ref = xr.open_dataset(os.path.join(datadir, f"lat_plev_{nyear}yr_{resolution}_{season_i}_ctrl.nc"))
            for f in lat_plev_fields:
                Y_raw_plev.append(dataset_plev[f])
                Y_obs_raw_plev.append(dataset_plev_ref[f])
                plev_mask.append(np.isnan(dataset_plev[f]).sum(axis = 0) == 0)
    else:
        dataset_plev = xr.open_dataset( os.path.join(datadir, f"lat_plev_{nyear}yr_{resolution}_{season}.nc"))
        if target_source == 'obs':
            dataset_plev_ref = xr.open_dataset(os.path.join(datadir, f"lat_plev_{resolution}_{season}_obs.nc"))
        elif target_source == 'ctrl':
            dataset_plev_ref = xr.open_dataset(os.path.join(datadir, f"lat_plev_{nyear}yr_{resolution}_{season}_ctrl.nc"))
        Y_raw_plev = [dataset_plev[f] for f in lat_plev_fields]  # simulation data
        Y_obs_raw_plev = [dataset_plev_ref[f] for f in lat_plev_fields]  # obs data
        plev_mask = [np.isnan(dataset_plev[f]).sum(axis=0) == 0 for f in lat_plev_fields] #mask NA values (antarctica, primarily)
    area_weights_plev = dataset_plev.area[0]
    normalized_area_weights_plev = area_weights_plev/area_weights_plev.sum()
    return Y_raw_plev, Y_obs_raw_plev, nlat_plev_fields, plev_mask, normalized_area_weights_plev, area_weights_plev

def load_global_data(global_fields, RESTOM_target, nyear, datadir):
    """Load global fields, just RESTOM for now

    Parameters
    ----------
    global_fields: list
        [] or ["RESTOM"] specifying if RESTOM is used as a target or not
    RESTOM_target: float
        a number to use as the "observational data" for RESTOM
    nyear: int
        number of years in simulation run; either 5 or 10 currently
    datadir: str
        where to find the data; have used "../../data"

    Returns
    -------
    Y_raw_global : list 
        list of length nglobal_fields, each with first dimension governing 250 simulation runs, xarray time-averaged-simulation results
    Y_obs_raw_global : list
        list of length nglobal_fields, with one fewer dimension compared to Y_raw, that are xarray time-averaged observations (or control)
    """

    if len(global_fields) > 0:
        if 'RESTOM' in global_fields:
            dataset = xr.open_dataset(os.path.join(datadir, f"lat_lon_{nyear}yr_180x360_ANN.nc"))
            area_weights = dataset.area[0]  # lat lon area weights
            normalized_area_weights = area_weights / area_weights.sum()
            # Compute restom for each simulation run
            RESTOM = np.multiply(dataset['FSNT'] - dataset['FLNT'], normalized_area_weights).sum(axis = 1).sum(axis = 1)
            Y_raw_global = [RESTOM]
            Y_obs_raw_global = [RESTOM_target]
    else:
        Y_raw_global = []
        Y_obs_raw_global = []
    return Y_raw_global, Y_obs_raw_global

def load_input_parameters(datadir):
    """Load input parameters that were changed in perturbed parameter ensemble, which are saved in most .nc files

    Parameters
    ----------
    datadir: str
        where to find the data; have used "../../data"

    Returns
    -------
    X_xr : xarray.DataArray
        xarray dataset of dimensions n_samples in PPE times n_perturbed_parameters
    X_bounds : numpy.array
        numpy array of dimensions n_perturbed_parameters times 2, describing lower and upper bounds of each parameter
    """
    
    dataset = xr.open_dataset(os.path.join(datadir, f"lat_lon_10yr_24x48_ANN.nc"))
    X_xr = dataset["lhs"]
    X_bounds = dataset["lhs_bnds"].values
    return X_xr, X_bounds




class JoinTransform(sklearn.base.TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        # assert isinstance(Y, list), "Input must be a list"
        self.split_index = np.concatenate([[0], np.cumsum([Xi.shape[1] for Xi in X])])
        return self

    def transform(self, X):
        """Join together the multiple objectives"""
        Yhat = np.hstack(X)
        return Yhat

    def inverse_transform(self, Yhat):
        """Split the multiple objectives"""
        Y_recon = []
        si = self.split_index
        ndim = Yhat.ndim
        if ndim == 1:
            # add a dimension if array is 1d
            Yhat = Yhat[np.newaxis, :]
        for i in range(len(si) - 1):
            Y_recon.append(Yhat[:, si[i] : si[i + 1]])
        if ndim == 1:
            # if 1d, convert back to 1d output
            Y_recon = [Ytemp.flatten() for Ytemp in Y_recon]
        return Y_recon


# create target scaling for multi-objective optimization
class SigmaScaling(sklearn.base.TransformerMixin):
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X / self.sigma

    def inverse_transform(self, Xhat):
        return self.sigma * Xhat

def transform_data(Y_raw, Y_raw_plev, Y_raw_global, Y_obs_raw, Y_obs_raw_plev, Y_obs_raw_global,
         nlat_lon_fields, nlat_plev_fields, global_fields, area_weights, area_weights_plev, normalized_area_weights, normalized_area_weights_plev, cfg):
    """Big messy functional that preps data according to options for the surrogate

    Parameters
    ----------
    global_fields : list
        [] or ["RESTOM"] specifying if RESTOM is used as a target or not
    cfg : dict
        configuration options
    everything_else : 
        results from load_lat_lon_data, load_lat_plev_data, load_global_data
    
    Returns
    -------
    Y, Y_obs, S, W, W_plev, scalar_function, scalar_function_plev, surrogate_scorer, plev_mask_flatten
    Y : list
        length nlat_lon_fields + nlat_plev_fields + nglobal_fields, each entry a np.array with 250 rows and columns the number of gridpoints used for that field
    Y_obs : list
        length nlat_lon_fields + nlat_plev_fields + nglobal_fields, each entry a np.array with entries of the number of gridpoints used for that field
    S : list
        same dimensions as Y_obs, giving standard deviation of each observational field
    W : np.array
        describing area weights for lat/lon fields
    W_plev: list
        describing area weights for each lat/plev field separately
    plev_mask_flatten: list
        describing which entries of each lat/plev fields to use
    others: 
        functions governing fitting process
    """
    
    # if specified, subtract ensemble mean of each field
    Y_ens_mean = [Yi.mean(axis = 0) for Yi in Y_raw]
    Y_ens_mean_plev = [Yi.mean(axis = 0) for Yi in Y_raw_plev]
    Y_ens_mean_global = [Yi.mean() for Yi in Y_raw_global]
    if cfg['subtract_ens_mean']:
        Y_raw = [Y_raw[i] - Y_ens_mean[i] for i in range(len(Y_raw))]
        Y_obs_raw = [Y_obs_raw[i] - Y_ens_mean[i] for i in range(len(Y_obs_raw))]
        Y_raw_plev = [Y_raw_plev[i] - Y_ens_mean_plev[i] for i in range(len(Y_raw_plev))]
        Y_obs_raw_plev = [Y_obs_raw_plev[i] - Y_ens_mean_plev[i] for i in range(len(Y_obs_raw_plev))]
        Y_raw_global = [Y_raw_global[i] - Y_ens_mean_global[i] for i in range(len(Y_raw_global))]
        Y_obs_raw_global = [Y_obs_raw_global[i] - Y_ens_mean_global[i] for i in range(len(Y_obs_raw_global))]

    # define transforms for full field, zonal and scalar outputs
    flatten_transform = cpp.FlattenData(dims=["lat", "lon"])
    flatten_transform_plev = cpp.FlattenData(dims=["lat", "plev"])
    zonal_mean_transform = cpp.MarginalizeOutTransform(dims=["lon"], lat_lon_weights=area_weights)
    area_weights_plev2d = area_weights_plev.expand_dims(dim = {'plev': 37})
    zonal_mean_transform_plev = cpp.MarginalizeOutTransform(dims=["plev"], lat_lon_weights=area_weights_plev2d)
    mse_transform = cpp.MarginalizeOutTransform(dims=["lat", "lon"], lat_lon_weights=area_weights)

    # compute an xarray data array of scalers, descriibing standard deviation of each observational field
    Sigmas = [np.sqrt(Yi.var()) + 0 * Yi for Yi in Y_obs_raw]
    Sigmas_plev = [np.sqrt(Yi.var()) + 0 * Yi for Yi in Y_obs_raw_plev]
    
    if cfg["target_type"] == "full":
        # prep lat/lon and lat/plev for entire fields : mainly vectorizing everything
        Y_obs = [flatten_transform.fit_transform(Yi) for Yi in Y_obs_raw]
        Y = [flatten_transform.fit_transform(Yi) for Yi in Y_raw]
        S = [flatten_transform.fit_transform(s) for s in Sigmas]
        W = normalized_area_weights.values.flatten()
        
        # we want to remove any plev entry that has an nan -- mostly near Antarctica
            # but this may vary for different plev fields, complicating things
        plev_mask = [np.isnan(Yi).sum(axis = 0) < 1 for Yi in Y_raw_plev] # a list describing nan structure for each lat/plev field
        plev_mask_flatten = [flatten_transform_plev.fit_transform(plevi) for plevi in plev_mask]
        Y_obs_plev = [flatten_transform_plev.fit_transform(Y_obs_raw_plev[ival])[plev_mask_flatten[ival]] for ival in range(len(Y_obs_raw_plev))]
        Y_plev = [flatten_transform_plev.fit_transform(Y_raw_plev[ival])[:,plev_mask_flatten[ival]] for ival in range(len(Y_raw_plev))]
        S_plev = [flatten_transform_plev.fit_transform(Sigmas_plev[ival])[plev_mask_flatten[ival]] for ival in range(len(Sigmas_plev))]
        W_plev = [np.reshape(np.tile(normalized_area_weights_plev.values.flatten(), (37,1)),(-1))[plev_mask_flatten[ival]]/37 for ival in range(len(Y_raw_plev))]

    if cfg["target_type"] == "zonal":
        # collapse lat/lon fields along longitude dimension
        Y_obs = [zonal_mean_transform.fit_transform(Yi) for Yi in Y_obs_raw]
        Y = [zonal_mean_transform.fit_transform(Yi) for Yi in Y_raw]
        S = [zonal_mean_transform.fit_transform(s) for s in Sigmas]
        weights = zonal_mean_transform.fit_transform(area_weights).values
        W = weights / weights.sum()

        # collapse lat/plev fields along plev dimension
        Y_obs_plev = [zonal_mean_transform_plev.fit_transform(Y_obs_raw_plev[ival]) for ival in range(len(Y_obs_raw_plev))]
        Y_plev = [zonal_mean_transform_plev.fit_transform(Y_raw_plev[ival]) for ival in range(len(Y_raw_plev))]
        plev_mask = [np.isnan(Y_plev).sum(axis = 0) < 1 for Yi in Y_raw_plev]
        plev_mask_flatten = plev_mask
        S_plev = [zonal_mean_transform_plev.fit_transform(Sigmas_plev[ival]) for ival in range(len(Sigmas_plev))]
        W_plev = [normalized_area_weights_plev.values for ival in range(len(Y_raw_plev))]

    ### Prep for cfg["target_type"] == 'scalar'  or cfg["target_type"] = 'scalarmean'
    # define scalar error
    def scalar_function(Y, Y_obs):
        # Y and Y_obs should be the same shape
        return np.sqrt(mse_transform.fit_transform((Y - Y_obs) ** 2))

    # make plev weights defined on entire field instead of just a vector
    area_weights_plev_all = np.outer(np.ones(37), area_weights_plev.values) # constant plev and varying lat weights for now
    area_weights_plev_xr = xr.DataArray(area_weights_plev_all, dims = ["plev", "lat"])
    mse_transform_plev = cpp.MarginalizeOutTransform(dims=["plev", "lat"], lat_lon_weights=area_weights_plev_xr)
    def scalar_function_plev(Y, Y_obs):
        return np.sqrt(mse_transform_plev.fit_transform((Y - Y_obs) ** 2))

    # load custom scalar function
    # will overwrite scalar_function above
    if cfg["custom_scalar_function"]["file"] is not None:
        print("\nLoading scalar function...")
        exec(open(cfg["custom_scalar_function"]["file"]).read())
        # overwrite function above with custom
        scalar_function = custom_scalar_function

    if cfg["custom_scalar_plev_function"]["file"] is not None:
        print("\nLoading scalar plev function...")
        exec(open(cfg["custom_scalar_plev_function"]["file"]).read())
        # overwrite function above with custom
        scalar_function_plev = custom_scalar_plev_function

    # option for how surrogate function is estimated
    if cfg["custom_surrogate_fit_function"]["file"] is not None:
        print("\nLoading surrogate fitting function...")
        exec(open("custom_surrogate_fit_function.py").read())
        surrogate_scorer = make_scorer(custom_surrogate_fit_function)
    else:
        surrogate_scorer = "neg_root_mean_squared_error"



    if cfg["target_type"] == "scalarmean":

        Ytemp = [
            (flatten_transform.fit_transform(Yi) *W_prelim).sum(axis = 1).expand_dims(dim="new", axis=1)
            for i, Yi in enumerate(Y_raw)
        ]
        Y = [Yi for Yi in Ytemp]  # compute Root mean square error
        Y_obs = [(flatten_transform.fit_transform(Yi) * W_prelim).sum() for Yi in Y_obs_raw]
        S = [s.mean(dim=["lat", "lon"]) ** 1 for s in Sigmas]
        W = np.ones(1)
        
        plev_mask = [np.isnan(Yi).sum(axis = 0) < 1 for Yi in Y_raw_plev]
        plev_mask_flatten = [flatten_transform_plev.fit_transform(plevi) for plevi in plev_mask]
        W_plev = [np.reshape(np.tile(normalized_area_weights_plev.values.flatten(), (37,1)),(-1))[plev_mask_flatten[ival]]/37 for ival in range(len(Y_raw_plev))]
        Ytemp_plev = [
            (flatten_transform_plev.fit_transform(Y_raw_plev[ival])[:,plev_mask_flatten[ival]]* W_plev[ival]).sum(axis = 1).expand_dims(dim="new", axis = 1) for ival in range(len(Y_raw_plev))
        ]
        Y_plev = [Yi for Yi in Ytemp_plev]  # compute Root mean square error
        Y_obs_plev = [(flatten_transform_plev.fit_transform(Y_obs_raw_plev[ival])[plev_mask_flatten[ival]]* W_plev[ival]).sum() for ival in range(len(Y_obs_raw_plev))]
        S_plev = [s.mean(dim=["plev", "lat"]) ** 1 for s in Sigmas_plev]
        W_plev = [np.ones(1)] * nlat_plev_fields
        plev_mask_flatten = np.repeat([True], nlat_plev_fields)

    if cfg["target_type"] == "scalar":
        # Compute squared error at each location, try to match it to 0
        Ytemp = [
            scalar_function(Yi, Y_obs_raw[i]).expand_dims(dim="new", axis=1)
            for i, Yi in enumerate(Y_raw)
        ]
        Y = [Yi for Yi in Ytemp]  # compute Root mean square error
        Y_obs = [xr.DataArray(0)] * (nlat_lon_fields)  # Y_obs = 0
        S = [s.mean(dim=["lat", "lon"]) ** 1 for s in Sigmas]
        W = np.ones(1)
        #S = [s.mean(dim=["lat", "lon"]) ** 1/np.sqrt(s.size) for s in Sigmas]

        Ytemp_plev = [
            scalar_function_plev(Yi, Y_obs_raw_plev[i]).expand_dims(dim="new", axis=1)
            for i, Yi in enumerate(Y_raw_plev)
        ]
        Y_plev = [Yi for Yi in Ytemp_plev]  # compute Root mean square error
        Y_obs_plev = [xr.DataArray(0)] * nlat_plev_fields  # Y_obs = 0
        #S_plev = [s.mean(dim=["plev", "lat"]) ** 1/np.sqrt(s.size) for s in Sigmas_plev]
        S_plev = [s.mean(dim=["plev", "lat"]) ** 1 for s in Sigmas_plev]

        W_plev = [np.ones(1)] * nlat_plev_fields
        plev_mask_flatten = np.repeat([True], nlat_plev_fields)

    # add global fields if any
    if len(global_fields) > 0:
        if 'RESTOM' in global_fields:
            Y_global = [np.reshape(Yi.values, (Yi.values.size,1)) for Yi in Y_raw_global]
            Y_obs_global = [np.reshape(Yi,(1)) for Yi in Y_obs_raw_global]
            S_global = [np.reshape(1.0,(1))]
    else:
        Y_global = []
        Y_obs_global = []
        S_global = []

    # convert xarray to numpy arrays, combine everything
    Y_lat_lon = [Yi.values for Yi in Y]
    Y_plev_lat = [Yi.values for Yi in Y_plev]
    Y = Y_lat_lon + Y_plev_lat + Y_global

    Y_lat_lon_obs = [Yi.values for Yi in Y_obs]
    Y_plev_lat_obs = [Yi.values for Yi in Y_obs_plev]
    Y_obs = Y_lat_lon_obs + Y_plev_lat_obs + Y_obs_global

    S_lat_lon = [s.values for s in S]
    S_plev_lat = [s.values for s in S_plev]
    S = S_lat_lon + S_plev_lat + S_global
    
    # for each spatial gridpoint, calculate the variance of Y/S over the 250 simulations
    # then, remake S_global so that it matches the average
    if len(global_fields) > 0:
        var_over_sims = np.array([(Y[i]/S[i]).var(axis = 0).mean() for i in range(len(Y)-1)])
        S_global = [np.sqrt(Y_global[0].var()) / np.sqrt(var_over_sims.mean())]

    S = S_lat_lon + S_plev_lat + S_global

    return Y, Y_obs, S, W, W_plev, scalar_function, scalar_function_plev, surrogate_scorer, plev_mask_flatten


# options for Polynomial Chaos Expansion, computation 
n_cores_half = int(mp.cpu_count() / 2)
pce_grid = [
    {
        "order": list(range(1, 12)),
        "mindex_type": ["total_order", "hyperbolic"],
        "fit_type": ["linear", "ElasticNetCV"],
        "fit_params": [
            {
                "alphas": np.logspace(-8, 4, 20),
                "max_iter": 100000,
                "tol": 5e-2,
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

mlp_grid = {
    "hidden_layer_sizes": [
        (64,) * 2,
        (64,) * 4,
        (128,) * 2,
        (128,) * 6,
        (256,) * 4,
        (512,) * 2,
        (512,) * 4,
        (1024,) * 1,
    ],
    "solver": ["sgd", "adam"],
    "activation": ["relu"],
    "max_iter": [10000],
    "batch_size": ["auto"],
    "learning_rate": ["invscaling", "adaptive"],
    # "alpha": [1e-3, 1e-4],
    # "tol": [1e-3],
    # "random_state": [0],
}

# random forest regressor
rf_grid = {
    "n_estimators": [200, 500, 1000],
    "max_features": ["sqrt", "log2", 1],
    "max_depth": [2, 5, 10, 15],
}

# gaussian process regressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, Matern

gpr_grid = {
    "kernel": [
        1.0 * RBF(0.1) + 0.01**2 * WhiteKernel(0.01),
        1.0 * RBF(0.1) + 0.01**2 * WhiteKernel(0.01) + 1.0 * DotProduct(0.1),
        1.0 * Matern(length_scale=0.1, nu=1.5) + 0.1**2 * WhiteKernel(0.1),
    ],
    "alpha": [1e-10],
    "optimizer": ["fmin_l_bfgs_b"],
    "n_restarts_optimizer": [10],
    "random_state": [0, 99],
}

model_grid_dict = {
    "pce": pce_grid,
    "mlp": mlp_grid,
    "rf": rf_grid,
    "gpr": gpr_grid,
}




def compute_cv_score(
    surrogate, X, y, regressor="pce", target_transform=None, scoring="r2"
):
    """compute cross-validation score based on surrogate

    Parameters
    ----------
    surrogate : 
        constructed surrogate
    X : 
        input parameters changes
    y : 
        E3SM model output


    Returns
    -------
    scores.mean()
        mean cv score across folds
    surrogate_clone
        a copy of surrogate
    """

    # First clone the surrogate using the best hyper parameters
    n_components = len(surrogate.best_params_)
    reg_custom_list = [regressor for i in range(n_components)]
    reg_param_list = surrogate.best_params_

    if target_transform is None:
        target_transform = surrogate.TT

    surrogate_clone = tesuract.MRegressionWrapperCV(
        regressor=reg_custom_list,
        reg_params=reg_param_list,
        custom_params=True,
        target_transform=target_transform,
        target_transform_params={},
        n_jobs=-1,
        verbose=0,
    )

    scores = sklearn.model_selection.cross_val_score(
        surrogate_clone, X, y, scoring=scoring, n_jobs=-1
    )
    # print("Mean CV score:", scores.mean())

    return scores.mean(), surrogate_clone


def compute_cv_score_multiple(
    self, X, y, regressor="pce", target_transform=None, scoring="r2"
):
    """compute cross-validation score based on surrogate

    main difference with compute_cv_score is that it can compute multiple metrics  (r2, RMSE, median absolute error,etc.)
    
    Parameters
    ----------
    surrogate :
        constructed surrogate
    X :
        input parameters changes
    y :
        E3SM model output


    Returns
    -------
    scores.mean()
        mean cv score across folds
    surrogate_clone
        a copy of surrogate
    """

    # First clone the surrogate using the best hyper parameters
    n_components = len(self.best_params_)
    reg_custom_list = [regressor for i in range(n_components)]
    reg_param_list = self.best_params_
    if target_transform is None:
        # only works if n_comp is set to exact value
        # will not work if using "auto"
        target_transform = self.TT
    surrogate_clone = tesuract.MRegressionWrapperCV(
        regressor=reg_custom_list,
        reg_params=reg_param_list,
        custom_params=True,
        target_transform=target_transform,
        target_transform_params={},
        n_jobs=-1,
        verbose=0,
    )
    scores = sklearn.model_selection.cross_validate(surrogate_clone, X, y, scoring=scoring, n_jobs=-1)
    mean_scores = {k: np.mean(v) for k, v in scores.items()}
    return mean_scores, surrogate_clone
