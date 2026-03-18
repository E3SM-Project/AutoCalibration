import numpy as np
import tesuract
import sklearn
import xarray as xr
import os
import clif.preprocessing as cpp
import glob
import pdb
import pickle

def get_idx_keep(X, X_bounds):
    idx_keep = [idx for idx in range(len(X)) if np.all(X[idx] >= X_bounds[:,0]) and np.all(X[idx] <= X_bounds[:,1])]
    return idx_keep

def load_data(obs_path, ens_path, responses, param_bounds = None, enforce_bounds = False, return_lat_lon=False):
    """
    Returns
    -------
    Y_raw : list 
        list of length n_responses, each with first dimension governing nsim simulation runs, xarray time-averaged-simulation results
    Y_obs_raw : list
        list of length n_responses, with one fewer dimension compared to Y_raw, that are xarray time-averaged observations
    n_responses : int
        number of response variables
    normalized_area_weights : xarray.DataArray
        xarray object that describes area taken up by each grid point, with normalized_area_weights.sum() = 1
    area_weights : xarray.DataArray
        xarray object that describes area taken up by each grid point, with area_weights.sum() != 1
    """
    ens_path_list = glob.glob(ens_path)
    obs_path_list = glob.glob(obs_path)

    Y_raw = []; Y_obs_raw = []; area_weights = []; normalized_area_weights = []; lat_list = []; lon_list = []
    X_list = []; x_labels_list = []; workdir_list = []
    for src_name in responses.keys():
        # load ensemble data
        ens_path_src = [path for path in ens_path_list if src_name in path]
        if len(ens_path_src) == 1:
            ens_path_src = ens_path_src[0]
        elif len(ens_path_src) == 0:
            raise Exception(f"No files match {ens_path} AND contain {src_name}")
        else:
            raise Exception(f"Multiple files match {ens_path} AND contain {src_name}")
        dataset_ens = xr.open_dataset(ens_path_src)
        X_list.append(dataset_ens.params)
        x_labels_list.append(dataset_ens.input_params)
        workdir_list.append(dataset_ens.workdir.values)

        # load observations
        obs_path_src = [path for path in obs_path_list if src_name in path]
        if len(obs_path_src) == 1:
            obs_path_src = obs_path_src[0]
        elif len(obs_path_src) == 0:
            raise Exception(f"No files match {obs_path} AND contain {src_name}")
        else:
            raise Exception(f"Multiple files match {obs_path} AND contain {src_name}")    
        dataset_obs = xr.open_dataset(obs_path_src)
        
        # extract desired responses
        for r in responses[src_name].values():
            # ensemble
            Y_raw.append(dataset_ens[r])
            
            # area weights
            if 'area' in dataset_obs:
                W = dataset_obs.area
                dims_to_add = {}
                dim_axis_to_add = []
                for dim in dataset_obs[r].dims:
                    if dim not in W.dims: # expand area weights across time and/or other dimensions
                        dim_axis = dataset_obs[r].dims.index(dim)
                        dim_axis_to_add.append(dim_axis)
                        dims_to_add[dim] = dataset_obs[r].shape[dim_axis]
                if len(dims_to_add) > 0:
                    W = W.expand_dims(dims_to_add, dim_axis_to_add)
                for dim in W.dims:
                    if dim not in dataset_obs[r].dims: # expand area weights across time and/or other dimensions
                        W = W.isel(**{dim: 0})
                area_weights.append(W)
                normalized_area_weights.append(W / W.sum())
            else:
                W = xr.DataArray(np.ones(np.shape(dataset_obs[r])))
                area_weights.append(W)
                normalized_area_weights.append(W / W.sum())
            
            # obs
            Y_obs_raw.append(dataset_obs[r].squeeze())

            if return_lat_lon:
                dims = dataset_obs[r].squeeze().dims
                shape = dataset_obs[r].squeeze().shape
                
                if len(dims) > 0 and dims[0] == 'lat':
                    lat_list.append(np.repeat(dataset_obs['lat'].values, shape[1]))
                elif len(dims) > 1 and dims[1] == 'lat':
                    lat_list.append(np.tile(dataset_obs['lat'].values, shape[0]))
                else:
                    lat_list.append(None)
                    
                if len(dims) > 0 and dims[0] == 'lon':
                    lon_list.append(np.repeat(dataset_obs['lon'].values, shape[1]))
                elif len(dims) > 1 and dims[1] == 'lon':
                    lon_list.append(np.tile(dataset_obs['lon'].values, shape[0]))
                else:
                    lon_list.append(None)
    
    # get rid of workdirs if they don't contain all variables
    common_workdirs = np.array(list(set.intersection(*[set(wd) for wd in workdir_list])))
    idx_keep = np.array([idx for idx, wd in enumerate(workdir_list[0]) if wd in common_workdirs])
    workdirs = workdir_list[0][idx_keep]
    X = X_list[0][idx_keep]
    
    n_responses = len(Y_raw)
    for i in range(n_responses):
        idx_keep = [idx for idx in range(len(Y_raw[i])) if Y_raw[i].workdir.values[idx] in workdirs]
        Y_raw[i] = Y_raw[i][idx_keep]
    
    assert all([(xl == x_labels_list[0]).all() for xl in x_labels_list]), "X labels should be the same for all datasets"
    x_labels = x_labels_list[0].values
    n_inputs = len(x_labels)
    if param_bounds is None: # get parameter bounds from ensemble
        X_bounds = np.vstack([np.min(X, axis=0),
                              np.max(X, axis=0)]).T
    else:
        X_bounds = np.array([[float(x[0]), float(x[1])] for x in list(param_bounds.values())])

    if enforce_bounds:
        idx_keep = get_idx_keep(X, X_bounds) 
        X = X[idx_keep]
        Y_raw = [Yi[idx_keep] for Yi in Y_raw]
        workdirs = workdirs[idx_keep]

    out = (X, x_labels, X_bounds, n_inputs, Y_raw, Y_obs_raw, n_responses, normalized_area_weights, area_weights, workdirs)

    if return_lat_lon:
        out += (lat_list, lon_list)

    return out 


def load_val_data(val_path, val_workdirs, responses, M):
    val_path_list = glob.glob(val_path)

    Y_val_raw = []; X_list = []; x_labels_list = []
    for src_name in responses.keys():
        val_path_src = [path for path in val_path_list if src_name in path]
        if len(val_path_src) == 1:
            val_path_src = val_path_src[0]
        elif len(val_path_src) == 0:
            raise Exception(f"No files match {val_path} AND contain {src_name}")
        else:
            raise Exception(f"Multiple files match {val_path} AND contain {src_name}")
        dataset_src = xr.open_dataset(val_path_src)
        dataset_src = dataset_src.sel(workdir=dataset_src.workdir.isin(val_workdirs))

        # response_names = list(responses[src_name].values())
        # response_names += ['params']
        # dataset_src = []
        # for wd in val_names:
        #     wd_path = os.path.join(datadir, src_name + "_" + wd + ".nc")
        #     dataset_wd = xr.open_dataset(wd_path)[response_names]
        #     dataset_src.append(dataset_wd)

        # if len(dataset_src)==1:
        #     dataset_src = dataset_src[0].expand_dims(dim = 'workdir')
        # else:
        #     dataset_src = xr.concat(dataset_src, dim = 'workdir')
        # dataset_src = dataset_src.assign_coords(workdir = val_names)
        X_list.append(dataset_src.params)
        x_labels_list.append(dataset_src.input_params)
        
        for r in responses[src_name].values():
            Y_val_raw.append(dataset_src[r])

    nsim = len(Y_val_raw[0])
    n_responses = len(Y_val_raw)
    Y_val = [np.reshape(Y_val_raw[i].values, newshape=(nsim,-1))[:,M[i]] for i in range(n_responses)]
            
    # Get inputs
    assert all([(x == X_list[0]).all() for x in X_list]), "X should be the same for all datasets"
    X_val = X_list[0]
    assert all([(xl == x_labels_list[0]).all() for xl in x_labels_list]), "X labels should be the same for all datasets"
    
    return X_val, Y_val_raw, Y_val
    

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


# create response transforms for multi-objective optimization
class response_transform(sklearn.base.TransformerMixin):
    def __init__(self, response_transforms, joinT, response_names):
        self.response_transforms = response_transforms
        self.joinT = joinT
        self.response_names = response_names

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, X_is_std=False, mean=None):
        if self.response_transforms is None:
            return X
        else:
            X_recon = self.joinT.inverse_transform(X)
            for res, trans in self.response_transforms.items():
                assert (trans in ['log', 'sqrt']) or ('root' in trans),\
                    f"Transform {trans} not implemented"
                if trans == 'log':
                    if X_is_std:
                        def t(x): # Plug-in estimator assuming x ~ log-Normal
                            mn = mean[self.response_names.index(res)]
                            return np.sqrt(np.log(x**2/mn**2 + 1.0))
                    else:
                        t = np.log
                elif trans == 'sqrt':
                    t = np.sqrt
                elif 'root' in trans:
                    k = float(trans.split('root')[0])
                    def t(x):
                        return np.power(x, 1.0/k)                    
                idx_trans = self.response_names.index(res)
                X_recon[idx_trans] = t(X_recon[idx_trans])
            return self.joinT.transform(X_recon)

    def inverse_transform(self, Xhat, Xhat_is_std=False, mean=None):
        if self.response_transforms is None:
            return Xhat
        else:
            Xhat_recon = self.joinT.inverse_transform(Xhat)
            for res, trans in self.response_transforms.items():
                if trans == 'log':
                    if Xhat_is_std:
                        def t(x):
                            mn = mean[self.response_names.index(res)]
                            return (np.exp(x**2) - 1.0) * mn**2
                    else: 
                        inv_t = np.exp
                elif trans == 'sqrt':
                    def inv_t(x):
                        return np.square(x * (x > 0))
                elif 'root' in trans:
                    k = float(trans.split('root')[0])
                    def inv_t(x):
                        return np.power(x * (x > 0), k)
                else:
                    assert (trans in ['log', 'sqrt']) or ('root' in trans), f"Transform {trans} not implemented"
                idx_trans = self.response_names.index(res)
                Xhat_recon[idx_trans] = inv_t(Xhat_recon[idx_trans])
            return self.joinT.transform(Xhat_recon)


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

    
def transform_data(Y_raw, Y_obs_raw, n_responses, normalized_area_weights, response_transforms, response_names, stdz_by_ncol=True, norm_method=None, response_weights=None):
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
        
    # translate string norm_method to dictionary and extract normalization file if applicable
    if type(norm_method) is str:
        if norm_method.split('.')[-1] == 'pkl': # file containing norm constants
            with open(norm_method, 'rb') as f:
                norm_dat_raw = pickle.load(f)
            norm_dat_raw = [norm_dat_raw[response_names[i]] for i in range(n_responses)]
            norm_nan = [np.isnan(nd) for nd in norm_dat_raw]
            norm_zero = [nd == 0 for nd in norm_dat_raw]
            method = '.pkl'
        else:
            method = norm_method
        norm_method = {}
        for res_name in response_names:
            norm_method[res_name] = method
        
    # identify non-missing values in each field
    ens_nan = [np.isnan(Yi).any('workdir') for Yi in Y_raw]
    ens_allequal = [(Yi == Yi[0]).all('workdir') for Yi in Y_raw]
    obs_nan = [np.isnan(Yi) for Yi in Y_obs_raw]
    if 'norm_nan' in locals():
        mask = [~xr.concat([ens_nan[i], ens_allequal[i], obs_nan[i], norm_nan[i], norm_zero[i]], 'condition', coords='minimal').any('condition') for i in range(len(Y_raw))]
    else:
        mask = [~xr.concat([ens_nan[i], ens_allequal[i], obs_nan[i]], 'condition').any('condition') for i in range(len(Y_raw))]
    # vectorize everything and downsample according to mask
    M = [mask[i].values.flatten() for i in range(n_responses)]
    Y_obs = [Y_obs_raw[i].values.flatten()[M[i]] for i in range(n_responses)]
    W = [normalized_area_weights[i].values.flatten()[M[i]] for i in range(n_responses)]
    
    nsim = len(Y_raw[0])
    Y = [np.reshape(Y_raw[i].values, newshape=(nsim,-1))[:,M[i]] for i in range(n_responses)]
    
    # compute normalizing constant for each transformed field
    joinT = JoinTransform()
    Y_joined = joinT.fit_transform(Y)
    trans = response_transform(response_transforms, joinT, response_names)
    Y_trans_joined = trans.fit_transform(Y_joined)
    Y_trans = joinT.inverse_transform(Y_trans_joined)

    Y_obs_joined = joinT.transform(Y_obs)
    Y_obs_trans_joined = trans.transform(Y_obs_joined)
    Y_obs_trans = joinT.inverse_transform(Y_obs_trans_joined)

    if 'norm_dat_raw' in locals():
        norm_dat = [norm_dat_raw[i].values.flatten()[M[i]] for i in range(n_responses)]
        norm_dat_trans = joinT.inverse_transform(trans.transform(joinT.transform(norm_dat),
                                                                 X_is_std = True,
                                                                 mean = Y_obs))
    
    if stdz_by_ncol:
        norm_c = np.sqrt(np.max([len(Yi) for Yi in Y_obs_trans]))
        S = [np.repeat(np.sqrt(len(Yi))/norm_c, len(Yi)) for Yi in Y_obs_trans]
    else:
        S = [np.ones(len(Yi)) for Yi in Y_obs_trans]

    if response_weights is not None:
        for res_name,weight in response_weights.items():
            idx_res = response_names.index(res_name)
            S[idx_res] /= weight

    if norm_method is not None:
        for res_name,method in norm_method.items(): 
            assert method in ['obs_field_std', 'obs_field_median', 'obs_field_q', 'ens_col_std', '.pkl'],\
                f"norm_method '{method}' is not implemented"
            idx_res = response_names.index(res_name)
            if method == 'obs_field_std':
                norm = np.std(Y_obs_trans[idx_res])
            elif method == 'obs_field_median':
                norm = np.median(Y_obs_trans[idx_res])
            elif 'obs_field_q' in method:
                k = float(method.split('q')[1]) / 100.0
                norm = np.quantile(Y_obs_trans[idx_res], k)
            elif method == 'ens_col_std':
                norm = np.std(Y_trans[idx_res], axis = 0)
            elif method == '.pkl': # assume that norm_method is a file path
                norm = norm_dat_trans[idx_res]
                
            S[idx_res] *= norm

    sigmas_joined = joinT.transform(S)
    scalingT = SigmaScaling(sigmas_joined)
    Y_s = scalingT.fit_transform(Y_trans_joined)
    
    return Y, Y_obs, joinT, Y_joined, trans, Y_obs_joined, scalingT, Y_s, S, W, M, mask


def transform_val_data(Y_val_raw, n_responses, M):
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

    # vectorize everything and downsample according to mask
    nval=len(Y_val_raw[0])
    Y_val = [np.reshape(Y_val_raw[i].values, newshape=(nval,-1))[:,M[i]] for i in range(n_responses)]

    return Y_val
