# Import libraries for your custom cost function
import numpy as np
import xarray as xr


# Specify weights for each cost component (required) 
weights = {
    "RESTOM": 0.2, # 0.1,
    "dnet_cld_dir": 8.0, # 8.0
    "DJF_SWCF": 1.25,
    "MAM_SWCF": 1.25,
    "JJA_SWCF": 1.25,
    "SON_SWCF": 1.25,
    "DJF_LWCF": 1.75,
    "MAM_LWCF": 1.75,
    "JJA_LWCF": 1.75,
    "SON_LWCF": 1.75,
    "DJF_PRECT": 1.5, # 1.5
    "MAM_PRECT": 1.5, # 1.5
    "JJA_PRECT": 1.5, # 1.5
    "SON_PRECT": 1.5, # 1.5
    "DJF_TREFHT": 1.0,
    "MAM_TREFHT": 1.0,
    "JJA_TREFHT": 1.0,
    "SON_TREFHT": 1.0,    
    "DJF_PSL": 0.75,
    "MAM_PSL": 0.75,
    "JJA_PSL": 0.75,
    "SON_PSL": 0.75,    
    "DJF_Z500": 0.4,
    "MAM_Z500": 0.4,
    "JJA_Z500": 0.4,
    "SON_Z500": 0.4,    
    "DJF_U850": 1.25,
    "MAM_U850": 1.25,
    "JJA_U850": 1.25,
    "SON_U850": 1.25,    
    "DJF_U200": 1.5,
    "MAM_U200": 1.5,
    "JJA_U200": 1.5,
    "SON_U200": 1.5,    
    "DJF_RELHUM": 0.15,
    "MAM_RELHUM": 0.15,
    "JJA_RELHUM": 0.15,
    "SON_RELHUM": 0.15,    
    "DJF_T": 1.0,
    "MAM_T": 1.0,
    "JJA_T": 1.0,
    "SON_T": 1.0,    
    "DJF_U": 1.5,
    "MAM_U": 1.5,
    "JJA_U": 1.5,
    "SON_U": 1.5,
    }


# Auxiliary functions
def rmse(y, yhat, area_weights=None, s=1.0):
    """This function is needed for the template example. You can create other functions with different names and different parameters for your custom cost_function
    
    Parameters
    ----------
    y : numpy array of shape (ncol,) containing observations of a vectorized output variable
    yhat : numpy array of shape (ncol,) containing predictions of a vectorized output variable
    area_weights : a numpy array of shape (ncol,) containing area weights for each column; defaults to equal weights
    s : normalization constant for the output variable

    Returns 
    ----------
    rmse_out : a single-number calculation of RMSE, weighted by area_weights and normlaized by s
    """

    if area_weights is None:
        area_weights = 1/len(y)

    rmse_out = np.sqrt(np.sum(area_weights * ((y - yhat) / s)**2))

    return rmse_out


# The custom cust function
def cost_function(obs, preds, area_weights, norm_constants, lat=None, lon=None, return_components=False):
    """compute the cost of estimating "preds" when the truth is "obs" -- optimization.py is expecting a function called "cost_function" with these same parameters 

    Parameters - NOTE: optimization.py is expecting a function with these same parameters
    ----------
    obs :
        dictionary of length n_outputs, where each element is a numpy array of shape (ncol,) containing observations of a vectorized output variable (ncol may differ by output variable)  
    preds :
        dictionary of length n_outputs, analogous to "obs," i.e., each element is a numpy array of shape (ncol,) containing predictions of a vectorized output variable 
    area_weights :
        dictionary of length n_outputs, analogous to "obs" and "preds," where each element is a numpy array of shape (ncol,) containing an area weight (normalized to sum to 1) for each column of the corresponding output variable
    lat, lon :
        dictionaries of length n_outputs, analogous to "obs" and "preds," where each element is a numpy array of shape (ncol,) containing the latitude (longitude) for each column of the corresponding output variable
    return_components:
        a boolean indicating whether to return individual cost components and their weights in the overall cost 

    Returns - NOTE: optimization.py is expecting a function with these same returns
    -------
    cost:
        a single-number cost evaluation, given by the weighted sum of component_cost 
    component_cost: 
        a dictionary, where each element is a single-number evaluation of a cost component; only returned if return_component_cost is True 
    weights: 
        a dictionary, where each element is the weight of a cost component; only returned if return_component_cost is True 
    """

    # Compute component_cost
    component_cost = {output_var: None for output_var in weights}
    for output_var in obs:
        if "SWCF" in output_var:
            mask = (lat[output_var] >= -60)  &  (lat[output_var] <= 60)
            component_cost[output_var] = rmse(obs[output_var][mask], preds[output_var].squeeze()[mask], area_weights[output_var][mask] / np.sum(area_weights[output_var][mask]), norm_constants[output_var][mask]) 
        else:
            component_cost[output_var] = rmse(obs[output_var], preds[output_var], area_weights[output_var], norm_constants[output_var]) 
    assert list(weights.keys()) == list(component_cost.keys()), "'weights' and 'component_cost' keys must match, including their order"

    # Compute overall cost
    cost = np.dot(list(component_cost.values()), list(weights.values()))
     
    if return_components:
        return cost, component_cost, weights
    else:
        return cost

