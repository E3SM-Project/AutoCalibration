import numpy as np


def custom_calibration_function(Y, Y_obs):
    """Custom function d(Y,Y_obs) -> scalar valued function

    Parameters
    ----------
    Y : xarray.DataArray
        Sample data of shape (nsamples,*shape) where shape is the shape of the data, e.g., (24,48)
    Y_obs : xarray.DataArray
        Reference or observational data of size (*shape). Shape of Y_obs must match shape of Y[i] for all i

    Returns
    -------
    np.array
        Array of size (nsamples,) where each row corresponds to the metric evalution
    """
    assert (
        Y.shape[1:] == Y_obs.shape
    ), "Y and Y_obs are not compatible. Might return incorrect results."
    #############################
    # anything below this line can be changed as long as d(Y,Y_obs) return a shape of (nsamples,)
    diff_sq = np.abs((Y - Y_obs)) ** 2
    rmses = np.sqrt(diff_sq.mean(dim=["lat", "lon"]))
    #############################
    output = rmses
    assert output.shape == (Y.shape[0],), "output must be of size (nsamples,)."
    return output
