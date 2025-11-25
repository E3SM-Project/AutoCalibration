import numpy as np


def custom_surrogate_fit_function(Y_true, Y_pred):
    """Custom function d(Y,Y_obs) -> scalar valued function

    Parameters
    ----------
    Returns
    -------
    """
    assert (
        Y_true.shape == Y_pred.shape
    ), "Y_true and Y_pred are not compatible. Might return incorrect results."
    #############################
    value = -np.median(np.abs(Y_pred - Y_true))
    #value = -np.sqrt(np.mean((Y_pred - Y_true)**2))
    
    return value
