"""
Title: Bayesian calibration using emcee sampler
Authors: Kenny Chowdhary, Julian Cooper
Purpose: Helper functions for x, y, z ...

"""
import os
import math
import pathlib
import numpy as np

from autocorr import integrated_time


def my_scorer(ytrue, ypred):
    """ Returns negative mean square error """
    mse = np.mean((ytrue - ypred)**2)/np.mean(ytrue**2)
    return -mse


def log_prob_single_param(surrogate, x: float, y: np.array, 
    param_idx: int, sigma=15) -> float:
    """
    Function that takes input x for a single parameter and returns surrogate prediction error

    :params x: float, input value for predictor variable associated with param_num
    :param param_idx: int, index associated with SWCF parameter of interest
    :returns: float, prediction error
    """
    x0 = np.array([0., 0., 0., 0., 0.])
    x0[param_idx] = x
    sigma = np.array([sigma])

    return log_prob(x=np.concatenate([x0, sigma]), y=y, surrogate=surrogate)

# use a weighted likelihood to account for changes in grid area per latitude
file_path = pathlib.Path(__file__).parent.resolve()
weights = np.loadtxt(os.path.join(file_path, "weight_tensor_24x48.txt")).flatten()[np.newaxis, :]
def log_like(y, x, sigma, surrogate, weighted=True):
    """ implements log likelihood """
    # should be similar to the surrogate error function. Look at the notes
    # we want to maximize the returned value of this function
    n = y.shape[1]
    sigma_sq = sigma**2
    y_pred = surrogate.predict(x)
    if weighted:
        error = (0.5/sigma_sq) * n * np.sum((y_pred - y)**2 * weights)
    else:
        error = (0.5/sigma_sq) * np.sum((y_pred - y)**2)
    # relative_error = (0.5/sigma_sq) * np.sum((y_pred - y)**2)/np.sum(y**2)
    log_like = - 0.5*np.log(2*math.pi*sigma_sq)*n - error

    return log_like


def log_prior_x(x):
    """ x is 0 on [-1,1]^5 and -inf elsewhere, or -1e12 """
    # interval = [-1, 1]    # x has already been scaled
    # bool_x = (x >= interval[0]) & (x <= interval[1])
    # log_prior_x = 0 if bool_x.all() else -1e12  # log(0) ~ -inf, which we estimate as -1e12

    # equivalent to checking |x| > 1
    if np.amax(np.abs(x)) > 1:
        log_prior_x = -1e12
    else:
        log_prior_x = 0

    return log_prior_x


def log_prior_tau(sigma, k, theta):
    """ use the gamma prior and set default values to k and theta """ 
    # suggest gamma parameters of k=2 and theta=0.1
    # https://homepage.divms.uiowa.edu/~mbognar/applets/gamma.html
    sigma_sq = sigma**2
    tau = 1/sigma_sq

    # prior_tau = (1/(gamma_k*theta**k)) * (tau**(k-1)) * np.exp(-tau/theta)
    # don't need a check on sigma > 0 since it is squared
    log_prior_tau = (k - 1)*np.log(tau) - (tau / theta) #if sigma >= 0 else -1e12

    return log_prior_tau


def log_prob(x, y, surrogate):
    """ define log prob of x and sigma given y """
    # log prob p(x,sigma|y) \propto log p(y|x,sigma) + log prior(x,sigma)
    #input for x[-1] should be log(sigma)
    params = x[:-1]
    sigma = np.exp(x[-1])

    # prepare log priors
    log_prior_x_value = log_prior_x(params)
    log_prior_tau_value = log_prior_tau(sigma, k=2, theta=0.1)

    # check log_prior_x in bounds
    if (log_prior_x_value == 0) and (log_prior_tau_value > -1e12):

        # evaluate log likelihood
        log_like_value = log_like(y, params, sigma, surrogate)

        # sum log prior and log likelihood terms
        log_prob = log_like_value + log_prior_x_value + log_prior_tau_value

    else:
        log_prob = -1e12

    return log_prob

def log_prob_for_mle(x, y, surrogate):
    """ define log prob of x and sigma given y """
    # input is log(sigma) so that is unbounded
    # log prob p(x,sigma|y) \propto log p(y|x,sigma) + log prior(x,sigma)
    params = x[:-1]
    sigma = np.exp(x[-1]) # invert to obtain sigma, e.g., exp(log(s)) = s

    # prepare log priors
    log_prior_x_value = log_prior_x(params)
    log_prior_tau_value = log_prior_tau(sigma, k=2, theta=0.1)

    # log_like_value = log_like(y, params, sigma, surrogate)

    # check log_prior_x in bounds
    if (log_prior_x_value == 0) and (log_prior_tau_value > -1e12):

        # evaluate log likelihood
        log_like_value = log_like(y, params, sigma, surrogate)

        # sum log prior and log likelihood terms
        log_prob = log_like_value + log_prior_x_value + log_prior_tau_value

    else:
        log_prob = -1e12

    return log_prob


def log_prob_gradient(x, y_ref, surrogate, h=1e-8):
    """ compute gradient for each dimension """
    # TODO: if x out of bounds, return some sensible gradient manually
    # TODO: lookup bounded no u-turn sampler

    # initialise graident
    gradient_vec = np.array([0.]*(len(x)))

    # calculate f_x
    x_clone = x.copy()
    x_clone[-1] = np.exp(x[-1])
    f_x = log_prob(x_clone, y=y_ref, surrogate=surrogate)

    # create integrator matrix H
    H = np.identity(6) * h

    # calculate partial derivative x1 -> x6
    # add for loop logic to create entry for each para
    Xh = np.ones([len(x), len(x)]) * x + H
    Xh[:, -1] = np.exp(Xh[:, -1])
    f_Xh = np.array([log_prob(Xh_i, y=y_ref, surrogate=surrogate) for Xh_i in Xh])

    # compute graident
    gradient_vec = (f_Xh - f_x) / h

    return f_x, gradient_vec


def neg_log_prob_gradient(x, y_ref, surrogate):
    """ negative variant for log prob gradient for descent func"""
    f_x, gradient_vec = log_prob_gradient(x=x, y_ref=y_ref, surrogate=surrogate)
    return -f_x, gradient_vec


def surrogate_error_single_param(surrogate, x: float, param_num: int, Y_ref: np.array) -> float:
    """
    Function that takes input x for a single parameter and returns surrogate prediction error

    :params x: float, input value for predictor variable associated with param_num
    :param param_num: int, index associated with SWCF parameter of interest
    :returns: float, prediction error
    """
    x0 = np.array([0., 0., 0., 0., 0.])
    x0[param_num] = x
    Y_tilde = surrogate.predict(x0)
    error = np.mean((Y_tilde - Y_ref)**2)

    return error


def surrogate_error_five_params(surrogate, x: float, Y_ref: np.array) -> float:  
    """
    Function that takes array input x for five parameters and returns surrogate prediction error

    :params x: array, input values for five parameters
    :returns: float, prediction error
    """
    Y_tilde = surrogate.predict(x)
    error = np.mean((Y_tilde - Y_ref)**2)

    return error


def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i


def function_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series

    Args:
        x: The series as a 1-D numpy array.

    Returns:
        array: The autocorrelation function of the time series.

    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= acf[0]
    return acf


def function_2d(samples):
    """Implements function_1d for all parameters
    Args:
        samples: shape(nsamples, ndim) array

    Returns:
        array: 2-D numpy array of autocorrelation
    """
    output = list()
    for idx in range(samples.shape[1]):
        autocorr_result = function_1d(samples[:, idx])
        output.append(autocorr_result)

    return np.stack(output, axis=1)


def take_mean_of_walkers(samples, nwalkers=100):
    """
    Take samples (iterations*nwalkers, ndims) and 
    returns 2d array (iterations, ndims) 
    """
    output = list()
    ndims = samples.shape[1]
    for i in range(ndims):
        mean_of_walkers = np.mean(
            samples[:, i].reshape(-1, nwalkers),
            axis=1)
        output.append(mean_of_walkers)

    return np.stack(output, axis=1)


def get_autocorr_time(x, thin=1, **kwargs):
    """Compute an estimate of the autocorrelation time for each parameter
    Args:
        thin (Optional[int]): Use only every ``thin`` steps from the
            chain. The returned estimate is multiplied by ``thin`` so the
            estimated time is in units of steps, not thinned steps.
            (default: ``1``)
        discard (Optional[int]): Discard the first ``discard`` steps in
            the chain as burn-in. (default: ``0``)
    Other arguments are passed directly to
    :func:`emcee.autocorr.integrated_time`.
    Returns:
        array[ndim]: The integrated autocorrelation time estimate for the
            chain for each parameter.
    """
    # x = self.get_chain(discard=discard, thin=thin)
    return thin * integrated_time(x, **kwargs)
