"""
Title: Bayesian calibration using emcee sampler
Authors: Kenny Chowdhary, Julian Cooper
Purpose: Plot function for x, y, z ...

"""
import os
import corner
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from functions import surrogate_error_single_param, log_prob_single_param


def plot_multi_scatter(data: np.array, plot_dir='/'):
    """ Plots each para autocorrelation over course of sampling """
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(data.shape[1]):
        ax.plot(data[:, i], '.', ms=0.5)

    # loc = plticker.MultipleLocator(base=50.0)
    # ax.xaxis.set_major_locator(loc)

    # save figure
    file_path = os.path.join(plot_dir, 'autocorrelation_multi.png')
    plt.savefig(
        file_path,
        # bbox_inches='tight'
        )

    return None


def compare_rse_logprob(y: np.array, xlabels: list, solve_mse: np.array, solve_log_prob: np.array, sigma=15) -> None:
    """
    Plots each of the five univariate parameter error charts with
    5-dim MLE represented with red vertical line
    """
    for param_idx, param in enumerate(xlabels):
        fig, axs = plt.subplots(2)
        fig.suptitle(xlabels[param_idx]) 

        x = np.linspace(-1, 1, 100)
        y1 = [surrogate_error_single_param(x=xi, param_num=param_idx, Y_ref=y) for xi in x]
        y2 = [log_prob_single_param(x=np.array([xi]), y=y, param_idx=param_idx, sigma=sigma) for xi in x]

        axs[0].scatter(x, y1)
        axs[0].set_title('mean square error')
        axs[0].vlines(x=solve_mse[param_idx], ymin=min(y1), ymax=max(y1), linestyles='dashed', colors='red')

        axs[1].scatter(x, y2)
        axs[1].set_title('log probability')
        axs[1].vlines(x=solve_log_prob[param_idx], ymin=min(y2), ymax=max(y2), linestyles='dashed', colors='orange')

        plt.show()

    return None


def plot_square_error(solve_x: np.array, xlabels: list, param_idx: int):
    """ Plot MLE against univariate square error for input para """
    x = np.linspace(-1, 1, 100)
    y = [surrogate_error_single_param(x=xi, param_num=param_idx, Y_ref=Y_ref) for xi in x]

    plt.title(xlabels[param_idx])
    plt.scatter(x, y)
    plt.vlines(x=solve_x[param_idx], ymin=min(y), ymax=max(y), linestyles='dashed', colors='red')
    plt.show()


def plot_log_probability(Y_ref, solve_x: np.array, xlabels: list, param_idx: int, sigma=15):
    """ Plot MAP against univariate log probability for input para """
    x = np.linspace(-1, 1, 100)
    y = [log_prob_single_param(x=np.array([xi]), y=Y_ref, param_idx=param_idx, sigma=sigma) for xi in x]

    plt.title(xlabels[param_idx])
    plt.scatter(x, y)
    plt.vlines(
        x=solve_x[param_idx],
        ymin=min(y), ymax=max(y),
        linestyles='dashed',
        colors='red')
    plt.show()


def plot_bayes_univar_dist(xindex, xlabel, samples, bayes_map, log_prob_mle, plot_dir='/'):
    """ Plot bayes joint posterior ... """
    # figure dimensions
    plt.figure(figsize=(6, 8), dpi=60)

    # plot data
    plt.hist(samples, 100, color="k", histtype="step")
    plt.vlines(
        x=bayes_map,
        ymin=0,
        ymax=plt.gca().get_ylim()[1],
        linestyles='solid',
        colors='navy',
        label='MAP')
    plt.vlines(
        x=log_prob_mle,
        ymin=0,
        ymax=plt.gca().get_ylim()[1],
        linestyles='dashed',
        colors='orange',
        label='MLE')

    # formatting
    plt.xlabel(r"$\theta_{}$, {}".format(xindex, xlabel))
    plt.ylabel(r"$p(\theta_{}$)".format(xindex))
    plt.gca().set_yticks([])
    plt.legend()

    # save figure
    file_path = os.path.join(plot_dir, 'bayes_univar_dist_{}.png'.format(xlabel))
    plt.savefig(file_path)


def plot_bayes_corner_plot(samples, labels, truths=None, control = None, plot_dir='/'):
    """ Plot and save bayes covariance corner chart """
    fig = corner.corner(samples, labels=labels,quantiles=[0.16, 0.5, 0.84],truths=None,labelpad = .2)

    corner.overplot_points(fig, truths[None], marker="s", color="C1")
    corner.overplot_points(fig, control[None], marker="o", color="C2")
    # save figure
    file_path = os.path.join(plot_dir, 'bayes_corner_plot.png')
    plt.savefig(
        file_path,
	pad_inches = 0.25,
        bbox_inches='tight'
        )
def plot_bayes_corner_plot_range(samples, labels, ranges, truths=None, control= None, plot_dir='/'):
    """ Plot and save bayes covariance corner chart """
    fig = corner.corner(samples, labels=labels,quantiles=[0.16, 0.5, 0.84],truths=None,range=ranges,labelpad = .2)
    corner.overplot_points(fig, truths[None], marker="s", color="C1")
    corner.overplot_points(fig, control[None], marker="o", color="C2")

    # save figure
    file_path = os.path.join(plot_dir, 'bayes_corner_plot_range.png')
    plt.savefig(
        file_path,
	pad_inches = 0.25,
        bbox_inches='tight'
        )

def plot_trace_for_param(samples_x, xlabel, xindex, plot_dir='/'):
    """ Plot trace, with each walker treated as a separate chain """
    # figure dimensions
    plt.figure(figsize=(12, 6), dpi=60)

    # plot data
    plt.plot(samples_x, linewidth=0.2, color="black")

    # formatting
    plt.xlabel("step number")
    plt.ylabel(r"$\theta_{}$".format(xindex))

    # save figure
    file_path = os.path.join(
        plot_dir,
        'trace_plot_{}.png'.format(xlabel)
    )
    plt.savefig(file_path)
