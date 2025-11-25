from prettytable import PrettyTable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sklearn
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
import netCDF4 as nc
import numpy as np
import regression_params
import os
import tesuract
import xarray as xr
import git
import pdb
import pickle

import preprocessing

###################
def paste_nonempty(words, sep=""):
    words_copy = words.copy()
    if "" in words_copy:
        words_copy.remove("")
    return sep.join(words_copy)

def save_predictions(savedir, savename, X, Y_joined, feature_transform, surrogate, M, workdirs):
    """save surrogate predictions at perturbed parameters

    in netcdf file

    Parameters
    ----------
    filename :
        information on setup to save in filename
    savedir :
        where to save predictions
    X_s :
        perturbed parameter info
    surrogate :
        fitted surrogate
    M:
        list of spatial masks, one per variable. 

    others :
        make sure plev fields are handled correctly since the masking is hard
    """
    X_s = feature_transform.fit_transform(X)
    Y_pred = surrogate.predict(X_s)
   
    pred_filename = paste_nonempty(["pred", savename], "_") + ".nc"
    full_pred_save_path = os.path.join(savedir, pred_filename)

    ds_nc = nc.Dataset(full_pred_save_path, 'w', format='NETCDF4')
    ens_idx = ds_nc.createDimension('ens_idx', X.shape[0])
    input_param = ds_nc.createDimension('input_param', X.shape[1])
    var = ds_nc.createDimension('var', Y_pred.shape[1])

    workdir = ds_nc.createVariable('workdir', str, 'ens_idx')
    workdir[:] = workdirs
    params = ds_nc.createVariable('params', 'f4', ('ens_idx', 'input_param'))
    params[:] = X 
    surrogate_preds = ds_nc.createVariable('surrogate_preds', 'f4', ('ens_idx', 'var'))
    surrogate_preds[:] = Y_pred
    model_output = ds_nc.createVariable('model_output', 'f4', ('ens_idx', 'var'))
    model_output[:] = Y_joined
    ds_nc.close()

    #write the mask
    with open(os.path.join(savedir, 'mask.pkl'), 'wb') as f:
        pickle.dump(M, f)
    
def save_pc_info(savedir, savename, X_s, surrogate, target_transform, n_components, Y_joined, Y_obs_joined):
    """save principal component info

    in netcdf file

    Parameters
    ----------
    filename :
        information on setup to save in filename
    savedir :
        where to save predictions
    X_s :
        perturbed parameter info
    surrogate :
        fitted surrogate
    """
    
    pc_filename = paste_nonempty(["pc", savename], "_") + ".nc"
    full_pc_save_path = os.path.join(savedir, pc_filename)
    
    Y_pred = surrogate.predict(X_s)
    pc_scores_surr_vals = target_transform.transform(Y_pred)
    
    ds_nc = nc.Dataset(full_pc_save_path, 'w', format='NETCDF4')
    X = ds_nc.createDimension('X', X_s.shape[0])
    X_single = ds_nc.createDimension('X_single', 1)
    index = ds_nc.createDimension('index', Y_obs_joined.shape[0])
    comp = ds_nc.createDimension('comp', n_components)
    
    # PC scores
    # Principal component scores of each simulation run
    pc_scores_model = ds_nc.createVariable('pc_scores_model', 'f4', ('X', 'comp'))
    pc_scores_model[:] = target_transform.transform(Y_joined)
     # Principal component scores of each surrogate prediction
    pc_scores_surr = ds_nc.createVariable('pc_scores_surr', 'f4', ('X', 'comp'))
    pc_scores_surr[:] = target_transform.transform(Y_pred)
     # Principal component scores of obs
    pc_scores_obs = ds_nc.createVariable('pc_scores_obs', 'f4', ('X_single', 'comp'))
    pc_scores_obs[:] =  target_transform.transform(np.reshape(Y_obs_joined, (1, Y_obs_joined.shape[0])))
    
    # other PC info
    #  pc_vec: principal component vectors
    #  ev:  explained  variance of PCs used
    #  sing_vals:  singular values from decomposition
    #  prop_var: proportion of variance explained by that component 
    
    pc_vecs = ds_nc.createVariable('pc_vals', 'f4', ('comp', 'index'))
    pc_vecs[:] = target_transform['pca'].components_
    ev = ds_nc.createVariable('ev', 'f4', ('comp'))
    ev[:] = target_transform['pca'].explained_variance_
    sing_vals = ds_nc.createVariable('sing_vals', 'f4', ('comp'))
    sing_vals[:] = target_transform['pca'].singular_values_
    
    prop_var = ds_nc.createVariable('prop_var', 'f4', ('comp'))
    prop_var[:] = surrogate.target_transform['pca'].explained_variance_ratio_
    ds_nc.close()


###################
# printing results from optimization
###################
# printing model summary...

def print_opt_params(
    xopt_,
    x_labels,
    cost_function_name,
    R,
    feature_dim,
    solver="L-BFGS-B",
    filename=None,
):
    """print optimization results

    in a .npy and .txt file
    """

    title = f"Optimized Solution for Cost Function '{cost_function_name}' ({R} cores using {solver})"

    # solution
    table_opt = PrettyTable()
    table_opt.title = title
    table_opt.field_names = ["Optimal parameters", "Values"]
    for j in range(feature_dim):
        table_opt.add_row([x_labels[j], "{0:8.8f}".format(xopt_[j])])
    table_opt.align = "r"
    print(table_opt)

    if filename is not None:
        with open(filename, "w") as file:
            for j in range(feature_dim):
                file.write(x_labels[j] + " = " + str(xopt_[j]) + "\n")

    return table_opt


def load_params(
    filename, x_labels
):
    """print optimization results

    in a .npy and .txt file
    """
    params = {}
    # fill in the dictionary    
    with open(filename, "r") as file:
        for line in file:
            line_split = line.split('=')
            param_label = line_split[0].strip()
            if param_label in x_labels:
                param_str = line_split[1].strip()
                params[param_label] = float(param_str.replace('D', 'E'))
    return list(params.values())


def save_specs(dirname, workdirs, save_surrogate_specs=False, save_optimization_specs=False):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
    files_to_cp = ['preprocessing.py', 'postprocessing.py', '../README.md']
    dest_files = files_to_cp.copy()
    dest_files[-1] = 'README.md'
 
    if save_surrogate_specs:
        surr_files = ['surrogate.py', 'config_surrogate.yaml', 'regression_params.py']
        files_to_cp += surr_files
        dest_files += surr_files 
   
    if save_optimization_specs:
        opt_files = ['optimization.py', 'config_optimization.yaml']
        files_to_cp += opt_files
        dest_files += opt_files 
        os.system('cp -r ../cost_functions' + ' ' + os.path.join(dirname, 'cost_functions'))

    for i in range(len(files_to_cp)):
        os.system('cp ./' + files_to_cp[i] + ' ' + os.path.join(dirname, dest_files[i]))

    # create file listing workdirs included in training the surrogate
    workdirs = sorted(workdirs)
    with open(os.path.join(dirname, "workdirs"), 'w') as f:
        for wd in workdirs:
            f.write(wd + "\n")

            
def save_environment(dirname, machine):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    git_repo = git.Repo(search_parent_directories=True) 
    git_root = git_repo.git.rev_parse('--show-toplevel')

    files_to_cp = [f'autotuning_environment_{machine}.yml', f'autotuning_spec_file_{machine}.txt', 'README.md', f'requirements_{machine}.txt']
    for f in files_to_cp:
        os.system('cp ' + os.path.join(git_root, f) + ' ' + os.path.join(dirname, f))
    
    # create environment provenance file
    git_hash = git_repo.head.object.hexsha # what was the current git hash (aka git sha)?
    model_dir = os.path.abspath("./") # where did we run this script?
    with open(os.path.join(dirname, "provenance"), 'w') as f:
        f.write("git hash: " + git_hash + "\n")
        f.write("machine: " + machine + "\n")
        f.write("model-fitting directory: " + model_dir + "\n")


def root_mean_squared_error(y_true, y_pred):
    rmse_all = sklearn.metrics.root_mean_squared_error(y_true, y_pred, multioutput='raw_values')
    rmse_avg = np.mean(rmse_all)
    return rmse_avg


def compute_cv_score(
        self, X, y, regressor="pce", folds=None, prefit_params = True, n_jobs=1
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
    if prefit_params:
        n_components = len(self.best_params_)
        reg_list = [regressor for i in range(n_components)]
        reg_params = self.best_params_
        n_jobs = -1
    else:
        reg_list = [regressor]   
        reg_params = [regression_params.params[regressor]]
        n_jobs = n_jobs 

    target_transform = sklearn.pipeline.Pipeline([("pca", PCA(n_components=n_components, whiten=True))])
      
    surrogate_clone = tesuract.MRegressionWrapperCV(
        regressor=reg_list,
        reg_params=reg_params,
        custom_params=prefit_params,
        target_transform=target_transform,
        target_transform_params={},
        scorer=self.scorer,
        n_jobs=-1,
        verbose=0,
    )

    scores = sklearn.model_selection.cross_validate(surrogate_clone, X, y, scoring={'r2': make_scorer(sklearn.metrics.r2_score, multioutput='variance_weighted'), 'rmse': make_scorer(root_mean_squared_error)}, n_jobs=n_jobs, cv=folds)
    
    return scores


#### Plotting ####
def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def plot_cost_corr(savedir, name, cost_arr, cost_labels):
    n_components = len(cost_labels)

    plt.rcParams["figure.autolayout"] = False

    plt.matshow(np.corrcoef(cost_arr.T))
    plt.xticks(ticks = range(n_components), labels = cost_labels, fontsize=14, rotation=45, ha='left')
    plt.yticks(ticks = range(n_components), labels = cost_labels, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Cost Component Correlation', fontsize=20);

    filename = os.path.join(savedir, f"cost_corr_{name}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close('all')

def plot_components_v_total_cost(savedir, name, component_arr, total_cost, component_labels):
    plt.rcParams["figure.figsize"] = [7.0, 5.0]
    plt.rcParams["figure.autolayout"] = True

    n_components = len(component_labels) 
    if n_components > 6:
        nrow,ncol = 3,3
    elif n_components > 4:
        nrow,ncol = 2,3
    else:
        nrow,ncol = 2,2
    for i in range(n_components + (nrow*ncol) - ((n_components-1) % (nrow*ncol) + 1)):
        idx = i % (nrow*ncol)
        if idx == 0:
            fig, ax = plt.subplots(nrow, ncol)
            fig.suptitle('Component Cost vs. Total Cost')
        idx_row = int(np.floor(idx/ncol))
        idx_col = idx % ncol
    
        if i >= n_components:
            ax[idx_row, idx_col].set_axis_off()
        else:
            ax[idx_row, idx_col].scatter(component_arr[:,i], total_cost, s=10.)
            ax[idx_row, idx_col].set_xlabel(component_labels[i])
            ax[idx_row, idx_col].set_ylabel('Total Cost')

    filename = os.path.join(savedir, f"components_v_total_cost_{name}.pdf")
    save_multi_image(filename)
    plt.close('all')
    
def plot_surrogate_v_actual_cost(savedir, name, surr_cost, actual_cost, component_labels, subsets):
    plt.rcParams["figure.figsize"] = [7.0, 5.0]
    plt.rcParams["figure.autolayout"] = True
    
    nfigs = len(component_labels) 
    if nfigs > 6:
        nrow,ncol = 3,3
    elif nfigs > 4:
        nrow,ncol = 2,3
    else:
        nrow,ncol = 2,2
    for i in range(nfigs + (nrow*ncol) - ((nfigs-1) % (nrow*ncol) + 1)):
        idx = i % (nrow*ncol)
        if idx == 0:
            fig, ax = plt.subplots(nrow, ncol)
            fig.suptitle('Surrogate Skill at Predicting Cost')
            fig.supxlabel('Surrogate Prediction')
            fig.supylabel('Model Output')
        idx_row = int(np.floor(idx/ncol))
        idx_col = idx % ncol
        
        if i >= nfigs:
            ax[idx_row, idx_col].set_axis_off()
        else:
            for s in subsets:
                ax[idx_row, idx_col].scatter(surr_cost[subsets[s],i],
                                             actual_cost[subsets[s],i],
                                             s=10.)
            ax[idx_row, idx_col].title.set_text(component_labels[i])
            center = np.median(actual_cost[subsets[s],i])
            ax[idx_row, idx_col].axline((center, center),
                                    slope=1., color='C3')
        if idx == nrow*ncol-1:
            plt.figlegend(list(subsets.keys()), loc = 'upper left')

    filename = os.path.join(savedir, f"surrogate_v_actual_cost_{name}.pdf")
    save_multi_image(filename)
    plt.close('all')

def plot_inputs_v_cost(savedir, name, input_arr, cost_arr, input_labels, cost_labels, subsets):
    plt.rcParams["figure.figsize"] = [7.0, 5.0]
    plt.rcParams["figure.autolayout"] = True
   
    n_inputs = len(input_labels)

    if n_inputs > 6:
        nrow, ncol = 3, 3
        ncol = 3
    elif n_inputs > 4:
        nrow, ncol = 2, 3
    else:
        nrow, ncol = 2, 2
    for i in range(np.shape(cost_arr)[1]):
        for j in range(n_inputs + (nrow*ncol) - ((n_inputs-1) % (nrow*ncol) + 1)):
            idx = j % (nrow*ncol)
            if idx == 0:
                fig, ax = plt.subplots(nrow, ncol)
                fig.suptitle(cost_labels[i] + " by Input Parameter")
                fig.supylabel(cost_labels[i])
            idx_row = int(np.floor(idx/ncol))
            idx_col = idx % ncol
            
            if j >= n_inputs:
                ax[idx_row, idx_col].set_axis_off()
            else:
                for subset in subsets:
                    ax[idx_row, idx_col].scatter(input_arr[subsets[subset],j],
                                                 cost_arr[subsets[subset],i],
                                                 s=10.)
                ax[idx_row, idx_col].set_xlabel(input_labels[j])
            
            if idx == nrow*ncol-1:
                plt.figlegend(list(subsets.keys()), loc = 'upper left')
    
    filename = os.path.join(savedir, f"inputs_v_cost_{name}.pdf")
    save_multi_image(filename)
    plt.close('all')

def plot_cost_boxplots(savedir, name, cost_arr, cost_labels, subsets):
    random_scatter = np.random.uniform(0.9, 1.1, size=len(cost_arr))

    plt.rcParams["figure.figsize"] = [7.0, 5.0]
    plt.rcParams["figure.autolayout"] = True

    nfigs = len(cost_labels) 
    if nfigs > 6:
        nrow,ncol = 3,3
    elif nfigs > 4:
        nrow,ncol = 2,3
    else:
        nrow,ncol = 2,2
    for i in range(nfigs + (nrow*ncol) - ((nfigs-1) % (nrow*ncol) + 1)):
        idx = i % (nrow*ncol)
        if idx == 0:
            fig, ax = plt.subplots(nrow, ncol)
            fig.suptitle('Component Cost')
        idx_row = int(np.floor(idx/ncol))
        idx_col = idx % ncol

        if i >= nfigs:
            ax[idx_row, idx_col].set_axis_off()
        else:
            for subset in subsets:
                if subset == 'ens':
                    alpha = 0.2
                else:
                    alpha = 1.0
                ax[idx_row, idx_col].scatter(random_scatter[subsets[subset]],
                                             cost_arr[subsets[subset],i],
                                             s=20., alpha = alpha)
            ax[idx_row, idx_col].boxplot(cost_arr[subsets['ens'],i])
            ax[idx_row, idx_col].set_ylabel(cost_labels[i])
            ax[idx_row, idx_col].set_xticks(ticks = [])

        if idx == nrow*ncol-1:
            plt.figlegend(list(subsets.keys()), loc = 'upper left')

    filename = os.path.join(savedir, f"cost_boxplots_{name}.pdf")
    save_multi_image(filename)
    plt.close('all')

def plot_component_impact(savedir, name, component_arr, component_weights, component_labels):
    plt.subplots(figsize=(12, 6))
    plt.bar(component_labels,
            np.std(component_arr * component_weights, axis=0)
            )
    plt.title('Component Impact on Total Cost',
              fontsize = 18)
    plt.xlabel('Component',
               fontsize = 14)
    plt.ylabel('Standard Deviation of Weighted Cost',
               fontsize = 14)
    plt.xticks(rotation = 90,
               fontsize = 14)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f"component_impact_{name}.png"),
                dpi=100)
    plt.close('all')
    
    
def plot_surrogate_rsquared(savedir, name, surr_cost, actual_cost, component_labels):
    plt.subplots(figsize=(12, 6))
    plt.bar(component_labels,
            [np.corrcoef(actual_cost[:,i], surr_cost[:,i])[0,1]**2 for i in range(len(component_labels))]
            )
    plt.title('Surrogate Skill at Predicting Cost',
              fontsize = 18)
    plt.xlabel('Component',
               fontsize = 14)
    plt.ylabel('R-Squared of Predicting Cost',
               fontsize = 14)
    plt.xticks(rotation = 90,
               fontsize = 14)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f"surrogate_rsquared_{name}.png"),
                dpi=100)
    
