from prettytable import PrettyTable

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import os
import tesuract
import xarray as xr
###################
def save_predictions(filename, savedir, X_s, surrogate, target_type,
        nlat_plev_fields, plev_mask_flatten):
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
    others :
        make sure plev fields are handled correctly since the masking is hard
    """
    
    pred_filename = "pred_" + filename + ".nc"
    full_pred_save_path = os.path.join(savedir, pred_filename)

    Y_pred = surrogate.predict(X_s)
    if target_type == 'scalar' or target_type == 'scalarmean':
        plev_mask_flatten = np.zeros((1, Y_pred.shape[1]))

    ds_nc = nc.Dataset(full_pred_save_path, 'w', format='NETCDF4')
    X = ds_nc.createDimension('X', X_s.shape[0])

    if nlat_plev_fields > 0 and target_type == 'full':
        plev_mask_array = np.stack(plev_mask_flatten)
        nvar = ds_nc.createDimension('nvar', plev_mask_array.shape[0])
        ngrid = ds_nc.createDimension('ngrid', plev_mask_array.shape[1])
        mask = ds_nc.createVariable('mask', 'f4', ('nvar', 'ngrid'))
        mask[:] = plev_mask_array
    elif  nlat_plev_fields > 0 and target_type == 'zonal':
        plev_mask_array = np.stack(plev_mask_flatten)
        nvar = ds_nc.createDimension('nvar', plev_mask_array.shape[0])
        ngrid = ds_nc.createDimension('ngrid', plev_mask_array.shape[2])
        mask = ds_nc.createVariable('mask', 'f4', ('nvar', 'X', 'ngrid'))
        mask[:] = plev_mask_array

    index = ds_nc.createDimension('index', Y_pred.shape[1])
    values = ds_nc.createVariable('values', 'f4', ('X', 'index'))
    values[:] = Y_pred
    ds_nc.close()

def save_pc_info(filename, savedir, X_s, surrogate, target_transform, n_components, Y_joined, Y_obs_joined):
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
    
    pcs_filename = "pcs_" + filename + ".nc"
    full_pcs_save_path = os.path.join(savedir, pcs_filename)
    
    Y_pred = surrogate.predict(X_s)
    pc_scores_surr_vals = target_transform.transform(Y_pred)
    
    ds_nc = nc.Dataset(full_pcs_save_path, 'w', format='NETCDF4')
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
    prop_var[:] = surrogate.target_transform.steps[1][1].explained_variance_ratio_
    ds_nc.close()

def plot_surrogate_results(filename, savedir, X_s, surrogate, season, nlat_lon_fields, Y_raw, n_spat_grid, lat_lon_fields,
        normalized_area_weights, resolution):
    """save plots from surrogate

    save difference plot for each of the fields, and r-squareds 
    """
    import clif.visualization as cviz
    random_index = np.random.randint(0, X_s.shape[0])
    random_param_set = X_s[random_index,:]

    y_pred_all = surrogate.predict(random_param_set)
    if season == 'ALL':
        nlat_lon_fields_use = int(nlat_lon_fields/4)
        season_vec = np.repeat(['DJF', 'MAM', 'JJA', 'SON'], nlat_lon_fields_use)
    else:
        season_vec = np.repeat(season, nlat_lon_fields)
        nlat_lon_fields_use = nlat_lon_fields
    for i in range(nlat_lon_fields):
        if (i < nlat_lon_fields):
            y_data = Y_raw[i][random_index,:,:]
            lat, lon = Y_raw[0]["lat"], Y_raw[0]["lon"]
            y_pred = y_pred_all[:,(i*n_spat_grid):((i+1)*n_spat_grid)]
            y_pred_xr = xr.DataArray(np.reshape(y_pred, ( int(resolution.split('x')[0]),int(resolution.split('x')[1]))), coords={"lat": lat, "lon": lon})
            ll_plot_diff = cviz.contour.plot_lat_lon(
                cmap_name="e3sm_default_diff",
                title=lat_lon_fields[i % nlat_lon_fields_use] + ' ' +  season_vec[i] + " Simulation vs. Surrogate",
                #rhs_title=r"$\Delta$" + unit,
                lhs_title="E3SMv2 - Surrogate",
            )
            field_use = lat_lon_fields[i % nlat_lon_fields_use]
            season_use = season_vec[i]
            ll_plot_diff.show(
                y_pred_xr - y_data,
                save=True,
                file_name=f"{savedir}/plots/{filename}_{field_use}_{season_use}.png",
            )
            plt.close()
    y_pred_all = surrogate.predict(X_s)
    r2s = np.zeros(nlat_lon_fields)
    for i in range(nlat_lon_fields):
        if (i < nlat_lon_fields):
            y_data = Y_raw[i]
            ensemble_mean = np.mean(y_data, axis = 0)
            y_data_sub = y_data - ensemble_mean
            y_pred = y_pred_all[:,(i*n_spat_grid):((i+1)*n_spat_grid)]
            null_MSE = np.mean((normalized_area_weights*(y_data_sub - np.mean(y_data_sub, axis = (1,2)))).values ** 2)
            pred_MSE = np.mean((normalized_area_weights*(y_data_sub  - np.reshape(y_pred, (X_s.shape[0], int(resolution.split('x')[0]), int(resolution.split('x')[1])))+ensemble_mean)).values ** 2)
            r2s[i] = 1 - pred_MSE/null_MSE
    if season == 'ALL':
        lat_lon_fields_vec = lat_lon_fields + lat_lon_fields + lat_lon_fields +lat_lon_fields
    else:
        lat_lon_fields_vec = lat_lon_fields
    lat_lon_fields_vec = np.array(lat_lon_fields_vec)
    info_vec = [lat_lon_fields_vec[i] + '\n' +  season_vec[i] for i in range(nlat_lon_fields)]
    if season == 'ALL':
        plt.scatter(lat_lon_fields_vec[season_vec == 'DJF'], r2s[season_vec == 'DJF'])
        plt.scatter(lat_lon_fields_vec[season_vec == 'MAM'], r2s[season_vec == 'MAM'])
        plt.scatter(lat_lon_fields_vec[season_vec == 'JJA'], r2s[season_vec == 'JJA'])
        plt.scatter(lat_lon_fields_vec[season_vec == 'SON'], r2s[season_vec == 'SON'])
        plt.legend(['DJF', 'MAM', 'JJA', 'SON'])
        plt.xlabel('Variable')
        plt.ylabel('R-squared')
        plt.title('R-squared by variable and season')
        plt.savefig(f"{savedir}/plots/{filename}_r2s.png")

###################
# printing results from optimization
###################
# printing model summary...


def print_opt_results(
    xopt_,
    x_labels,
    opt_type,
    R,
    feature_dim,
    cv_score,
    solver="L-BFGS-B",
    filename=None,
):
    """print optimization results

    in a .npy and .txt file
    """

    title = f"{opt_type} ({R} cores using {solver})"

    # solution
    table_opt = PrettyTable()
    table_opt.title = title
    table_opt.field_names = ["Optimal parameters", "Values"]
    table_opt.add_row([x_labels[0], "{0:4.2f}".format(xopt_[0])])
    table_opt.add_row([x_labels[1], "{0:2.2f}".format(xopt_[1])])
    table_opt.add_row([x_labels[2], "{0:2.3f}".format(xopt_[2])])
    table_opt.add_row([x_labels[3], "{0:5.2f}".format(xopt_[3])])
    table_opt.add_row([x_labels[4], "{0:2.5f}".format(xopt_[4])])
    for key in cv_score:
        table_opt.add_row([key, cv_score[key]])
    table_opt.align = "r"
    print(table_opt)

    if filename is not None:
        data = table_opt.get_string()
        with open(filename, "w") as f:
            f.write(data)

    return table_opt


def plot_results(
    objfun,
    X_train,
    xopt_s,
    x_s_default,
    opt_type,
    filename=None,
):

    mse_training = [objfun(x) for x in X_train]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(mse_training, bins=30, density=True, alpha=0.5, label="training")
    mse_opt = objfun(xopt_s)
    mse_default = objfun(x_s_default)
    ax.axvline(x=mse_opt, color="k", linestyle="--", label=opt_type)
    ax.axvline(x=mse_default, color="r", linestyle="--", label="default")
    ax.set_xlabel(r"$\propto$" + " MSE")
    ax.legend(fancybox=True, framealpha=0.5)
    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()



def compute_validation(filename, savedir, opt_type, season, lat_lon_fields, lat_plev_fields, global_fields, xopt_s, x_s_default, Y_obs_raw, Y_obs_raw_plev, params, target_type, surrogate, plev_mask):
    """compare predicted errors on optimized parameters and control parameters

    """
    if season == 'ALL':
        fields = list(np.tile(lat_lon_fields, 4)) + list(np.tile(lat_plev_fields, 4)) + global_fields
        seasons = list(np.repeat(['DJF', 'MAM', 'JJA', 'SON'], params['nll_fields']/4)) + list(np.repeat(['DJF', 'MAM', 'JJA', 'SON'], params['nlp_fields']/4)) + list(np.repeat('NA', params['ng_fields']))
    else:
        fields = lat_lon_fields + lat_plev_fields + global_fields
        seasons = list(np.repeat(season, params['ntot_fields']))
    if target_type == "full":
        # load the data
        Y_opt = surrogate.predict(xopt_s)  # get surrogate prediction

        # compute default
        Y_default = surrogate.predict(x_s_default)

        # now compare to Y_obs_raw
        default_sur_errors = {}
        xopt_sur_errors = {}
        for i in range(params['ntot_fields']):
            Y_use = params['joinT'].inverse_transform(Y_opt)[i]
            Y_use_default = params['joinT'].inverse_transform(Y_default)[i]
            if i < params['nll_fields']:
                z = params['W'] * np.reshape((Y_obs_raw[i].values - Y_obs_raw[i].values.mean()) ** 2, (-1))

                default_sur_errors[fields[i] + '_' + seasons[i]] = np.sqrt(
                        np.mean((params['W']*(Y_use_default - np.reshape(Y_obs_raw[i].values, (-1))) ** 2) / z)
                )
                xopt_sur_errors[fields[i] + '_' + seasons[i]] = np.sqrt(
                        np.mean((params['W']*(Y_use - np.reshape(Y_obs_raw[i].values, (-1))) ** 2) / z)
                )
            elif i < params['nll_fields'] + params['nlp_fields']:
                W_use = params['W_plev'][i - params['nll_fields']]
                plev_mask_use = plev_mask[i - params['nll_fields']].values.reshape((-1))
                Y_obs_use = np.reshape(Y_obs_raw_plev[i-params['nll_fields']].values, (-1))[plev_mask_use]
                z = W_use * ((Y_obs_use - Y_obs_use.mean()) ** 2)
                default_sur_errors[fields[i] + '_' + seasons[i]] = np.sqrt(
                        np.mean((W_use*((Y_use_default - Y_obs_use)) ** 2) / z)
                )
                xopt_sur_errors[fields[i] + '_' + seasons[i]] = np.sqrt(
                        np.mean((W_use*((Y_use - Y_obs_use)) ** 2) / z)
                )
            elif i == params['nll_fields'] + params['nlp_fields']:
                default_sur_errors[fields[i] + '_' + seasons[i]] = Y_use_default[0][0]
                xopt_sur_errors[fields[i] + '_' + seasons[i]] = Y_use[0][0]

        print("Default errors:\n", default_sur_errors)
        print("Opt soln errors:\n", xopt_sur_errors)
        with(open(f"{savedir}/opt_{filename}_{opt_type}_rmse_pred.txt", 'w') as f):
            f.write(str("Default errors:\n" + str(default_sur_errors)))
            f.write('\n')
            f.write(str("Opt soln errors:\n" + str(xopt_sur_errors)))
    if target_type == "scalar": 
        default_sur_errors = {}
        xopt_sur_errors = {}
        for i in range(params['ntot_fields']):
            default_sur_errors[fields[i] + '_' + seasons[i]] = surrogate.predict(x_s_default)[0][i]
            xopt_sur_errors[fields[i] + '_' + seasons[i]] = surrogate.predict(xopt_s)[0][i]


    return  xopt_sur_errors, default_sur_errors, seasons, fields

def plot_validation(xopt_sur_errors, default_sur_errors, seasons, fields, params, season, target_type, opt_type,
        field_str_list, savedir):
    """plot computed validation results

    """
    default_errors = default_sur_errors
    xopt_errors = xopt_sur_errors
    # plot paired bar chart
    labels = np.array([x1 + '_' +   x2 for x1,x2 in zip(fields,seasons)])
    default_means = [np.around(default_errors[fields[i] + '_' + seasons[i]], 3) for i in range(len(fields))]
    at_means = [np.around(xopt_errors[fields[i] + '_' + seasons[i]], 3) for i in range(len(fields))]

    at_means = [np.around(at_means[i]/default_means[i],3) for i in range(len(fields))]
    default_means = [1.0 for i in range(len(fields))]
    if params['ng_fields'] > 0:
        index = len(at_means)-1
        labels = labels[0:index]
        at_means = at_means[0:index]
        seasons = seasons[0:index]
        fields = fields[0:index]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects2 = ax.bar(x + width / 2, at_means, width, label="autotuned")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("normalized RMSE")
    ax.set_title(f"{season}, {target_type} field, {opt_type} optimization")
    ax.set_xticks(x, labels, rotation = 45)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fancybox=True,loc = 'lower left')

    ax.bar_label(rects2, padding=2)

    plot_savename = f"plots/validation_plot_{field_str_list}_{target_type}_{opt_type}_{season}.png"
    fig.savefig(os.path.join(savedir, plot_savename))
    if season == 'ALL':
        x = np.arange(len(labels)/4)
        fig, ax = plt.subplots()
        width = 0.15
        rects1 = ax.bar(x - 3*width/2, np.array(at_means)[np.array(seasons) == 'DJF'], width, label="DJF")
        rects2 = ax.bar(x - width/2, np.array(at_means)[np.array(seasons) == 'MAM'], width, label="MAM")
        rects3 = ax.bar(x + width/2, np.array(at_means)[np.array(seasons) == 'JJA'], width, label="JJA")
        rects4 = ax.bar(x + 3*width/2, np.array(at_means)[np.array(seasons) == 'SON'], width, label="SON")

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("normalized RMSE")
        ax.set_title(f"{season}, {target_type} field, {opt_type} optimization")
        ax.set_xticks(x, np.array(fields)[np.array(seasons) == 'DJF'], rotation = 45)
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fancybox=True,loc = 'lower left')

        ax.bar_label(rects1, padding=2)
        ax.bar_label(rects2, padding=2)
        ax.bar_label(rects3, padding=2)
        ax.bar_label(rects4, padding=2)

        plot_savename = f"plots/validation_plot_seasons_{field_str_list}_{target_type}_{opt_type}_{season}.png"
        fig.savefig(os.path.join(savedir, plot_savename))

    return None

