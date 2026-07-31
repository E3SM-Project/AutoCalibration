import os
import pdb
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from cartopy import crs
from cartopy.util import add_cyclic_point
import cmocean
import copy


def load_merged_nc(data_file_path):
    data_file_path = os.path.expanduser(data_file_path)
    all_cases = xr.load_dataset(data_file_path)
    return all_cases

def fix_lon(lon):
    return np.where(lon > 180, lon - 360, lon)

def ds_to_array(ds, varnames=None):
    """
    Convert an xarray Dataset with variables shaped by ens_idx into a 2-D array:

        n_variables x n_ensemble_members

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to convert.

    varnames : list of str or None
        Variable ordering. If None, use ds.data_vars order.

    Returns
    -------
    arr : np.ndarray
        Array with shape n_variables x n_ensemble_members.
    """

    if varnames is None:
        varnames = list(ds.data_vars)

    arr = []

    for var in varnames:
        if var not in ds:
            arr.append(np.full(ds.sizes["ens_idx"], np.nan))
            continue

        da = ds[var]

        if "ens_idx" not in da.dims:
            continue

        da = da.transpose("ens_idx")
        arr.append(da.values)

    return np.asarray(arr)


def spatial_weighted_mean_da(da, area):
    """
    Spatial mean for one DataArray.

    Handles variables with dimensions like:

        lat, lon
        lev, lat
        lat
        lev
        no spatial dimensions

    Uses area weighting for lat/lon fields, latitude weighting for lat/lev fields,
    and uniform weighting otherwise.
    """

    spatial_dims = [d for d in ("lat", "lon", "lev") if d in da.dims]

    if len(spatial_dims) == 0:
        return da

    if "lat" in spatial_dims and "lon" in spatial_dims:
        weights = area

    elif "lat" in spatial_dims:
        weights = area.sum(dim="lon")

    else:
        weights = xr.ones_like(da[spatial_dims[0]])

    if "lev" in spatial_dims and "lev" not in weights.dims:
        weights = weights * xr.ones_like(da["lev"])

    valid = np.isfinite(da)

    numerator = (da.where(valid) * weights.where(valid)).sum(dim=spatial_dims)
    denominator = weights.where(valid).sum(dim=spatial_dims)

    return numerator / denominator


def spatial_weighted_mean_ds(ds, area):
    """
    Apply spatial_weighted_mean_da to every numeric variable in a Dataset.
    """

    out = xr.Dataset()

    for var in ds.data_vars:
        if np.issubdtype(ds[var].dtype, np.number):
            out[var] = spatial_weighted_mean_da(ds[var], area)

    return out


# Produces box plots:
#   1. Standardized global mean bias
#   2. Standardized RMSE
#   3. Spatial correlation
def plot_box_figs(all_cases):
    vars_exclude = [
        "params",
        "area",
        "RESTOM",
        "SWCRE_ano_grd_adj",
        "LWCRE_ano_grd_adj",
        "dnet_cld_dir",
    ]

    cases_for_stats = all_cases.drop_vars(vars_exclude, errors="ignore")

    ctrl_data = (
        cases_for_stats
        .where(cases_for_stats.workdir.str.contains("ctrl"), drop=True)
        .sel(time="ANN")
        .isel(ens_idx=0, drop=True)
        .sel(product="mod", drop=True)
    )

    obs = (
        cases_for_stats
        .sel(product="obs", drop=True)
        .sel(time="ANN", drop=True)
        .isel(ens_idx=0, drop=True)
    )

    obs_gm = spatial_weighted_mean_ds(obs, all_cases.area)

    all_mod_ann = (
        cases_for_stats
        .sel(time="ANN", drop=True)
        .sel(product="mod", drop=True)
    )

    all_gm = spatial_weighted_mean_ds(all_mod_ann, all_cases.area)

    all_scaled = (all_gm - obs_gm) / obs_gm

    arr_all = ds_to_array(all_scaled)
    arr_ens = ds_to_array(
        all_scaled.where(all_scaled.workdir.str.contains("ens"), drop=True)
    )
    arr_hm = ds_to_array(
        all_scaled.where(all_scaled.workdir.str.contains("hm"), drop=True)
    )
    arr_val = ds_to_array(
        all_scaled.where(
            (
                all_scaled.workdir.str.contains("valid")
                | all_scaled.workdir.str.contains("dnet")
            ),
            drop=True,
        )
    )

    i_sort = np.argsort(np.nanmean(arr_ens, axis=-1))
    arr_varnames = np.array(list(all_scaled.data_vars))

    common_figsize = (7, 8)
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=common_figsize)

    ticks = np.arange(0, len(arr_ens))
    boxticks = ticks + 1

    showfliers = True
    showcaps = False
    sym = "."
    offset = 0.22
    wid = 0.15

    # ============================================================
    # Panel 1: standardized global mean bias
    # ============================================================

    ax = axs[0]

    ax.plot(
        [boxticks[0] - 0.5, boxticks[-1] + 0.5],
        [0, 0],
        "--",
        color="grey",
    )

    bp0 = ax.boxplot(
        arr_ens[i_sort].transpose(),
        positions=boxticks - offset,
        widths=wid,
        patch_artist=True,
        showfliers=showfliers,
        showcaps=showcaps,
        sym=sym,
    )

    bp1 = ax.boxplot(
        arr_hm[i_sort].transpose(),
        positions=boxticks,
        widths=wid,
        patch_artist=True,
        showfliers=showfliers,
        showcaps=showcaps,
        sym=sym,
    )

    bp2 = ax.boxplot(
        arr_val[i_sort].transpose(),
        positions=boxticks + offset,
        widths=wid,
        patch_artist=True,
        showfliers=showfliers,
        showcaps=showcaps,
        sym=sym,
    )

    for box in bp0["boxes"]:
        box.set(edgecolor="black", facecolor="none")

    for box in bp1["boxes"]:
        box.set(edgecolor="red", facecolor="none")

    for box in bp2["boxes"]:
        box.set(edgecolor="green", facecolor="none")

    ax.set_xticks(boxticks)
    ax.set_xticklabels([])

    ctrl_scaled = ds_to_array(
        all_scaled.where(all_scaled.workdir.str.contains("ctrl"), drop=True),
        varnames=arr_varnames,
    ).squeeze()

    bpc = ax.scatter(
        boxticks - offset * 1.5,
        ctrl_scaled[i_sort],
        facecolors="none",
        edgecolor="black",
        marker="D",
        s=30,
    )

    ax.set_ylabel("Standardized global mean bias")

    ax.legend(
        [bp0["boxes"][0], bp1["boxes"][0], bp2["boxes"][0], bpc],
        ["LHS", "Adaptive", "Validate", "Ctrl"],
    )

    # ============================================================
    # Panel 2: standardized RMSE
    # ============================================================

    ax = axs[1]

    se = (all_mod_ann - obs) ** 2

    rmse = spatial_weighted_mean_ds(se, all_cases.area) ** 0.5

    rmse_ctrl = (
        rmse
        .where(rmse.workdir.str.contains("ctrl"), drop=True)
        .isel(ens_idx=0, drop=True)
    )

    rmse_scaled_by_ctrl = rmse / rmse_ctrl

    arr_rmse_scaled_all = ds_to_array(
        rmse_scaled_by_ctrl,
        varnames=arr_varnames,
    )

    arr_ens = ds_to_array(
        rmse_scaled_by_ctrl.where(
            rmse_scaled_by_ctrl.workdir.str.contains("ens"),
            drop=True,
        ),
        varnames=arr_varnames,
    )

    arr_hm = ds_to_array(
        rmse_scaled_by_ctrl.where(
            rmse_scaled_by_ctrl.workdir.str.contains("hm"),
            drop=True,
        ),
        varnames=arr_varnames,
    )

    arr_val = ds_to_array(
        rmse_scaled_by_ctrl.where(
            (
                rmse_scaled_by_ctrl.workdir.str.contains("valid")
                | rmse_scaled_by_ctrl.workdir.str.contains("dnet")
            ),
            drop=True,
        ),
        varnames=arr_varnames,
    )

    ax.plot(
        [boxticks[0] - 0.5, boxticks[-1] + 0.5],
        [1, 1],
        "--",
        color="grey",
    )

    bp0 = ax.boxplot(
        arr_ens[i_sort].transpose(),
        positions=boxticks - offset,
        widths=wid,
        patch_artist=True,
        showfliers=showfliers,
        showcaps=showcaps,
        sym=sym,
    )

    bp1 = ax.boxplot(
        arr_hm[i_sort].transpose(),
        positions=boxticks,
        widths=wid,
        patch_artist=True,
        showfliers=showfliers,
        showcaps=showcaps,
        sym=sym,
    )

    bp2 = ax.boxplot(
        arr_val[i_sort].transpose(),
        positions=boxticks + offset,
        widths=wid,
        patch_artist=True,
        showfliers=showfliers,
        showcaps=showcaps,
        sym=sym,
    )

    for box in bp0["boxes"]:
        box.set(edgecolor="black", facecolor="none")

    for box in bp1["boxes"]:
        box.set(edgecolor="red", facecolor="none")

    for box in bp2["boxes"]:
        box.set(edgecolor="green", facecolor="none")

    ax.set_xticks(boxticks)
    ax.set_xticklabels([])
    ax.set_ylabel("Standardized RMSE")

    # ============================================================
    # Panel 3: spatial correlation
    # ============================================================

    ax = axs[2]

    all_gridpoint = all_mod_ann

    corr_ds = xr.Dataset(
        coords={
            "ens_idx": all_gridpoint.ens_idx,
            "workdir": ("ens_idx", all_gridpoint.workdir.values),
        }
    )

    print("computing spatial correlation")

    for v in arr_varnames:
        if v not in all_gridpoint:
            continue

        if "lat" not in all_gridpoint[v].dims:
            continue

        if "ens_idx" not in all_gridpoint[v].dims:
            continue

        case_corrs = []

        for case in all_gridpoint.ens_idx:
            moddat = np.asarray(
                all_gridpoint[v].sel(ens_idx=case).data
            ).ravel()

            obsdat = np.asarray(obs[v].data).ravel()

            mask = np.isfinite(moddat) & np.isfinite(obsdat)

            if np.sum(mask) < 2:
                cc = np.nan
            else:
                cc = np.corrcoef(moddat[mask], obsdat[mask])[0, -1]

            case_corrs.append(cc)

        corr_ds[v] = xr.DataArray(
            case_corrs,
            dims=["ens_idx"],
            coords={"ens_idx": all_gridpoint.ens_idx},
        )

    arr_all_corr = ds_to_array(corr_ds, varnames=arr_varnames)

    arr_ens = ds_to_array(
        corr_ds.where(corr_ds.workdir.str.contains("ens"), drop=True),
        varnames=arr_varnames,
    )

    arr_hm = ds_to_array(
        corr_ds.where(corr_ds.workdir.str.contains("hm"), drop=True),
        varnames=arr_varnames,
    )

    arr_val = ds_to_array(
        corr_ds.where(
            (
                corr_ds.workdir.str.contains("valid")
                | corr_ds.workdir.str.contains("dnet")
            ),
            drop=True,
        ),
        varnames=arr_varnames,
    )

    ax.plot(
        [boxticks[0] - 0.5, boxticks[-1] + 0.5],
        [1, 1],
        "--",
        color="grey",
    )

    bp0 = ax.boxplot(
        arr_ens[i_sort].transpose(),
        positions=boxticks - offset,
        widths=wid,
        patch_artist=True,
        showfliers=showfliers,
        showcaps=showcaps,
        sym=sym,
    )

    bp1 = ax.boxplot(
        arr_hm[i_sort].transpose(),
        positions=boxticks,
        widths=wid,
        patch_artist=True,
        showfliers=showfliers,
        showcaps=showcaps,
        sym=sym,
    )

    bp2 = ax.boxplot(
        arr_val[i_sort].transpose(),
        positions=boxticks + offset,
        widths=wid,
        patch_artist=True,
        showfliers=showfliers,
        showcaps=showcaps,
        sym=sym,
    )

    for box in bp0["boxes"]:
        box.set(edgecolor="black", facecolor="none")

    for box in bp1["boxes"]:
        box.set(edgecolor="red", facecolor="none")

    for box in bp2["boxes"]:
        box.set(edgecolor="green", facecolor="none")

    ax.set_xticks(boxticks)
    ax.set_xticklabels(arr_varnames[i_sort], rotation=45, fontsize=8)

    corr_ctrl = ds_to_array(
        corr_ds.where(corr_ds.workdir.str.contains("ctrl"), drop=True),
        varnames=arr_varnames,
    ).squeeze()

    bpc = ax.scatter(
        boxticks - offset * 1.5,
        corr_ctrl[i_sort],
        facecolors="none",
        edgecolor="black",
        marker="D",
        s=30,
    )

    ax.set_ylabel("Spatial Correlation")

    fig.tight_layout()

    os.makedirs("pdf", exist_ok=True)
    plt.savefig("pdf/gm_bias_rmse_corr_boxplot.pdf")

    return fig, axs

# contour plot 
def plot_contour(
    all_cases,
    case_data_list,
    case_names_list,
    v,
    subtract_control=True,
    time="ANN",
):
    """
    Map plot comparing E3SM model output, surrogate output, and surrogate - model
    for selected cases.

    Columns:
        1. E3SMv3 model
        2. Surrogate
        3. Surrogate - E3SMv3

    This version gets obs and ctrl directly from all_cases.
    """

    os.makedirs("png", exist_ok=True)

    # ------------------------------------------------------------
    # Select obs and control internally from all_cases
    # ------------------------------------------------------------

    ctrl_ds = all_cases.where(
        all_cases.ens_idx.str.contains("ctrl"),
        drop=True,
    )

    if ctrl_ds.sizes.get("ens_idx", 0) == 0:
        raise ValueError("No control case found using ens_idx contains 'ctrl'.")

    ctrl_ds = ctrl_ds.isel(ens_idx=0, drop=True)

    obs_field = (
        all_cases[v]
        .sel(product="obs", time=time)
        .isel(ens_idx=0, drop=True)
    )

    ctrl_mod = ctrl_ds[v].sel(product="mod", time=time)
    ctrl_sur = ctrl_ds[v].sel(product="sur", time=time)

    lat = all_cases["lat"]
    lon = all_cases["lon"]

    # ------------------------------------------------------------
    # Helper for extracting mod/sur fields
    # ------------------------------------------------------------

    def get_case_field(case, product):
        """
        Return one 2-D lat/lon field for this case and product.
        """

        if isinstance(case, xr.DataArray):
            da = case.sel(product=product, time=time)

        elif isinstance(case, xr.Dataset):
            da = case[v].sel(product=product, time=time)

            if "ens_idx" in da.dims:
                da = da.isel(ens_idx=0, drop=True)

        else:
            # Assume case is an ens_idx label.
            da = all_cases[v].sel(
                ens_idx=case,
                product=product,
                time=time,
            )

        if "ens_idx" in da.dims:
            da = da.isel(ens_idx=0, drop=True)

        if "lev" in da.dims:
            raise ValueError(
                f"{v!r} has a lev dimension after selection. "
                "This contour function expects a 2-D lat/lon field."
            )

        return da

    # ------------------------------------------------------------
    # Colormap/unit settings
    # ------------------------------------------------------------

    var_info = {
        "SWCF": {
            "cmap": "cmo.ice",
            "units": r"W m$^{-2}$",
        },
        "LWCF": {
            "cmap": "hot_r",
            "units": r"W m$^{-2}$",
        },
        "PRECT": {
            "cmap": "BrBG",
            "units": r"mm day$^{-1}$",
        },
        "SWCRE_ano_grd_adj": {
            "cmap": "cmo.balance",
            "units": r"W m$^{-2}$",
        },
        "LWCRE_ano_grd_adj": {
            "cmap": "cmo.balance",
            "units": r"W m$^{-2}$",
        },
    }

    info = var_info.get(
        v,
        {
            "cmap": "magma",
            "units": "",
        },
    )

    cmap = info["cmap"]

    if subtract_control:
        cmap = "cmo.balance"

        if v == "PRECT":
            cmap = "BrBG"

    # ------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------

    ncolumns_plot = 3
    proj = crs.Robinson()

    fig, axs = plt.subplots(
        nrows=len(case_data_list),
        ncols=ncolumns_plot,
        subplot_kw={"projection": proj},
        figsize=(9.6, 3.8),
    )

    axs = np.atleast_2d(axs)

    # ------------------------------------------------------------
    # Establish common vmin/vmax across all three columns
    # ------------------------------------------------------------

    vals_for_limits = []

    for case in case_data_list:
        sur = get_case_field(case, "sur")
        mod = get_case_field(case, "mod")

        if subtract_control:
            vals_for_limits.append((mod - ctrl_mod).values)
            vals_for_limits.append((sur - ctrl_sur).values)
            vals_for_limits.append((sur - mod).values)
        else:
            vals_for_limits.append(mod.values)
            vals_for_limits.append(sur.values)
            vals_for_limits.append((sur - mod).values)

    vmin = np.nanmin([np.nanmin(x) for x in vals_for_limits])
    vmax = np.nanmax([np.nanmax(x) for x in vals_for_limits])

    if subtract_control:
        vmax_abs = np.nanmax(np.abs([vmin, vmax]))
        vmin = -vmax_abs
        vmax = vmax_abs

    # ------------------------------------------------------------
    # Add cyclic points for control fields
    # ------------------------------------------------------------

    ctrl_sur_cyclic, cyclic_lons = add_cyclic_point(
        ctrl_sur.values,
        coord=lon.values,
    )

    ctrl_mod_cyclic, cyclic_lons = add_cyclic_point(
        ctrl_mod.values,
        coord=lon.values,
    )

    # Available if needed later.
    obs_cyclic, cyclic_lons = add_cyclic_point(
        obs_field.values,
        coord=lon.values,
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    for row, case in enumerate(case_data_list):
        sur = get_case_field(case, "sur")
        mod = get_case_field(case, "mod")

        sur_cyclic, cyclic_lons = add_cyclic_point(
            sur.values,
            coord=lon.values,
        )

        mod_cyclic, cyclic_lons = add_cyclic_point(
            mod.values,
            coord=lon.values,
        )

        diff_cyclic = sur_cyclic - mod_cyclic

        if subtract_control:
            mod_plot = mod_cyclic - ctrl_mod_cyclic
            sur_plot = sur_cyclic - ctrl_sur_cyclic
            diff_plot = diff_cyclic

            title_mod = f"E3SMv3 {case_names_list[row]} - Control"
            title_sur = f"Surrogate {case_names_list[row]} - Control"
            title_diff = f"Surrogate - E3SMv3 {case_names_list[row]}"

        else:
            mod_plot = mod_cyclic
            sur_plot = sur_cyclic
            diff_plot = diff_cyclic

            title_mod = f"E3SMv3 {case_names_list[row]}"
            title_sur = f"Surrogate {case_names_list[row]}"
            title_diff = f"Surrogate - E3SMv3 {case_names_list[row]}"

        pl = axs[row, 0].pcolor(
            cyclic_lons,
            lat.values,
            mod_plot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=crs.PlateCarree(),
            shading="auto",
        )

        pl = axs[row, 1].pcolor(
            cyclic_lons,
            lat.values,
            sur_plot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=crs.PlateCarree(),
            shading="auto",
        )

        pl = axs[row, 2].pcolor(
            cyclic_lons,
            lat.values,
            diff_plot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=crs.PlateCarree(),
            shading="auto",
        )

        axs[row, 0].set_title(title_mod)
        axs[row, 1].set_title(title_sur)
        axs[row, 2].set_title(title_diff)

    for a in axs.ravel():
        a.coastlines()

    units = info["units"]
    fig.suptitle(f"{v} {units}", fontsize=12)

    fig.tight_layout()
    fig.subplots_adjust(right=0.86)

    cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.7])

    fig.colorbar(
        pl,
        cax=cbar_ax,
    )

    if subtract_control:
        savename = f"png/contour_{v}_minus_ctrl_with_sur_minus_mod.png"
    else:
        savename = f"png/contour_{v}_with_sur_minus_mod.png"

    if "Min" in case_names_list[0]:
        if subtract_control:
            savename = f"png/contour_{v}_minus_ctrl_ens_min_max_with_sur_minus_mod.png"
        else:
            savename = f"png/contour_{v}_ens_min_max_with_sur_minus_mod.png"

    plt.savefig(
        savename,
        dpi=300,
        bbox_inches="tight",
    )

    return fig, axs


# Heatmap plot
def plot_heatmap(
    all_cases,
    seasonal_vars=(
        "SWCF",
        "LWCF",
        "PRECT",
        "TREFHT",
        "PSL",
        "Z500",
        "U850",
        "U200",
        "RELHUM",
        "T",
        "U",
    ),
    seasons=("DJF", "MAM", "JJA", "SON"),
    model_product="mod",
    obs_product="obs",
    case_patterns=("L001", "L002", "L003", "H001", "H002", "H003"),
    case_titles=None,
    output_file="png/every_heatmap.png",
    vmin=-45,
    vmax=45,
    dpi=300,
):
    """
    Compute seasonal area-weighted RMSE costs from an xarray Dataset
    and plot six percent-change heatmaps relative to the control case.
    """

    if case_titles is None:
        case_titles = [
            r"L1 $\lambda = -2.11$",
            r"L2 $\lambda = -1.97$",
            r"L3 $\lambda = -2.05$",
            r"H1 $\lambda = -1.41$",
            r"H2 $\lambda = -1.43$",
            r"H3 $\lambda = -1.36$",
        ]

    seasonal_vars = list(seasonal_vars)
    seasons = list(seasons)

    def first_match(mask, label):
        idx = np.flatnonzero(mask.values)

        if len(idx) == 0:
            raise ValueError(f"No ensemble member found for {label}")

        if len(idx) > 1:
            print(f"Warning: found {len(idx)} matches for {label}; using first.")

        return idx[0]

    ctrl_idx = first_match(
        all_cases.ens_idx.str.contains("ctrl"),
        "ctrl",
    )

    case_idx = [
        first_match(all_cases.ens_idx.str.match(pattern), pattern)
        for pattern in case_patterns
    ]

    def seasonal_rmse_cost(ds, ens_sel):
        """
        Return area/latitude-weighted RMSE costs.

        Handles variables with dimensions like:

            time, product, ens_idx, lat, lon
            time, product, ens_idx, lev, lat
            time, product, ens_idx, lat
            time, product, ens_idx
        """

        costs = []
        keep_dims = {"time", "ens_idx"}

        for var in seasonal_vars:
            if var not in ds:
                raise ValueError(f"Variable {var!r} not found in dataset.")

            model = ds[var].isel(ens_idx=ens_sel).sel(
                product=model_product,
                time=seasons,
            )

            obs = ds[var].isel(ens_idx=ens_sel).sel(
                product=obs_product,
                time=seasons,
            )

            err2 = (model - obs) ** 2

            reduce_dims = [
                d for d in err2.dims
                if d not in keep_dims
            ]

            if len(reduce_dims) == 0:
                rmse = np.sqrt(err2)
                costs.append(rmse)
                continue

            if "lat" in reduce_dims and "lon" in reduce_dims:
                weights = ds["area"]

            elif "lat" in reduce_dims:
                weights = ds["area"].sum(dim="lon")

            else:
                weights = xr.ones_like(err2[reduce_dims[0]])

            if "lev" in reduce_dims and "lev" not in weights.dims:
                weights = weights * xr.ones_like(ds["lev"])

            valid = np.isfinite(err2)

            numerator = (err2.where(valid) * weights.where(valid)).sum(
                dim=reduce_dims
            )

            denominator = weights.where(valid).sum(
                dim=reduce_dims
            )

            rmse = np.sqrt(numerator / denominator)

            costs.append(rmse)

        cost = xr.concat(
            costs,
            dim=xr.IndexVariable("metric", seasonal_vars),
        )

        order = ["metric", "time"] + [
            d for d in cost.dims
            if d not in ["metric", "time"]
        ]

        return cost.transpose(*order)

    ctrl_cost = seasonal_rmse_cost(all_cases, ctrl_idx)
    case_cost = seasonal_rmse_cost(all_cases, case_idx)

    outdir = os.path.dirname(output_file)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(8, 9),
    )

    colors = plt.get_cmap("coolwarm", 9)(np.arange(9))
    colors[4] = [1.0, 1.0, 1.0, 1.0]
    cmap = ListedColormap(colors)

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    for col, a in enumerate(ax.ravel()):
        pct_improv = 100.0 * (
            case_cost.isel(ens_idx=col) - ctrl_cost
        ) / ctrl_cost

        pct_vals = np.squeeze(pct_improv.values)

        if pct_vals.ndim != 2:
            raise ValueError(
                f"Expected 2-D heatmap data, got shape {pct_vals.shape}. "
                f"pct_improv dims are {pct_improv.dims}."
            )

        im = a.imshow(
            pct_vals,
            cmap=cmap,
            norm=norm,
            alpha=0.7,
        )

        for i in range(len(seasonal_vars)):
            for j in range(len(seasons)):
                value = pct_vals[i, j]

                if np.isfinite(value):
                    label = str(int(value))
                else:
                    label = "nan"

                a.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    color="k",
                )

        a.set_xticks(np.arange(len(seasons)))
        a.set_yticks(np.arange(len(seasonal_vars)))

        a.set_xticklabels(seasons)
        a.set_yticklabels(seasonal_vars)

        title_color = "blue" if col < 3 else "red"
        a.set_title(case_titles[col], color=title_color)

    fig.subplots_adjust(
        left=0.12,
        right=0.86,
        bottom=0.08,
        top=0.94,
        wspace=0.35,
        hspace=0.35,
    )

    cbar_ax = fig.add_axes([0.89, 0.18, 0.025, 0.64])

    fig.colorbar(
        im,
        cax=cbar_ax,
        label="RMSE % change",
    )

    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")

    return fig, ax


def scat(swfd_mean, lwfd_mean, all_cases, valid_lo, valid_hi):
    fig, axs = plt.subplots(nrows= 1 , ncols=3,figsize=(9,3.25))
    ax=axs[0]
    ax.axis('equal')
    ax.scatter( all_cases.dnet_cld_dir.sel(product='mod').sel(time='ANN'), swfd_mean + lwfd_mean, alpha=0.5, facecolors='grey', edgecolors='grey')
    for case in xr.merge( [valid_lo, valid_hi],compat='override').ens_idx:
        dnet = all_cases.where(all_cases.ens_idx.str.match( str(case.values )), drop=True)
        cf   = (swfd_mean + lwfd_mean).where(all_cases.ens_idx.str.match( str(case.values )), drop=True)
        color='blue'; bbox={'visible':False}
        if 'H' in str(case.values ):color='red'
        if 'L001' in str(case.values ) or 'H003' in str(case.values ):
            bbox = {'facecolor': 'white', 'alpha': 0.6, 'pad': 2}
        ax.text(dnet.dnet_cld_dir.sel(product='mod').sel(time='ANN'), cf, s = str(case.values).replace("0", "").replace("L", "").replace("H", ""),
                color=color, bbox=bbox, horizontalalignment='center', verticalalignment='center', fontweight='bold')
        
        
        
    ctrl = all_cases.where(all_cases.ens_idx.str.match('ctrl'), drop=True)
    ax.scatter(ctrl.dnet_cld_dir.sel(product='mod').sel(time='ANN'), (swfd_mean + lwfd_mean).where(all_cases.ens_idx.str.match('ctrl'), drop=True), color='k', marker='D', s=100)
    ax.set_xlabel('$\\lambda$ Wm$^{-2}$K$^{-1}$' )
    ax.set_ylabel('$\\lambda_{cld}$ Wm$^{-2}$K$^{-1}$')
    
    ax=axs[1]
    ax.axis('equal')
    ax.scatter( all_cases.dnet_cld_dir.sel(product='mod').sel(time='ANN'), swfd_mean, label='Shortwave', alpha=0.5, facecolors='grey', edgecolor='grey' )
    ax.scatter( all_cases.dnet_cld_dir.sel(product='mod').sel(time='ANN'), lwfd_mean, label='Longwave', marker='.', color='k')
    ax.scatter( ctrl.dnet_cld_dir.sel(product='mod').sel(time='ANN'), swfd_mean.where(all_cases.ens_idx.str.match('ctrl'), drop=True),facecolors='grey',edgecolor='white', marker='D' , s=100)
    ax.scatter( ctrl.dnet_cld_dir.sel(product='mod').sel(time='ANN'), lwfd_mean.where(all_cases.ens_idx.str.match('ctrl'), drop=True),facecolors='black',edgecolor='white' , marker='D', s=100)
    
    ax.set_xlabel('$\\lambda$ Wm$^{-2}$K$^{-1}$' )
    ax.set_ylabel('$\\lambda_{cld}$ Wm$^{-2}$K$^{-1}$')
    ax.legend()

    ax=axs[2]
    ax.axis('equal')
    ## Draw diagonal lines
    for pos in np.linspace(-2, 2, 11):
        plt.axline((pos, 0), slope=-1, color='grey', alpha=0.3)
    ax.scatter(swfd_mean, lwfd_mean, c=all_cases.dnet_cld_dir.sel(product='mod').sel(time='ANN'), alpha=0.5, label='$\\lambda_{Net}$')
    for case in xr.merge( [valid_lo, valid_hi], compat='override').ens_idx:
        swfd   = (swfd_mean).where(all_cases.ens_idx.str.match( str(case.values )), drop=True)
        lwfd   = (lwfd_mean).where(all_cases.ens_idx.str.match( str(case.values )), drop=True)
        color='blue'; bbox={'visible':False}
        if 'H' in str(case.values ):color='red';
        if 'L001' in str(case.values ) or 'H003' in str(case.values ):
            bbox = {'facecolor': 'white', 'alpha': 0.6, 'pad': 2}
        ax.text(swfd, lwfd, s = str(case.values).replace("0", "").replace("L", "").replace("H", ""),
                color=color, bbox=bbox, horizontalalignment='center', verticalalignment='center',fontweight='bold')
    ax.scatter(swfd_mean.where(swfd_mean.ens_idx.str.match( 'ctrl' )), lwfd_mean.where(lwfd_mean.ens_idx.str.match( 'ctrl' )), color='k', marker='D', s=100)
    ax.set(xlim=(-0.75, 0.75),ylim=(-0.75, 0.75))
    ax.set_xlabel('$\\lambda_{SWcld}$  Wm$^{-2}$K$^{-1}$' )
    ax.set_ylabel( '$\\lambda_{LWcld}$  Wm$^{-2}$K$^{-1}$' )
    fig.tight_layout()
    os.makedirs('pdf', exist_ok=True)
    plt.savefig('pdf/scat_feedbacks.pdf')


def scat_swcf_vs_swfeed(all_cases, valid_lo, valid_hi):
    plt.figure(figsize=(4.75,3.25))
    #plt.scatter( all_cases['SWCF'].sel(time='ANN').sel(product='mod').weighted(all_cases['area'].sel(time='ANN').sel(product='obs')).mean(['lat','lon']), all_cases['SWCRE_ano_grd_adj'].sel(time='ANN').sel(product='mod').weighted(all_cases['area'].sel(time='ANN').sel(product='obs')).mean(['lat','lon']) , color='grey',alpha=0.4  )
    restom = all_cases['RESTOM'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon'])
    #cmap  = 'cmo.balance'
    #cmap = plt.get_cmap('coolwarm',3)
    cmap = plt.get_cmap('coolwarm',8)

    plt.scatter( all_cases['SWCF'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon']), all_cases['SWCRE_ano_grd_adj'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon']) ,alpha=0.6 , c = restom, edgecolor='grey', cmap=cmap, vmin=-12, vmax=12)
    cbar = plt.colorbar(extend='max')
    #cbar.set_ticks([-12, -8,-4, 0, 4, 8, 12])
    cbar.set_ticks([-12, -9, -6, -3, 0, 3, 6, 9, 12])
    cbar.ax.set_title('RESTOM  Wm$^{-2}$')     
    
    for case in xr.merge( [valid_lo, valid_hi], compat='override').ens_idx:
        sw   = all_cases['SWCF'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon']).where(all_cases.ens_idx.str.match( str(case.values )), drop=True)
        swfd = all_cases['SWCRE_ano_grd_adj'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon']).where(all_cases.ens_idx.str.match( str(case.values )), drop=True)
        color='blue'; bbox={'visible':False}
        if 'H' in str(case.values ):color='red'
        if 'L001' in str(case.values ) or 'H003' in str(case.values ):
            bbox = {'facecolor': 'white', 'alpha': 0.6, 'pad': 2}
        plt.text(sw, swfd, s = str(case.values).replace("0", "").replace("L", "").replace("H", ""),
                 color=color, bbox=bbox, horizontalalignment='center', verticalalignment='center', fontsize=15,fontweight='bold')
    sw   = all_cases['SWCF'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon']).where(all_cases.ens_idx.str.match( 'ctrl'), drop=True)
    swfd = all_cases['SWCRE_ano_grd_adj'].sel(time='ANN').sel(product='mod').weighted(all_cases['area']).mean(['lat','lon']).where(all_cases.ens_idx.str.match( 'ctrl' ), drop=True)
    plt.scatter( sw, swfd,facecolors='black',edgecolor='white' , marker='D', s=100)
    lims=plt.gca().get_ylim()
    obs_swcf = all_cases['SWCF'].sel(time='ANN').sel(product='obs').weighted(all_cases['area']).mean(['lat','lon']).isel(ens_idx=0)
    plt.plot([obs_swcf, obs_swcf], [lims[0],lims[1]],'k--', label='CERES')
    plt.ylim(lims)
    plt.legend(loc='lower right')
    plt.ylabel( '$\\lambda_{SWcld}$  Wm$^{-2}$K$^{-1}$' )
    plt.xlabel( 'SWCF Wm$^{-2}$' )
    plt.tight_layout()
    os.makedirs('pdf', exist_ok=True)
    plt.savefig('pdf/scat_swcf_vs_swfeed.pdf')


if __name__ == "__main__":
    data_file_path = "~/Downloads/H003_rshp_w_obs_20260126.nc"

    all_cases = load_merged_nc(data_file_path)
    valid_hi = all_cases.where( (all_cases.ens_idx.str.match('H001') |
                             all_cases.ens_idx.str.match('H002') |
                             all_cases.ens_idx.str.match('H003')), drop=True)
    valid_lo = all_cases.where( (all_cases.ens_idx.str.match('L001') |
                             all_cases.ens_idx.str.match('L002') |
                             all_cases.ens_idx.str.match('L003')), drop=True)


    # ============================================================
    # Generate paper figures
    # ============================================================

    # ------------------------------------------------------------
    # Figure 1: Box plots
    #   - Standardized global mean bias
    #   - Standardized RMSE
    #   - Spatial correlation
    # ------------------------------------------------------------

    plot_box_figs(all_cases)

 
    # ------------------------------------------------------------
    # Figures 2, A2, A3: Contour maps
    #
    # For each selected variable, find the LHS ensemble members with
    # the minimum and maximum annual global-mean model value. Then plot:
    #   column 1: E3SMv3 minus control
    #   column 2: surrogate minus control
    #   column 3: surrogate minus E3SMv3
    # ------------------------------------------------------------
    ens = all_cases.where( all_cases.ens_idx.str.contains('ens'), drop=True) 
    extreme_d = {'SWCF':{},'LWCF':{},'PRECT':{}, 'SWCRE_ano_grd_adj':{},'LWCRE_ano_grd_adj':{}}
    for v in extreme_d:
        max_i = (ens[v].sel(time='ANN').sel(product='mod').weighted(ens['area'])).mean(('lat','lon')).idxmax(dim='ens_idx')
        min_i = (ens[v].sel(time='ANN').sel(product='mod').weighted(ens['area'])).mean(('lat','lon')).idxmin(dim='ens_idx')
        extreme_d[v]['min_data'] = ens[v].sel(ens_idx=min_i.values)
        extreme_d[v]['max_data'] = ens[v].sel(ens_idx=max_i.values)

    case_names = ["Ens. Min", "Ens. Max"]

    for sn in ["ANN"]:
        for v in ["LWCF", "SWCF", "PRECT"]:
            plot_contour(
                all_cases,
                [extreme_d[v]["min_data"], extreme_d[v]["max_data"]],
                case_names,
                v,
                subtract_control=True,
                time=sn,
            )

          
    # ------------------------------------------------------------
    # Figure 5: Heatmap
    #
    # Seasonal RMSE percent change relative to control for the selected
    # low- and high-ECS validation cases.
    # ------------------------------------------------------------
    fig, ax = plot_heatmap(all_cases) # Done. 

    # ------------------------------------------------------------
    # Figure 7: Scatterplot of cloud feedbacks
    #
    # x-axis: annual global-mean shortwave cloud feedback
    # y-axis: annual global-mean longwave cloud feedback
    # ------------------------------------------------------------
    swfd_mean = (all_cases['SWCRE_ano_grd_adj'].sel(time='ANN').sel(product='mod').weighted(all_cases['area'])).mean(('lat','lon'))
    lwfd_mean = (all_cases['LWCRE_ano_grd_adj'].sel(time='ANN').sel(product='mod').weighted(all_cases['area'])).mean(('lat','lon'))
    scat( swfd_mean, lwfd_mean, all_cases, valid_lo, valid_hi)

    
    # ------------------------------------------------------------
    # Figure 8: Scatterplot of shortwave cloud feedback vs SWCF
    # ------------------------------------------------------------
    scat_swcf_vs_swfeed( all_cases, valid_lo, valid_hi )




