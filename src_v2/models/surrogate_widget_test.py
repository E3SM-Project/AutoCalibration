import numpy as np
from sklearn.metrics import make_scorer
import xarray as xr
import yaml
import os, time, sys
import clif.preprocessing as cpp
import joblib

import tesuract
import postprocessing
from tesuract.preprocessing import DomainScaler
from tesuract.preprocessing import PCATargetTransform
import sklearn

from scipy.optimize import minimize
from functools import partial
import multiprocessing as mp
import tk
import matplotlib
matplotlib.rcParams['interactive'] = True
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print(plt.get_backend())
from matplotlib.widgets import Slider, Button
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.artist import Artist

from prettytable import PrettyTable
from tqdm import tqdm

from sklearn.model_selection import KFold
import pdb

# use cartopy to plot
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter


#############################
# load config file and data
#############################
# make config2.yaml a system argument
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

season = cfg["season"]
nyear = cfg["nyear"]
datadir = cfg["datadir"]
savedir = cfg["savedir"]
lat_lon_fields = cfg["fields"]
nlat_lon_fields = len(lat_lon_fields)

lat_plev_fields = cfg["lat_plev_fields"]
nlat_plev_fields = len(lat_plev_fields)
print(nlat_plev_fields)

target_type = cfg["target_type"]
surrogate_params = cfg["surrogate"]["options"]
opt_params = cfg["calibration"]["optimize"]
resolution = cfg["resolution"]
n_components = surrogate_params['n_components']

n_spat_grid = int(resolution.split('x')[0]) * int(resolution.split('x')[1])
n_plev_grid = 24*37
if cfg['subtract_ens_mean']:
    target_centering = 'ensmean'
else:
    target_centering = 'raw'

season_info_spat = []
season_info_plev = []
variable_info_spat = []
variable_info_plev = []
if season == 'ALL': 
    Y_raw = []
    Y_obs_raw = []
    Y_raw_plev = []
    Y_obs_raw_plev = []
    plev_mask = []
    nlat_lon_fields = nlat_lon_fields * 4
    nlat_plev_fields = nlat_plev_fields * 4
    for season_i in ['DJF', 'MAM', 'JJA', 'SON']:
        dataset = xr.open_dataset(os.path.join(datadir, f"lat_lon_{nyear}yr_{resolution}_{season_i}.nc"))
        dataset_ref = xr.open_dataset(os.path.join(datadir, f"lat_lon_{resolution}_{season_i}_obs.nc"))
        for f in lat_lon_fields:
            Y_raw.append(dataset[f])
            Y_obs_raw.append(dataset_ref[f])
            variable_info_spat.append(np.repeat(f, n_spat_grid))
            season_info_spat.append(np.repeat(season_i, n_spat_grid))
        dataset_plev = xr.open_dataset(os.path.join(datadir, f"lat_plev_{nyear}yr_{resolution}_{season_i}.nc"))
        dataset_plev_ref = xr.open_dataset(os.path.join(datadir, f"lat_plev_{resolution}_{season_i}_obs.nc"))
        for f in lat_plev_fields:
            Y_raw_plev.append(dataset_plev[f])
            Y_obs_raw_plev.append(dataset_plev_ref[f])
            plev_mask.append(np.isnan(dataset_plev[f]).sum(axis = 0) == 0)
            variable_info_plev.append(np.repeat(f, n_plev_grid))
            season_info_plev.append(np.repeat(season_i, n_plev_grid))
else: 
    dataset = xr.open_dataset( os.path.join(datadir, f"lat_lon_{nyear}yr_{resolution}_{season}.nc"))
    dataset_ref = xr.open_dataset(os.path.join(datadir, f"lat_lon_{resolution}_{season}_obs.nc"))
    Y_raw = [dataset[f] for f in lat_lon_fields]  # simulation data
    Y_obs_raw = [dataset_ref[f] for f in lat_lon_fields]  # obs data
    variable_info_spat.append(np.repeat(lat_lon_fields, n_spat_grid))
    season_info_spat.append(np.repeat(season, n_spat_grid*nlat_lon_fields))
    dataset_plev = xr.open_dataset( os.path.join(datadir, f"lat_plev_{nyear}yr_{resolution}_{season}.nc"))
    dataset_plev_ref = xr.open_dataset(os.path.join(datadir, f"lat_plev_{resolution}_{season}_obs.nc"))
    Y_raw_plev = [dataset_plev[f] for f in lat_plev_fields]  # simulation data
    Y_obs_raw_plev = [dataset_plev_ref[f] for f in lat_plev_fields]  # obs data
    plev_mask = [np.isnan(dataset_plev[f]).sum(axis=0) == 0 for f in lat_plev_fields]
    variable_info_plev.append(np.repeat(lat_plev_fields, n_plev_grid))
    season_info_plev.append(np.repeat(season, n_plev_grid*nlat_plev_fields))

variable_info = variable_info_spat + variable_info_plev
season_info = season_info_spat + season_info_plev
if nlat_lon_fields == 0:
    field_str_list = "_".join(lat_plev_fields)
elif nlat_plev_fields == 0:
    field_str_list = "_".join(lat_lon_fields)
else:
    field_str_list = "_".join(["_".join(lat_lon_fields), "_".join(lat_plev_fields)]) 

filename = f"{season}_{nyear}yr_{target_type}_{resolution}_{n_components}_{field_str_list}_{target_centering}"
model_filename = "model_" + filename + ".joblib"
full_save_path = os.path.join(savedir, model_filename)

# printing summary of experiment...
table_setup = PrettyTable()
table_setup.field_names = ["Season", "Sim. length", "lat/lon Fields","lat/plev Fields", "Target"]
table_setup.add_row([season, nyear, lat_lon_fields, lat_plev_fields, cfg["target_type"]])
table_setup.title = "E3SM Calibration Experiment"
print(table_setup)

#############################
# Load feature data (input)
#############################
X_xr = dataset["lhs"]
X = X_xr.values
x_labels = X_xr["x"].values
X_bounds = dataset["lhs_bnds"].values

feature_transform = DomainScaler(dim=X.shape[1],input_range=list(X_bounds),output_range=(-1, 1))
X_s = feature_transform.fit_transform(X)
feature_dim = X_s.shape[1]

default_params = np.array([500.0, 2.40, 0.12, 3600, -0.0007])
x_s_default = feature_transform.fit_transform(default_params)

#############################
# Transform target and obs data
#############################
area_weights = dataset.area[0]  # lat lon area weights
normalized_area_weights = area_weights / area_weights.sum()
normalized_area_weights_plev = dataset_plev.area[0]/dataset_plev.area[0].sum()

Y_ens_mean = [Yi.mean(axis = 0) for Yi in Y_raw]
Y_ens_mean_plev = [Yi.mean(axis = 0) for Yi in Y_raw_plev]

if cfg['subtract_ens_mean']:
    Y_raw = [Y_raw[i] - Y_ens_mean[i] for i in range(len(Y_raw))]
    Y_obs_raw = [Y_obs_raw[i] - Y_ens_mean[i] for i in range(len(Y_obs_raw))]
    Y_raw_plev = [Y_raw_plev[i] - Y_ens_mean_plev[i] for i in range(len(Y_raw_plev))]
    Y_obs_raw_plev = [Y_obs_raw_plev[i] - Y_ens_mean_plev[i] for i in range(len(Y_obs_raw_plev))]

# define transforms for full field, zonal and scalar outputs
flatten_transform = cpp.FlattenData(dims=["lat", "lon"])
flatten_transform_plev = cpp.FlattenData(dims=["lat", "plev"])
zonal_mean_transform = cpp.MarginalizeOutTransform(dims=["lon"], lat_lon_weights=area_weights)
mse_transform = cpp.MarginalizeOutTransform(dims=["lat", "lon"], lat_lon_weights=area_weights)

# compute an xarray data array of scalers
Sigmas = [np.sqrt(Yi.var()) + 0 * Yi for Yi in Y_obs_raw]
Sigmas_plev = [np.sqrt(Yi.var()) + 0 * Yi for Yi in Y_obs_raw_plev]

if cfg["target_type"] == "full":
    Y_obs = [flatten_transform.fit_transform(Yi) for Yi in Y_obs_raw]
    Y = [flatten_transform.fit_transform(Yi) for Yi in Y_raw]
    S = [flatten_transform.fit_transform(s) for s in Sigmas]
    W = normalized_area_weights.values.flatten()
    plev_mask = [np.isnan(Yi).sum(axis = 0) < 1 for Yi in Y_raw_plev]

    plev_mask_flatten = [flatten_transform_plev.fit_transform(plevi) for plevi in plev_mask]
    Y_obs_plev = [flatten_transform_plev.fit_transform(Y_obs_raw_plev[ival])[plev_mask_flatten[ival]] for ival in range(len(Y_obs_raw_plev))]
    Y_plev = [flatten_transform_plev.fit_transform(Y_raw_plev[ival])[:,plev_mask_flatten[ival]] for ival in range(len(Y_raw_plev))]
    S_plev = [flatten_transform_plev.fit_transform(Sigmas_plev[ival])[plev_mask_flatten[ival]] for ival in range(len(Sigmas_plev))]
    W_plev = [np.reshape(np.tile(normalized_area_weights_plev.values.flatten(), (37,1)),(-1))[plev_mask_flatten[ival]] for ival in range(len(Y_raw_plev))]

if cfg["target_type"] == "zonal":
    Y_obs = [zonal_mean_transform.fit_transform(Yi) for Yi in Y_obs_raw]
    Y = [zonal_mean_transform.fit_transform(Yi) for Yi in Y_raw]
    S = [zonal_mean_transform.fit_transform(s) for s in Sigmas]
    weights = zonal_mean_transform.fit_transform(area_weights).values
    W = weights / weights.sum()


# define scalar error
def scalar_function(Y, Y_obs):
    # Y and Y_obs should be the same shape
    return np.sqrt(mse_transform.fit_transform((Y - Y_obs) ** 2))


# load custom scalar function
# will overwrite scalar_function above
if cfg["custom_scalar_function"]["file"] is not None:
    print("\nLoading scalar function...")
    cwd = os.path.dirname(__file__)
    full_path = os.path.join(cwd, "custom_scalar_function.py")
    exec(open(full_path).read())
    # overwrite function above with custom
    scalar_function = custom_scalar_function

if cfg["custom_surrogate_fit_function"]["file"] is not None:
    print("\nLoading surrogate fitting function...")
    #cwd = os.path.dirname(__file__)
    #full_path = os.path.join(cwd, "custom_surrogate_fit_function.py")
    exec(open("custom_surrogate_fit_function.py").read())
    # overwrite function above with custom
    surrogate_scorer = make_scorer(custom_surrogate_fit_function)
else:
    surrogate_scorer = "neg_root_mean_squared_error"



if cfg["target_type"] == "scalar":
    # computes the root mean square error per simulation (skill score), not to be confused with RMSE per grid point
    Ytemp = [
        scalar_function(Yi, Y_obs_raw[i]).expand_dims(dim="new", axis=1)
        for i, Yi in enumerate(Y_raw)
    ]
    Y = [Yi for Yi in Ytemp]  # compute Root mean square error
    Y_obs = [xr.DataArray(0)] * nfields  # Y_obs = 0
    S = [s.mean(dim=["lat", "lon"]) ** 1 for s in Sigmas]
    W = np.ones(1)


# convert xarray to numpy arrays
Y_lat_lon = [Yi.values for Yi in Y]
Y_plev_lat = [Yi.values for Yi in Y_plev]
Y = Y_lat_lon + Y_plev_lat
Y_lat_lon_obs = [Yi.values for Yi in Y_obs]
Y_plev_lat_obs = [Yi.values for Yi in Y_obs_plev]
Y_obs = Y_lat_lon_obs + Y_plev_lat_obs

S_lat_lon = [s.values for s in S]
S_plev_lat = [s.values for s in S_plev]
S = S_lat_lon + S_plev_lat
#############################
# Join transform on the data
#############################

import preprocessing

joinT = preprocessing.JoinTransform()
Y_joined = joinT.fit_transform(Y)
Y_obs_joined = joinT.transform(Y_obs)
sigmas_joined = joinT.transform(S)

scalingT = preprocessing.SigmaScaling(sigma=sigmas_joined)

#import preprocessing

#joinT = preprocessing.JoinTransform()
#Y_joined = joinT.fit_transform(Y)
#Y_obs_joined = joinT.transform(Y_obs)
#sigmas_joined = joinT.transform(S)

#scalingT = preprocessing.SigmaScaling(sigma=sigmas_joined)
Y_s = scalingT.fit_transform(Y_joined)
Y_obs_s = scalingT.fit_transform(Y_joined)

#############################
# Create target transform pipeline
#############################
n_components = cfg["surrogate"]["options"]["n_components"]
from sklearn.decomposition import PCA

if cfg["target_type"] == "full" or cfg["target_type"] == "zonal":
    target_transform = sklearn.pipeline.Pipeline([("scale", preprocessing.SigmaScaling(sigma=sigmas_joined)),("pca", PCA(n_components=n_components, whiten=True)),])
elif cfg["target_type"] == "scalar":
    target_transform = sklearn.pipeline.Pipeline(
        [
            ("scale", preprocessing.SigmaScaling(sigma=sigmas_joined)),
            # ("pca", PCA(n_components=3, whiten=True)),
        ]
    )


def compute_cv_score_multiple(
    self, X, y, regressor="pce", target_transform=None, scoring="r2"
):
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
#############################
# Fit surrogate with ROM + PCE
#############################
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
 

if cfg["surrogate"]["fit"]:
    # fit surrogate model
    surrogate = tesuract.MRegressionWrapperCV(
        n_jobs=n_cores_half,
        regressor=["pce"],
        reg_params=[pce_grid],
        scorer=surrogate_scorer,
        target_transform=target_transform,
        target_transform_params={},
        verbose=0,
    )
    start = time.time()
    print("\nFitting surrogate...")
    fit_start = time.time()
    surrogate.fit(X_s, Y_joined)
    print("Total fitting time is {0:.3f} seconds".format(time.time() - fit_start))

    def score_func(y, y_pred):
        return np.linalg.norm(y - y_pred) / np.linalg.norm(y)

    # change auto target transform to manual for cloning
    if n_components == "auto":
        n_pc = len(surrogate.best_estimators_)
        if "pca" in target_transform.named_steps.keys():
            target_transform.set_params(pca__n_components=n_pc)
    # Clone and compute the cv score of full model
    print("\nComputing CV score and final fit...")

    cv_score, surrogate = compute_cv_score_multiple(surrogate,
        X=X_s, y=Y_joined, scoring={"r2", "neg_root_mean_squared_error", "neg_median_absolute_error"}, target_transform=target_transform
    )

    # Fit final, simplified model
    surrogate.fit(X_s, Y_joined)
    # save model
    joblib.dump(surrogate, full_save_path)
    print("Saving model to {0}...\n".format(full_save_path))

else:
    print("Loading model from {0}...".format(full_save_path))
    surrogate = joblib.load(full_save_path)
    cv_score = ""
    if surrogate_params["compute_cv_score"]:
        print("\nComputing CV score...")
        cv_score, _ = compute_cv_score_multiple(self = surrogate,X=X_s, y=Y_joined, scoring={"r2", "neg_root_mean_squared_error", "neg_median_absolute_error"})
    if surrogate_params["compute_raw_cv_scores"]:
        surrogate_clone = surrogate.clone()
        kf = KFold(n_splits=5)
        all_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X_s[train_index], X_s[test_index]
            Y_train, Y_test = Y_joined[train_index], Y_joined[test_index]
            surrogate_clone.fit(X_train, Y_train)
            Y_pred = surrogate_clone.predict(X_test)
            scores = sklearn.metrics.r2_score(Y_test, Y_pred, multioutput="raw_values")
            scores[scores < 0] = 0.0
            all_scores.append(scores)
        raw_cv_scores = np.mean(all_scores, axis=0)

cv_score = dict(sorted(cv_score.items()))
# printing model summary...
table_model = PrettyTable()
table_model.title = "ROM-based ML Model Summary"
table_model.field_names = ["Type", "Value"]
table_model.add_row(["Input size", f"({feature_dim},)"])
table_model.add_row(["# Components", f"({n_components},)"])
table_model.add_row(["Output size", f"({Y_joined.shape[1]},)"])

for key in cv_score:
    table_model.add_row([key, cv_score[key]])
table_model.float_format = ".6"
print(table_model)


#############################
# create function that is cartopy plotable
#############################

x0 = feature_transform.inverse_transform(np.zeros(feature_dim))[0]

# The parametrized function to be plotted
def f(x1=x0[0], x2=x0[1], x3=x0[2], x4=x0[3], x5=x0[4]):
    x = np.array([x1, x2, x3, x4, x5])
    x_s = feature_transform.fit_transform(x)
    y_pred_all = surrogate.predict(x_s)
    y_pred = y_pred_all[np.reshape(np.logical_and(season_info[0] == np.repeat('DJF',season_info[0].size), variable_info[0] == np.repeat('PRECT',variable_info[0].size)), y_pred_all.shape)].reshape(24, 48)
    y_pred_xr = xr.DataArray(y_pred, coords={"lat": dataset.lat, "lon": dataset.lon})
    if cfg["widget"]["plot_diff"]:
        y_diff_xr = y_pred_xr - Y_obs_raw[0]
        return y_diff_xr
    return y_pred_xr


# only works for a single field for now
def compute_wmse(x=None, Y=None):
    obs_factor = 1.0
    if cfg["widget"]["plot_diff"]:
        obs_factor = 0.0
    if x is not None:
        mse = (
            mse_transform.fit_transform((f(*x) - obs_factor * Y_obs_raw[0]) ** 2)
            / S[0].mean() ** 2
        )
    else:
        mse = (
            mse_transform.fit_transform((Y - obs_factor * Y_obs_raw[0]) ** 2)
            / S[0].mean() ** 2
        )
    return mse.values


#############################
# get e3sm color maps
#############################
from matplotlib.colors import LinearSegmentedColormap
from e3sm_cmap_colors import cet_rainbow, diverging_bwr, WBGYR


def convert_to_cmap(rgb_array):
    rgb_arr = rgb_array / 255.0
    cmap = LinearSegmentedColormap.from_list(name="temp", colors=rgb_arr)
    return cmap


colormap_dict = {
    "e3sm_default": convert_to_cmap(cet_rainbow),
    "e3sm_default_diff": convert_to_cmap(diverging_bwr),
    "e3sm_precip_diff": plt.get_cmap("BrBG"),
    "e3sm_precip": convert_to_cmap(WBGYR),
    "e3sm_wind": plt.get_cmap("PiYG_r"),
}

if cfg["widget"]["plot_diff"]:
    e3sm_cmap = colormap_dict["e3sm_default_diff"]
else:
    e3sm_cmap = colormap_dict["e3sm_default"]

#############################
# widget!
#############################

# setup figure and axis
fig = plt.figure(figsize=(12, 9))
panel = (0.1, 0.35, 0.8, 0.6)
ax = plt.axes(panel, projection=ccrs.PlateCarree())

# make initial plot
y0 = f(*x0)
y, lon = add_cyclic_point(y0, coord=y0.lon)
lat = y0.lat
ax.coastlines()
# cmap = plt.get_cmap("cividis")
cplot = ax.contourf(
    lon, lat, y, cmap=e3sm_cmap, extend="both", transform=ccrs.PlateCarree(), levels=30
)
gridliner = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.2)
gridliner.top_labels = False
gridliner.right_labels = False
divider = make_axes_locatable(ax)
cax = ax.inset_axes((1.02, 0, 0.02, 1))
# make a color bar axis
cbar = fig.colorbar(cplot, cax=cax, drawedges=True, alpha=0.5, pad=0.05)

# show titles
units = r"$W/m^2$"
ax.set_title("E3SMv2", loc="left", fontdict={"fontsize": 7.5})
ax.set_title(lat_lon_fields[0], fontdict={"fontsize": 14.0})
ax.set_title(units, loc="right", fontdict={"fontsize": 7.5})

# show stats
mse0 = compute_wmse(Y=y0)
stats_ = {"Min": y.min(), "Mean": y.mean(), "Max": y.max(), "wMSE": mse0}
stat_names = stats_.keys()
stat_values = np.array(list(stats_.values()))
stat_values = np.array(list(stats_.values()))
stat_values = ["{0:.3f}".format(v) for v in stat_values]
textvar1 = fig.text(
    panel[0] + panel[2] + 0.01,
    panel[0] + panel[3] + 0.22,
    "\n".join(stat_names),
    ha="left",
    fontdict={"fontsize": 7.5},
)
textvar2 = fig.text(
    panel[0] + panel[2] + 0.07,
    panel[0] + panel[3] + 0.22,
    "\n".join(stat_values),
    ha="right",
    fontdict={"fontsize": 7.5},
)

# Make a horizontal slider to control the frequency.
ax_slider1 = plt.axes([0.2, 0.3, 0.6, 0.03])
test_slider1 = Slider(
    ax=ax_slider1,
    label=x_labels[0],
    valmin=X_bounds[0][0],
    valmax=X_bounds[0][1],
    valinit=x0[0],
)

# Make a horizontal slider to control the frequency.
ax_slider2 = plt.axes([0.2, 0.25, 0.6, 0.03])
test_slider2 = Slider(
    ax=ax_slider2,
    label=x_labels[1],
    valmin=X_bounds[1][0],
    valmax=X_bounds[1][1],
    valinit=x0[1],
)

# Make a horizontal slider to control the frequency.
ax_slider3 = plt.axes([0.2, 0.2, 0.6, 0.03])
ax_slider3.xaxis.set_visible(True)
test_slider3 = Slider(
    ax=ax_slider3,
    label=x_labels[2],
    valmin=X_bounds[2][0],
    valmax=X_bounds[2][1],
    valinit=x0[2],
    alpha=0.8,
)
# # potential for posterior predictive
# ax_slider3.axvspan(0.3, 0.4, ymin=0.25, ymax=0.7, color="r", alpha=0.3)

# Make a horizontal slider to control the frequency.
ax_slider4 = plt.axes([0.2, 0.15, 0.6, 0.03])
test_slider4 = Slider(
    ax=ax_slider4,
    label=x_labels[3],
    valmin=X_bounds[3][0],
    valmax=X_bounds[3][1],
    valinit=x0[3],
)

# Make a horizontal slider to control the frequency.
ax_slider5 = plt.axes([0.2, 0.1, 0.6, 0.03])
test_slider5 = Slider(
    ax=ax_slider5,
    label=x_labels[4],
    valmin=X_bounds[4][0],
    valmax=X_bounds[4][1],
    valinit=x0[4],
)

# The function to be called anytime a slider's value changes
def update(val):
    yi0 = f(
        test_slider1.val,
        test_slider2.val,
        test_slider3.val,
        test_slider4.val,
        test_slider5.val,
    )
    yi, _ = add_cyclic_point(yi0, coord=y0.lon)
    ax.clear()
    ax.coastlines()
    cplot = ax.contourf(
        lon,
        lat,
        yi,
        cmap=e3sm_cmap,
        extend="both",
        transform=ccrs.PlateCarree(),
        levels=30,
    )
    gridliner = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.2)
    gridliner.top_labels = False
    gridliner.right_labels = False
    divider = make_axes_locatable(ax)
    cax = ax.inset_axes((1.02, 0, 0.02, 1))
    # make a color bar axis
    cbar = fig.colorbar(cplot, cax=cax, drawedges=True, alpha=1, pad=0.05)

    # show titles
    units = r"$W/m^2$"
    ax.set_title("E3SMv2", loc="left", fontdict={"fontsize": 7.5})
    ax.set_title(fields[0], fontdict={"fontsize": 14.0})
    ax.set_title(units, loc="right", fontdict={"fontsize": 7.5})

    # show stats
    msei = compute_wmse(Y=yi0)
    stats_ = {"Min": yi.min(), "Mean": yi.mean(), "Max": yi.max(), "wMSE": msei}
    stat_names = stats_.keys()
    stat_values = np.array(list(stats_.values()))
    stat_values = np.array(list(stats_.values()))
    stat_values = ["{0:.3f}".format(v) for v in stat_values]
    del fig.texts[-1]
    fig.text(
        panel[0] + panel[2] + 0.07,
        panel[0] + panel[3] + 0.22,
        "\n".join(stat_values),
        ha="right",
        fontdict={"fontsize": 7.5},
    )
    # fig.canvas.draw_idle()


# register the update function with each slider
test_slider1.on_changed(update)
test_slider2.on_changed(update)
test_slider3.on_changed(update)
test_slider4.on_changed(update)
test_slider5.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
button0 = 0.05
resetax = plt.axes([button0 + 0.2, 0.025, 0.1, 0.04])
default = plt.axes([button0 + 0.5, 0.025, 0.1, 0.04])
#mle_soln = plt.axes([button0 + 0.35, 0.025, 0.1, 0.04])
map_soln = plt.axes([button0 + 0.35, 0.025, 0.1, 0.04])
reset_button = Button(resetax, "Reset", hovercolor="0.475")
default_button = Button(default, "Default", hovercolor="0.475")
#mle_button = Button(mle_soln, "MLE", hovercolor="0.475")
map_button = Button(map_soln, "MAP", hovercolor="0.475")


def reset(event):
    test_slider1.reset()
    test_slider2.reset()
    test_slider3.reset()
    test_slider4.reset()
    test_slider5.reset()


# def mle_soln(event):
#     # combined SWCF, LSWCF, PRECT MLE with prescribed spatial variance
#     test_slider1.set_val(775.26)
#     test_slider2.set_val(2.66)
#     test_slider3.set_val(0.220)
#     test_slider4.set_val(6074.16)
#     test_slider5.set_val(-0.00042)


#def mle_soln(event):
#    # combined SWCF, LSWCF, PRECT MLE with prescribed scalar variance
#    test_slider1.set_val(431.72)
#    test_slider2.set_val(1.00)
#    test_slider3.set_val(0.50)
#    test_slider4.set_val(3306.73)
#    test_slider5.set_val(-0.00036)


# def mle_soln(event):
#     # individual MLE
#     test_slider1.set_val(840.0)
#     test_slider2.set_val(2.28)
#     test_slider3.set_val(0.1)
#     test_slider4.set_val(5528.79)
#     test_slider5.set_val(-0.00039)


def map_soln(event):
    # combined SWCF, LSWCF, PRECT MAP with prognostic single variance
    test_slider1.set_val(636.61)
    test_slider2.set_val(2.27)
    test_slider3.set_val(0.117)
    test_slider4.set_val(4516.43)
    test_slider5.set_val(-0.00034)


def default(event):
    test_slider1.set_val(500.0)
    test_slider2.set_val(2.40)
    test_slider3.set_val(0.12)
    test_slider4.set_val(3600)
    test_slider5.set_val(-0.0007)


reset_button.on_clicked(reset)
default_button.on_clicked(default)
#mle_button.on_clicked(mle_soln)
map_button.on_clicked(map_soln)

plt.show(block=False)
