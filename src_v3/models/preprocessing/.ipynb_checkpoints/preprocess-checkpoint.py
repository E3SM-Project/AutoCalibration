import numpy as np
import os, pdb
import pandas as pd
import fnmatch
from tqdm import tqdm
import pickle
from time import time
import glob
import sys
import xml.etree.ElementTree as ET
import yaml
import git
import copy
import xarray as xr
import dask
from sklearn.model_selection import ParameterGrid


# cfg = yaml.safe_load(open('config_preprocessing.yaml'))
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
    
savedir = cfg["savedir"]
if not os.path.exists(savedir):
    os.mkdir(savedir)
    
verbose = cfg["verbose"]

cfg_obs = cfg["obs"]
obs_variable_names = cfg_obs["obs_variable_names"]

cfg_model = cfg["model"]
cfg_model_responses = cfg_model["responses"]
model_variable_names = cfg_model_responses["model_variable_names"]
model_to_obs_transforms = cfg_model_responses["model_to_obs_transforms"]


def get_params(param_path):
    """
    Extract input parameters
    """
    params = {pn: None for pn in cfg_model["params"]["param_names"]}
    if cfg_model["params"]["file_type"] == "xml":
        atm_tree = ET.parse(param_path)
        atm_root = atm_tree.getroot()

        for pn in cfg_model["params"]["param_names"]:
            for x in atm_root.iter(pn):
                params[pn] = float(x.text)
    elif cfg_model["params"]["file_type"] == "atm_in":
        with open(param_path, "r") as file:
            for line in file:
                line_split = line.split('=')
                param_label = line_split[0].strip()
                if param_label in cfg_model["params"]["param_names"]:
                    param_str = line_split[1].strip()
                    params[param_label] = float(param_str.replace('D', 'E'))
    else:
        assert cfg_model["params"]["file_type"] in ["xml", "atm_in"], "file_type not supported"

    return params


def get_responses(srcdir, files, variable_names):
    full_paths = [os.path.join(srcdir, file) for file in files]
    ds_all = []
    for fp in full_paths:
        try:
            ds = xr.open_mfdataset(glob.glob(fp)[0], parallel=True)
               
            # extract specified output variables
            ds_vars = sorted(set(variable_names) & set(ds.keys()))
            arr_list = {}
            for var in ds_vars:
                arr_list[var] = ds[var].values
            ds = ds[ds_vars]
            if 'time' in ds.coords:
                ds = ds.drop_vars('time')

            # use xarray to open the data set and use dask to import as chunks
            ds_all.append(ds)
        except:
            if verbose:
                print(f"#### No data loaded from '{fp}' ####")
            return None

    ds_all = xr.merge(ds_all, compat = "override")
    return ds_all
    
    
def get_sim(response_srcdir, workdir, response_files, src_name, param_srcdir, param_file):
    obs_vars = obs_variable_names[src_name].copy() 
    model_vars = model_variable_names[src_name].copy()
    
    ds = get_responses(srcdir = os.path.join(response_srcdir, workdir),
                       files = response_files,
                       variable_names = model_vars)

    if ds is not None:
        if model_to_obs_transforms is not None:
            if src_name in model_to_obs_transforms:
                for mod_var in model_vars: # save model_vars as their own variables
                    exec(f"{mod_var} = ds['{mod_var}']")
                for obs_var in model_to_obs_transforms[src_name]: # calculate transforms 
                    exec(f"ds['{obs_var}'] = {model_to_obs_transforms[src_name][obs_var]}")
                for mod_var in list(set(ds.keys()) & (set(model_vars) - set(obs_vars))): # delete model_vars not in obs_vars
                    ds = ds.drop_vars(mod_var)

        # get input parameter, convert to xarray and add to dataset
        param_path = os.path.join(param_srcdir, workdir, param_file) 
        params = get_params(param_path)
        param_xr = xr.DataArray(
            list(params.values()), dims=("input_params"), coords={"input_params": cfg_model["params"]["param_names"]}
        )  # e.g., x_xr.sel(x='zmconv_dmpdz')
        ds["params"] = param_xr

    return ds


########################################################################
# get observational data
if cfg_obs["process"]:        
    obs_name = cfg_obs["name"]
    for src_name in cfg_obs["srcdirs"].keys():
        print(f"\n#### Exporting observations to '{src_name}_{obs_name}.nc' ####")
        obs = get_responses(srcdir = cfg_obs["srcdirs"][src_name],
                            files = cfg_obs["files"][src_name],
                            variable_names = obs_variable_names[src_name] 
                           )
        
        if obs is None:
            print(f"#### Obs not loaded for {obs_variable_names[src_name]} ####")
        else:
            savedir_obs = os.path.join(savedir, 'obs')
            if not os.path.exists(savedir_obs):
                os.mkdir(savedir_obs)
            savefile = f"{src_name}_{obs_name}.nc"
            obs.to_netcdf(os.path.join(savedir_obs, savefile))


########################################################################
# get model/simulation datasets
for ds_name,cfg_ds in cfg_model["datasets"].items():
    if cfg_ds["process"]:
        # get list of workdirs
        workdirs = []
        for p in cfg_ds["paths"]:
            workdirs.extend(
                glob.glob(os.path.join(
                    list(cfg_model["responses"]["srcdirs"].values())[0],
                    p
                ))
            )

        # extract and save all data from different output file names
        print(f"\n#### {ds_name}['paths'] includes {len(workdirs)} workdirs ####")
        for src_name in cfg_model["responses"]["srcdirs"].keys():
            savedir_ds = os.path.join(savedir, ds_name)
            print(f"\nExporting ensemble to '{savedir_ds}'/'{src_name}_{ds_name}.nc'")

            # extract data from all workdirs
            ds_list = []; valid_workdirs = []
            for wd in tqdm(workdirs):
                try:
                    ds_wd = get_sim(cfg_model["responses"]["srcdirs"][src_name],
                                    wd,
                                    cfg_model["responses"]["files"][src_name],
                                    src_name,
                                    cfg_model["params"]["srcdirs"][src_name],
                                    cfg_model["params"]["file"] 
                                   )
                except:
                    breakpoint()
                if ds_wd is not None:
                    ds_list.append(ds_wd)
                    valid_workdirs.append(wd)

            if len(ds_list) > 0:
                # concatenate data from all workdirs
                ds_all = xr.concat(ds_list, dim="workdir")
                ds_all = ds_all.assign_coords({"workdir": valid_workdirs})
            else:
                print(f"\nSimulation data not loaded for {model_variable_names[src_name]}")

            # save processed data
            if not os.path.exists(savedir_ds):
                os.mkdir(savedir_ds)
            savefile = f"{src_name}_{ds_name}.nc"
            ds_all.to_netcdf(os.path.join(savedir_ds, savefile))
            print(f"Exported {len(ds_list)} ensemble members (out of {len(workdirs)} workdirs)")
            

########################################################################
# Save specs
savedir_specs = os.path.join(savedir, 'specs')
if not os.path.exists(savedir_specs):
    os.mkdir(savedir_specs)
specdir = os.path.join(savedir_specs, "specs")
v = 1
while os.path.exists(specdir):
    v += 1
    specdir = os.path.join(savedir_specs, "specs" + str(v))
os.mkdir(specdir)

files_to_cp = ['preprocess.py', 'config_preprocessing.yaml']
for f in files_to_cp:
    os.system('cp ./' + f + ' ' + os.path.join(specdir, f))

# Save environment info
savedir_env = os.path.join(savedir, 'env')
if not os.path.exists(savedir_env):
    os.mkdir(savedir_env)
envdir = os.path.join(savedir_env, "env")
v = 1
while os.path.exists(envdir):
    v += 1
    envdir = os.path.join(savedir_env, "env" + str(v))
os.mkdir(envdir)

machine = cfg["machine"]
git_repo = git.Repo(search_parent_directories=True)
git_root = git_repo.git.rev_parse('--show-toplevel')

files_to_cp = [f'autotuning_environment_{machine}.yml', f'autotuning_spec_file_{machine}.txt', 'README.md', f'requirements_{machine}.txt']
for f in files_to_cp:
    os.system('cp ' + os.path.join(git_root, f) + ' ' + os.path.join(envdir, f))

# create environment provenance file
git_hash = git_repo.head.object.hexsha # what was the current git hash (aka git sha)?
preprocessing_dir = os.path.abspath("./") # where did we run this script?
with open(os.path.join(envdir, "provenance"), 'w') as f:
    f.write("git hash: " + git_hash + "\n")
    f.write("machine: " + machine + "\n")
    f.write("preprocessing directory: " + preprocessing_dir + "\n")

print("")
