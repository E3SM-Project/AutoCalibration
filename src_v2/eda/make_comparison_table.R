library(ncdf4)
library(fields)
library(dplyr)
library(tidyr)
# set surrogate settings and variables used
params_use <- list(nyears = '10yr',
                   resolution = '24x48', n_components = 16, 
                   variable = 'SWCF_LWCF_PRECT_PSL_Z500_U200_U850_TREFHT_U_RELHUM_T_RESTOM_0.7', 
                   season = 'ALL',
                   subtract = 'raw')
params <- params_use
print(unlist(params))

# break down variables by type
variable <- params[['variable']]
variables <- strsplit(variable, '_')[[1]]
if ('RESTOM' %in% variables) {
  variables <- variables[-length(variables)]
}
variable_types <- c('SWCF' = 'lat_lon', 'LWCF' = 'lat_lon', 'PRECT' = 'lat_lon',
                    'PSL' = 'lat_lon', 'Z500' = 'lat_lon', 'U200' = 'lat_lon',
                    'U850' = 'lat_lon', 'TREFHT' = 'lat_lon', 'U' = 'lat_plev',
                    'T' = 'lat_plev', 'RELHUM' = 'lat_plev', 'RESTOM' = 'global')
unique_variable_types  <-  unique(variable_types[variables])
variables_plev <- variables[variable_types[variables] == 'lat_plev']
variables_lon <- variables[variable_types[variables] == 'lat_lon']
variables_global <- variables[variable_types[variables] == 'global']
n_plev_vars  <- length(variables_plev)
n_lon_vars  <- length(variables_lon)
nvar <- length(variables)

# take into account seasons
if (params[['season']] == 'ALL') {
  eg <- expand.grid(variables, c('DJF', 'MAM', 'JJA', 'SON'),stringsAsFactors = F)
  if ('RESTOM' %in% variables) {
    eg <- rbind(eg, c('RESTOM', 'ANN'))
  }
  variables_vec <- as.character(eg[,1])
  seasons_vec <- eg[,2]
  variables <- sprintf('%s: %s', eg[,1], eg[,2])
  n_plev_vars  <- sum(variable_types[variables_vec] == 'lat_plev')
  n_lon_vars  <- sum(variable_types[variables_vec] == 'lat_lon')
  nvar <- n_plev_vars + n_lon_vars + 
    'RESTOM' %in% variables
} else {
  seasons_vec <- rep(params[['season']], nvar)
  variables_vec <- variables
}
misc_runs <- c('0522', '8fields_ALL_24x48', '8fields_ANN_24x48', 
               '11fields_ALL_orig', '11fields_ALL_20230227', '11fields_ALL_RESTOM.7', 
               'global_RMSE_RESTOM.7_sqrtn', 'global_RMSE_RESTOM.7_scalarboth', 
               'global_RMSE_RESTOM.7_scalarrmseatdefault')
# load in data
data_frame_list <- list()
for (j in 1:length(unique_variable_types)) {
  variables_use <- variables_vec[variable_types[variables_vec] == unique_variable_types[j]]
  seasons_use <- factor(seasons_vec[variable_types[variables_vec] == unique_variable_types[j]],
                        levels = c('DJF', 'MAM', 'JJA', 'SON'))
  
  if (unique_variable_types[j] == 'global') {
    # compute RESTOM
    ctrl_file <- nc_open(paste0('data/lat_lon_', params[['nyears']], '_180x360_ANN_ctrl.nc'))
    FLNT <- ncvar_get(ctrl_file, 'FLNT')
    FSNT <- ncvar_get(ctrl_file, 'FSNT')
    area <- ncvar_get(ctrl_file, 'area')
    nc_close(ctrl_file)
    area_norm <-  area/sum(area) 
    RESTOM_ctrl <- sum(area_norm * (FSNT - FLNT))
    data_frame_list[[length(data_frame_list) + 1]] <- data.frame('season' = NA, lon = NA, lat = NA, plev  = NA, type  = 'ctrl', 
                                                                 sim_run = NA, variable = 'RESTOM', res = NA, value = RESTOM_ctrl)
    
    RESTOM_misc <- rep(0, length(misc_runs))
    for (r in 1:length(misc_runs)) {
      misc_file <- nc_open(paste0('data/lat_lon_10yr_180x360_ANN_miscsim_',
                                  misc_runs[r], '.nc'))
      FLNT <- ncvar_get(misc_file, 'FLNT')
      FSNT <- ncvar_get(misc_file, 'FSNT')
      area <- ncvar_get(misc_file, 'area')
      area_norm <-  area/sum(area) 
      RESTOM_misc[r] <- sum(area_norm * (FSNT - FLNT))
      nc_close(misc_file)
    }
    names(RESTOM_misc) <- misc_runs
    data_frame_list[[length(data_frame_list) + 1]] <- data.frame('season' = NA, lon = NA, lat = NA, plev  = NA, type  = misc_runs,
                                                                 sim_run = NA, variable = 'RESTOM', res = NA, value = RESTOM_misc)
    next
  }
  
  q <- '180x360' # only do 180x360 and 180x37 fields
  for (i in 1:length(variables_use)) {
    obs_file <- nc_open(paste0('data/', unique_variable_types[j], '_',  q, '_', seasons_use[i], '_obs.nc'))
    ctrl_file <- nc_open(paste0('data/', unique_variable_types[j], '_', params[['nyears']], '_', 
                                q, '_', seasons_use[i], '_ctrl.nc'))
    if (unique_variable_types[j] == 'lat_lon') {
      lon <- obs_file[['dim']][['lon']][['vals']]
      lat <- obs_file[['dim']][['lat']][['vals']]
      lat_rep <- rep(lat, each = length(lon))
      plev <- NA
    } else {
      plev <- obs_file[['dim']][['plev']][['vals']]
      lon <- NA
      lat <- obs_file[['dim']][['lat']][['vals']]
      lat_rep <- lat
      plev <- rep(plev, each = length(lat_rep))
    }
    
    # obs
    data_frame_list[[length(data_frame_list) + 1]] <- 
      data.frame('season' = seasons_use[i], lon = lon, lat = lat_rep, 
                 plev = plev, type  = 'obs',  sim_run =NA, variable = variables_use[i], res = q,
                 value =  as.vector(ncvar_get(obs_file, variables_use[i])))
    
    # ctrl
    data_frame_list[[length(data_frame_list) + 1]] <- 
      data.frame('season' = seasons_use[i],lon = lon, lat = lat_rep, 
                 plev = plev, type = 'ctrl', sim_run =NA, variable = variables_use[i], res = q,
                 value =  as.vector(ncvar_get(ctrl_file, variables_use[i])))
    # misc
    for (k in 1:length(misc_runs)) {
      misc_file <- nc_open(paste0('data/', unique_variable_types[j], '_', params[['nyears']], '_', 
                                  q, '_',seasons_use[i], '_miscsim_', misc_runs[k], '.nc'))
      data_frame_list[[length(data_frame_list) + 1]] <- 
        data.frame('season' = seasons_use[i], lon = (lon), lat = (lat_rep), 
                   plev = (plev), type = misc_runs[k], sim_run =NA, variable = variables_use[i], res = q,
                   value =as.vector(ncvar_get(misc_file, as.character(variables_use[i]))))
      nc_close(misc_file)
    }
    nc_close(obs_file)
    nc_close(ctrl_file)
  }
}
df <- dplyr::bind_rows(data_frame_list)

# view RESTOM values
df %>%
  filter(variable == 'RESTOM') %>%
  mutate(value_round = round(value, 2)) %>%
  dplyr::select(type, value, value_round) %>%
  print(row.names = F)

# create area weights
area_file <- nc_open(paste0('surrogate_models/area_180x360.nc'))
area <- ncvar_get(area_file, 'values')
sim_file_test <- nc_open(paste0('data/lat_lon_', params[['nyears']], '_180x360_', 
                                'ANN', '.nc'))

area_lat <- sim_file_test$dim$lat$vals
area_lon <- sim_file_test$dim$lon$vals
area_norm <- area/mean(area)
nc_close(area_file)
nc_close(sim_file_test)
area_df <- data.frame(area_wt = as.vector(area_norm),
                       lat = rep(area_lat, each=  length(area_lon)), 
                       lon = area_lon, 
                       res = '180x360')

obs_file <- nc_open(paste0('data/lat_plev_10yr_180x360_DJF.nc'))
area_lat_plev <- ncvar_get(obs_file, 'area')

area_lat_plev_norm <- area_lat_plev[,1]/mean(area_lat_plev[,1])
lat_plev_vals <- obs_file$dim$lat$vals
area_lat_plev_df <- data.frame(area_wt_plev = area_lat_plev_norm,
                               lat = lat_plev_vals,
                               res = '180x360', is_plev = T)
nc_close(obs_file)

# merge with area weights
df_for_rmses <- df %>%
  filter(variable != 'RESTOM') %>%
  left_join(area_df) %>%
  mutate(is_plev = !is.na(plev)) %>%
  left_join(area_lat_plev_df) %>%
  mutate(area_wt_final = ifelse(is_plev, area_wt_plev, area_wt))

# compute rmse comparison
misc_runs_use  <- '11fields_ALL_RESTOM.7'
rmses <- df_for_rmses %>%
  tidyr::pivot_wider(names_from = 'type', values_from = 'value', 
                     id_cols = c('lon', 'lat', 'plev', 'variable', 'season', 'area_wt_final')) %>%
  group_by(season, variable) %>%
  summarize(rmse_ctrl = sqrt(mean(area_wt_final  * (ctrl - obs)^2, na.rm = T)),
            rmse_auto = sqrt(mean(area_wt_final * (`11fields_ALL_RESTOM.7` - obs)^2, na.rm = T))) %>%
  mutate(percent = (rmse_auto/rmse_ctrl - 1) * 100) %>%
  dplyr::select(season, variable, percent) %>%
  tidyr::pivot_wider(names_from = 'season', values_from = 'percent') 
# full table
rmses %>% 
  mutate_if(is.double, .funs = list(round), digits = 1)

# average over all entries
rmses %>% 
  select(-variable) %>%
  as.matrix(.) %>%
  mean(.) %>%
  data.frame(avg_value = ., avg_round = round(., 1))

# avg over seasons only
rmses %>%
  mutate(avg_value = 1/4 * (DJF + MAM + JJA + SON),
         avg_round = round(avg_value, 1)) %>%
  dplyr::select(variable, avg_value, avg_round) 

# avg over variables only
rmses %>% 
  select(-variable) %>%
  as.matrix(.) %>%
  colMeans(.) %>%
  data.frame(season = names(.), avg_value = ., avg_round = round(., 1))

