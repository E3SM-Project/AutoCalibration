library(ncdf4)
library(fields)
library(ggplot2)
library(dplyr)
theme_set(theme_bw())
params <- list(nyears = '10yr',
               resolution = '24x48', n_components = 16, variable = 'SWCF_LWCF_PRECT_PSL_Z500_U200_U850_TREFHT_U_RELHUM_T', season = 'ALL',
               subtract = 'raw')
print(unlist(params))
variable <- params[['variable']]
variables <- strsplit(variable, '_')[[1]]
variable_types <- c('SWCF' = 'lat_lon', 'LWCF' = 'lat_lon', 'PRECT' = 'lat_lon',
                    'PSL' = 'lat_lon', 'Z500' = 'lat_lon', 'U200' = 'lat_lon',
                    'U850' = 'lat_lon', 'TREFHT' = 'lat_lon', 'U' = 'lat_plev',
                    'T' = 'lat_plev', 'RELHUM' = 'lat_plev', 'RESTOM' = 'global')
units <- c('SWCF' = 'W/m^2', 'LWCF' = 'W/m^2','PRECT' = 'mm/day',
           'PSL' = 'Pa', 'Z500' = 'm', 'U200' = 'm/s',
           'U850' = 'm/s', 'TREFHT' = 'K', 'U' = 'm/s',
           'T' = 'K', 'RELHUM' = '%')

unique_variable_types  <-  unique(variable_types[variables])
variables_plev <- variables[variable_types[variables] == 'lat_plev']
variables_lon <- variables[variable_types[variables] == 'lat_lon']
n_plev_vars  <- length(variables_plev)
n_lon_vars  <- length(variables_lon)

nvar <- length(variables)
## Load model outputs
# get simulation runs created by Kenny's code
if (params[['season']] == 'ALL') {
  eg <- expand.grid(variables, c('DJF', 'MAM', 'JJA', 'SON'))
  variables_vec <- as.character(eg[,1])
  seasons_vec <- eg[,2]
  variables <- sprintf('%s: %s', eg[,1], eg[,2])
  n_plev_vars  <- sum(variable_types[variables_vec] == 'lat_plev')
  n_lon_vars  <- sum(variable_types[variables_vec] == 'lat_lon')
  nvar <- n_plev_vars + n_lon_vars
} else {
  seasons_vec <- rep(params[['season']], nvar)
  variables_vec <- variables
}
misc_runs <- c('0522', '8fields_ALL_24x48', '8fields_ANN_24x48', 
               '11fields_ALL_orig', '11fields_ALL_20230227', '11fields_ALL_RESTOM.7', 
               'global_RMSE_RESTOM.7_sqrtn', 'global_RMSE_RESTOM.7_scalarboth', 
               'global_RMSE_RESTOM.7_scalarrmseatdefault')
data_frame_list <- list()
j <- 1
surr_file <- nc_open(paste0('surrogate_models/pred_',params[['season']], '_',  params[['nyears']], '_full_', params[['resolution']], '_', params[['n_components']], '_',  params[['variable']], '_', params[['subtract']], '.nc'))

variables_use <- variables_vec[variable_types[variables_vec] == unique_variable_types[j]]
seasons_use <- seasons_vec[variable_types[variables_vec] == unique_variable_types[j]]
q <- '180x360'
for (i in 1:length(variables_use)) {
  obs_file <- nc_open(paste0('data/', unique_variable_types[j], '_',  q, '_', seasons_use[i], '_obs.nc'))
  ctrl_file <- nc_open(paste0('data/', unique_variable_types[j], '_', params[['nyears']], '_', 
                              q, '_', seasons_use[i], '_ctrl.nc'))
  lon <- obs_file[['dim']][['lon']][['vals']]
  lat <- obs_file[['dim']][['lat']][['vals']]
  lat_rep <- lat_surr_rep <- rep(lat, each = length(lon))
  plev <- plev_surr <- NA
  
  # obs
  data_frame_list[[length(data_frame_list) + 1]] <- 
    data.frame('season' = seasons_use[i], lon = (lon), lat = (lat_rep), 
               plev = (plev), type  = 'obs',  sim_run =NA, variable = variables_use[i], res = q,
               value =  as.vector(ncvar_get(obs_file, variables_use[i])))
  
  # ctrl
  data_frame_list[[length(data_frame_list) + 1]] <- 
    data.frame('season' = seasons_use[i],lon = (lon), lat = (lat_rep), 
               plev = (plev), type = 'ctrl', sim_run =NA, variable = variables_use[i], res = q,
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
df <- dplyr::bind_rows(data_frame_list)

df_compare <- df %>%
  filter(type %in% c('11fields_ALL_20230227', 'ctrl', 'obs')) %>%
  tidyr::pivot_wider(values_from = 'value', names_from = 'type',
                     id_cols = c('season', 'lon', 'lat', 'variable')) %>%
  mutate(diff_ctrl = ctrl - obs,
         diff_auto = `11fields_ALL_20230227` - obs) %>%
  tidyr::pivot_longer(values_to = 'value', cols = starts_with('diff'), 
                      names_to = 'name')
df_type   <- data.frame(name = c('diff_auto', 'diff_ctrl'),
                        name_label = c('v2 Autotuned', 'v2 Control'))

for (i in 1:length(unique(df_compare$variable))) {
  for (j in 1:4) {
    variable_plot <- unique(df_compare$variable)[i]
    season_plot <- c('DJF', 'MAM', 'JJA', 'SON')[j]
    units_plot <- units[[variable_plot]]
    quantile_diff <- df_compare %>% filter(variable == variable_plot, season == season_plot) %>%
      left_join(df_type) %>%
      pull(value) %>%
      quantile(c(.001, .999))
    a <- ggplot() +
      geom_raster(data = df_compare %>% filter(variable == variable_plot, season == season_plot) %>%
                    left_join(df_type),
                  aes(x = lon, y = lat, fill = value)) + 
      geom_polygon(data = map_data('world2'), aes(x = long, y = lat, group = group), 
                   fill = NA, color = 'grey10', alpha = .1, size = .02) + 
      facet_wrap(~name_label, ncol = 1) + 
      scale_fill_steps2(n.breaks = 12, limits = quantile_diff,
                        oob = scales::squish) + 
      coord_quickmap() +
      labs(fill = paste0(variable_plot, '\n', 'Diff', '\n(',units_plot, ')'),
           title = 'Difference between E3SM output and observations',
           subtitle = paste0(variable_plot, ', ', season_plot)) + 
      theme(axis.title = element_blank(),  legend.key.height = unit(1.5, 'cm')) +
      guides(fill = guide_colorsteps(show_limits = T))
    ggsave(plot = a, filename = paste0('src/eda/paper_plots/difference_plots/', variable_plot, '_', season_plot, '.png'),
           height = 6.25, width = 5.5)
  }
  print(i)
}
