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

res_vals_lon <- as.numeric(strsplit(params[['resolution']], 'x')[[1]])
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
               '11fields_ALL_24x48')

data_frame_list <- list()
sim_tensor <- array(dim = c(180, 360, 250, 8, 4))
j <- 1
surr_file <- nc_open(paste0('surrogate_models/pred_',params[['season']], '_',  params[['nyears']], '_full_', params[['resolution']], '_', params[['n_components']], '_',  params[['variable']], '_', params[['subtract']], '.nc'))

variables_use <- variables_vec[variable_types[variables_vec] == unique_variable_types[j]]
seasons_use <- seasons_vec[variable_types[variables_vec] == unique_variable_types[j]]
q <- '24x48'
for (i in 1:length(variables_use)) {
  sim_file <- nc_open(paste0('data/', unique_variable_types[j], '_', params[['nyears']], '_',
                             q, '_', 
                             seasons_use[i], '.nc'))
  obs_file <- nc_open(paste0('data/', unique_variable_types[j], '_',  q, '_', seasons_use[i], '_obs.nc'))
  ctrl_file <- nc_open(paste0('data/', unique_variable_types[j], '_', params[['nyears']], '_', 
                              q, '_', seasons_use[i], '_ctrl.nc'))
  lon <- sim_file[['dim']][['lon']][['vals']]
  lat <- sim_file[['dim']][['lat']][['vals']]
  lat_rep <- lat_surr_rep <- rep(lat, each = length(lon))
  plev <- plev_surr <- NA
  sim_run <- rep(1:250, each = length(lon) * length(lat)) 
  indexes_use  <- list(1:length(lon), 1:length(lat))
  indexes_variable_all <- (1 + (i-1)*prod(res_vals_lon)):
    (i*prod(res_vals_lon))
  if (q == params[['resolution']]) {
    surr_pred <- ncvar_get(surr_file, 'values')[indexes_variable_all,]
  }
  # simulations
  data_frame_list[[length(data_frame_list) + 1]] <- 
    data.frame('season' = seasons_use[i], lon, lat = lat_rep, plev, type  = 'sim', sim_run = as.character(sim_run), variable = variables_use[i], 
               res = q,
               value = as.vector(ncvar_get(sim_file, variables_use[i])))
  
  # surrogate
  if (params[['resolution']] == q) {
    data_frame_list[[length(data_frame_list) + 1]] <- 
      data.frame('season' = seasons_use[i], lon = (lon), lat = (lat_surr_rep), plev  = (plev_surr), type  = 'surr', 
                 sim_run = as.character(sim_run), variable = variables_use[i], res = q,
                 value = as.vector(surr_pred)) 
  }
  nc_close(sim_file)
}
df <- dplyr::bind_rows(data_frame_list)
df_compare <- df %>%
  filter(res == '24x48', type %in% c('surr', 'sim')) %>%
  tidyr::pivot_wider(values_from = 'value', names_from = 'type') %>%
  mutate(value = sim  - surr)
df_type   <- data.frame(name = c('diff_auto', 'diff_ctrl'),
                        name_label = c('v2 Autotuned', 'v2 Control'))
set.seed(10)
for (i in 1:length(unique(df_compare$variable))) {
  for (j in 1:4) {
    r_simu <- sample(1:250, 1)
    variable_plot <- unique(df_compare$variable)[i]
    season_plot <- c('DJF', 'MAM', 'JJA', 'SON')[j]
    units_plot <- units[[variable_plot]]
    a <- ggplot() +
      geom_raster(data = df_compare %>% filter(variable == variable_plot, season == season_plot),
                  aes(x = lon, y = lat, fill = value)) + 
      geom_polygon(data = map_data('world2'), aes(x = long, y = lat, group = group), 
                   fill = NA, color = 'grey10', alpha = .1, size = .02) + 
      scale_fill_steps2(n.breaks = 12)+
      coord_quickmap() +
      labs(fill = paste0(variable_plot, '\n', 'Diff', '\n(',units_plot, ')'),
        title = 'Difference between E3SM output and surrogate predictions',
        subtitle = paste0(variable_plot, ', ', season_plot)) + 
      theme(axis.title = element_blank(),  legend.key.height = unit(1.5, 'cm')) +
      guides(fill = guide_colorsteps(show_limits = T))
    ggsave(plot = a, filename = paste0('src/eda/paper_plots/residual_plots/', variable_plot, '_', season_plot, '.png'),
           height = 6.25/1.8, width = 5.5)
  }
  print(i)
}
