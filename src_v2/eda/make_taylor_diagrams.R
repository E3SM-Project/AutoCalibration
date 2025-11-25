library(ncdf4)
library(fields)
library(dplyr)
library(plotrix)
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

# take into oaccount seasons
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

# save simulation runs in tensors
data_frame_list <- list()
sim_tensor <- array(dim = c(180, 360, 250, 8, 4)) # lat, lon, simulation, variable, season
sim_tensor_plev <- array(dim = c(180, 37, 250, 3, 4)) # lat, plev, simulation, variable, season
unique_variable_types <- c('lat_lon', 'lat_plev')
for (j in 1:length(unique_variable_types)) {
  surr_file <- nc_open(paste0('surrogate_models/pred_',params[['season']], '_',  params[['nyears']], '_obs_full_', params[['resolution']], '_', params[['n_components']], '_',  params[['variable']], '_', params[['subtract']], '.nc'))
  
  variables_use <- variables_vec[variable_types[variables_vec] == unique_variable_types[j]]
  seasons_use <- factor(seasons_vec[variable_types[variables_vec] == unique_variable_types[j]],
                        levels = c('DJF', 'MAM', 'JJA', 'SON'))
  q <- '180x360'
  for (i in 1:length(variables_use)) {
    sim_file <- nc_open(paste0('data/', unique_variable_types[j], '_', params[['nyears']], '_',
                               q, '_', 
                               seasons_use[i], '.nc'))
    obs_file <- nc_open(paste0('data/', unique_variable_types[j], '_',  q, '_', seasons_use[i], '_obs.nc'))
    ctrl_file <- nc_open(paste0('data/', unique_variable_types[j], '_', params[['nyears']], '_', 
                                q, '_', seasons_use[i], '_ctrl.nc'))
    if (unique_variable_types[j] == 'lat_lon') {
      lon <- sim_file[['dim']][['lon']][['vals']]
      lat <- sim_file[['dim']][['lat']][['vals']]
      lat_rep <- rep(lat, each = length(lon))
      plev <- NA
      sim_run <- rep(1:250, each = length(lon) * length(lat)) 
    } else {
      mask <- ncvar_get(surr_file, 'mask')
      plev <- sim_file[['dim']][['plev']][['vals']]
      lon <- NA
      lat <- sim_file[['dim']][['lat']][['vals']]
      lat_rep <- lat
      plev <- rep(plev, each = length(lat_rep))
      sim_run <- rep(1:250, each = length(sim_file[['dim']][['plev']][['vals']]) * length(lat)) 
    }
    # simulations
    if (unique_variable_types[j] == 'lat_lon') {
      sim_tensor[,,,which(variables_lon == variables_use[i]),which(levels(seasons_use) == seasons_use[i])] <- ncvar_get(sim_file, variables_use[i])
    } else {
      sim_tensor_plev[,,,which(variables_plev == variables_use[i]),which(levels(seasons_use) == seasons_use[i])] <- ncvar_get(sim_file, variables_use[i])
    }
    
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
    nc_close(sim_file)
    nc_close(obs_file)
    nc_close(ctrl_file)
  }
}
df <- dplyr::bind_rows(data_frame_list)
area_file <- nc_open(paste0('surrogate_models/area_180x360.nc'))
area <- ncvar_get(area_file, 'values')
sim_file_test <- nc_open(paste0('data/lat_lon_', params[['nyears']], '_180x360_', 
                                'ANN', '.nc'))

area_lat <- sim_file_test$dim$lat$vals
area_lon <- sim_file_test$dim$lon$vals
area_norm <- area/mean(area)
nc_close(area_file)
nc_close(sim_file_test)
area_df2 <- data.frame(area_wt = as.vector(area_norm),
                       lat = rep(area_lat, each=  length(area_lon)), 
                       lon = area_lon, 
                       res = '180x360')

obs_file <- nc_open(paste0('data/lat_plev_5yr_180x360_DJF.nc'))
area_lat_plev <- ncvar_get(obs_file, 'area')

area_lat_plev_norm <- area_lat_plev[,1]/mean(area_lat_plev[,1])
lat_plev_vals <- obs_file$dim$lat$vals
area_lat_plev_df <- data.frame(area_wt_plev = area_lat_plev_norm,
                               lat = lat_plev_vals,
                               res = '180x360', is_plev = T)
nc_close(obs_file)

misc_runs_use  <- '11fields_ALL_RESTOM.7'
misc_runs_use_label <- 'Autotuned'
seq_unit <- seq(0, pi/2, length.out = 500)
variables_raw <- sapply(strsplit(variables, ':'), function(x) x[1])
variables_raw_unique <- unique(variables_raw)
variables_raw_unique <- variables_raw_unique[variables_raw_unique != 'RESTOM']
if (params[['subtract']] == 'ensmean') {
  df[['value_td']] <- df[['value']]  + df[['mv']]
} else {
  df[['value_td']] <- df[['value']]
}
png(paste0('src/eda/paper_plots/taylor_MAM_large.png'), width =2000, height =2000,res = 288)
par(xpd = FALSE) # make sure lines don't go outside plot
all_field <- filter(df, season == 'MAM', type != 'sim')
obs_field <- filter(all_field, type == 'obs', variable == variables_raw_unique[1]) %>% pull(value_td)

par(list(mfrow = c(1,1), mar = c(2,2,2,10))) # adjust margins
cex_use <- .5 # size of points
# initialize plot without plotting any points
taylor.diagram(obs_field,filter(all_field, type == 'ctrl', variable == variables_raw_unique[1]) %>% pull(value_td), 
               pch = -1, col ='white',
               main = '', 
               pcex = cex_use, show.gamma = T,
               normalize = T,
               xlab = 'Normalized Standard Deviation')
# use adjusted code that uses weights and has more control for plotting options
source('src/eda/taylor_diagram2.R')
taylor.diagram2(obs_field,filter(all_field, type == 'ctrl',  variable == variables_raw_unique[1]) %>% pull(value_td), 
                wts = area_norm, 
                col = which(variables_raw_unique == variables_raw_unique[1]), pch = 19, 
                pcex = cex_use, normalize = T, show.gamma = F, add = T)
lines(cos(seq_unit), sin(seq_unit))
for (r in 1:length(misc_runs_use)) {
  taylor.diagram2(obs_field,filter(all_field, type == misc_runs_use[r], variable == variables_raw_unique[1]) %>% pull(value_td), 
                  wts = area_norm, col = which(variables_raw_unique == variables_raw_unique[1]), pch = r, pcex = cex_use, normalize = T,
                  add = T,  show.gamma = F)
}
for (j in 2:length(variables_raw_unique)) {
  if (j > 8) {
    color_use <- c('orange', 'purple', 'pink')[j-8]
    names(area_lat_plev_norm) <- lat_plev_vals
    wts_use <- area_lat_plev_norm[as.character(filter(all_field, type == 'obs', 
                                                      variable == variables_raw_unique[j]) %>%pull(lat))]
  } else {
    color_use <- which(variables_raw_unique == variables_raw_unique[j])
    wts_use <- area_norm
  }
  obs_field <- filter(all_field, type == 'obs', variable == variables_raw_unique[j]) %>% pull(value_td)
  taylor.diagram2(obs_field,filter(all_field, type == 'ctrl', variable == variables_raw_unique[j]) %>% pull(value_td), 
                  wts = wts_use, col = color_use, pch = 19,
                  pcex = cex_use, normalize = T, add = T, show.gamma = F)
  for (r in 1:length(misc_runs_use)) {
    taylor.diagram2(obs_field,filter(all_field, type == misc_runs_use[r], variable == variables_raw_unique[j]) %>% pull(value_td), col = color_use, pch = r, pcex = cex_use, normalize = T,
                    add = T, wts = wts_use, show.gamma = F)
  }
  gc()
}
if (length(variables_raw_unique) > 8) {
  color_legend <- c(1:8, c('orange', 'purple', 'pink'))
} else {
  color_legend <- 1:length(variables_raw_unique)
}
legend('topleft', variables_raw_unique,ncol =2, col = color_legend, 
       pch = rep(19,  length(variables_raw_unique)), bg = 'white', cex = 1.2)
legend('topright', c('v2 Control', misc_runs_use_label), pch = c(19, 1:length(misc_runs_use)), bg = 'white',
       cex = 1.2)
dev.off()

seasons_unique <- c('DJF', 'MAM', 'JJA', 'SON')
for (q in seasons_unique) {
  gc()
  png(paste0('src/eda/paper_plots/taylor_',  q, '.png'), width =2000, height =2000,res = 288)
  par(xpd = FALSE)
  all_field_prelim <-  filter(df, season == q)
  
  obs_field <- filter(all_field_prelim, type == 'obs') %>% pull(value_td)
  par(list(mfrow = c(1,1), mar = c(4,4,4,4)))
  cex_use <- .75
  plot(xlim  = c(2/3, 4/3),  ylim = c(0, 2/3), 0, 0,
       main = '',
       xlab = 'Normalized Standard Deviation', 
       ylab = '')
  lines(cos(seq_unit), sin(seq_unit))
  source('src/eda/taylor_diagram2.R')
  for (j in 1:length(variables_raw_unique)) {
    all_field <-  filter(all_field_prelim, variable == variables_raw_unique[j])
    if (j > 8) {
      color_use <- c('orange', 'purple', 'pink')[j-8]
      wts_use <- area_lat_plev_norm[as.character(filter(all_field, type == 'obs') %>% pull(lat))]
    } else {
      color_use <- which(variables_raw_unique == variables_raw_unique[j])
      wts_use <- area_norm
    }
    obs_field <- filter(all_field, type == 'obs') %>% pull(value_td)
    for (i in 1:250) {
      if (j < 9) {
        taylor.diagram2(obs_field,as.double(sim_tensor[,,i,j,seasons_unique ==q ]), 
                        wts = wts_use, col = scales::alpha(color_use, .05), pch = 19, pcex = .4,
                        cex = .5, show.gamma = F, normalize = T, add = T)
      } else {
        plev_var <- as.double(sim_tensor_plev[,,i,variables_plev == variables_raw_unique[j],seasons_unique ==q ]) 
        taylor.diagram2(obs_field[!is.na(plev_var)],plev_var[!is.na(plev_var)], col = scales::alpha(color_use, .05), pch = 19, pcex = .4,
                        wts = wts_use[!is.na(plev_var)], cex = .5, show.gamma = F, normalize = T, add = T)
      }
    }
    print(j)
    gc()
  }
  
  all_field_prelim <- filter(df, res == '180x360', season == q)
  gc()
  for (j in 1:length(variables_raw_unique)) {
    all_field <-  filter(all_field_prelim, variable == variables_raw_unique[j])
    if (j > 8) {
      color_use <- c('orange', 'purple', 'pink')[j-8]
      wts_use <- area_lat_plev_norm[as.character(all_field$lat)]
    } else {
      color_use <- which(variables_raw_unique == variables_raw_unique[j])
      wts_use <- area_norm
    }
    obs_field <- filter(all_field, type == 'obs') %>% pull(value_td)
    taylor.diagram2(obs_field, wts = wts_use, filter(all_field, type == 'ctrl') %>% pull(value_td), col = color_use, pch = 19, 
                    pcex = cex_use, show.gamma = F,normalize = T, add = T)
    for (r in 1:length(misc_runs_use)) {
      taylor.diagram2(obs_field,wts = wts_use, filter(all_field, type == misc_runs_use[r]) %>% pull(value_td), col = color_use, pch = r, pcex = cex_use, normalize = T,
                      add = T, show.gamma = F)
    }
    gc()
  }
  if (length(variables_raw_unique) > 8) {
    color_legend <- c(1:8, c('orange', 'purple', 'pink'))
  } else {
    color_legend <- 1:length(variables_raw_unique)
  }
  legend('bottomleft', ncol = 2, variables_raw_unique, col =color_legend, 
         pch = rep(19,  length(variables_raw_unique)), bg = 'white', cex = 1)
  legend('bottomright',  c('v2 Control', misc_runs_use_label, '250 Simulation Runs'),
         pch = c(19, 1:length(misc_runs_use), 19), 
         bg = 'white', cex = 1,pt.cex = c(1,rep(1, length(misc_runs_use)),.4), 
         col = c(1, rep(1, length(misc_runs_use)),  scales::alpha('black', .1)))
  dev.off()
}
