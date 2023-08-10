library(ncdf4)
library(ggplot2)
library(dplyr)
library(cowplot)
library(purrr)
library(fields)
theme_set(theme_bw() + 
            theme(axis.text = element_blank(),
                  axis.title =  element_blank(),
                  axis.ticks = element_blank(),
                  legend.key.size = unit(.5, "lines"),
                  legend.text = element_text(size = 5), 
                  legend.title = element_text(size = 6),
                  strip.text = element_text(size = 6)))
# load principal components
params <- list(nyears = '10yr', resolution = '24x48', n_components = 16, 
                   variable = 'SWCF_LWCF_PRECT_PSL_Z500_U200_U850_TREFHT_U_RELHUM_T_RESTOM_0.7', 
                   season = 'ALL', subtract = 'raw')
pc_file <- nc_open(paste0('surrogate_models/pcs_',params[['season']], '_',  params[['nyears']], '_obs_full_',
                          params[['resolution']], '_', params[['n_components']], '_',  params[['variable']], '_', params[['subtract']], '.nc'))
pc_scores_model <- ncvar_get(pc_file, 'pc_scores_model')
pc_scores_surr <- ncvar_get(pc_file, 'pc_scores_surr')
pc_scores_obs <- ncvar_get(pc_file, 'pc_scores_obs')
pc_vals <- ncvar_get(pc_file, 'pc_vals')
ev <- ncvar_get(pc_file, 'ev')
sing_vals <- ncvar_get(pc_file, 'sing_vals')
pc_prop_var <- ncvar_get(pc_file, 'prop_var')
nc_close(pc_file)

# get simulation runs created by Kenny's code
print(unlist(params))
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

res_vals_lon <- as.numeric(strsplit(params[['resolution']], 'x')[[1]])
res_vals_plev <- c(24, 37)
nvar <- length(variables)
if (params[['season']] == 'ALL') {
  eg <- expand.grid(variables[variables != 'RESTOM'], c('DJF', 'MAM', 'JJA', 'SON'),
                    stringsAsFactors = F)
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
               '11fields_ALL_24x48')
data_frame_pcs_list <- list()
for (j in 1:length(unique_variable_types)) {
  surr_file <- nc_open(paste0('surrogate_models/pred_',params[['season']], '_',  params[['nyears']], 
                              '_obs_full_', params[['resolution']], '_', params[['n_components']], '_', 
                              params[['variable']], '_', params[['subtract']], '.nc'))
  variables_use <- variables_vec[variable_types[variables_vec] == unique_variable_types[j]]
  seasons_use <- seasons_vec[variable_types[variables_vec] == unique_variable_types[j]]
  if (unique_variable_types[j] == 'global') {
    data_frame_pcs_list[[length(data_frame_pcs_list) + 1]] <- 
      data.frame('season' = 'ANN', lon = NA, lat = NA, plev  = NA, pc = 1:(dim(pc_vals)[2]),
                 variable = 'RESTOM', res = '180x360',
                 value = as.vector(pc_vals[nrow(pc_vals),]) )
    next
  }
  q <- params[['resolution']]
  for (i in 1:length(variables_use)) {
    sim_file <- nc_open(paste0('data/', unique_variable_types[j], '_', params[['nyears']], '_',
                               q, '_', 
                               seasons_use[i], '.nc'))
    if (unique_variable_types[j] == 'lat_lon') {
      lon <- sim_file[['dim']][['lon']][['vals']]
      lat <- sim_file[['dim']][['lat']][['vals']]
      lat_rep <- lat_surr_rep <- rep(lat, each = length(lon))
      plev <- plev_surr <- NA
      sim_run <- rep(1:250, each = length(lon) * length(lat)) 
      indexes_use  <- list(1:length(lon), 1:length(lat))
      indexes_variable_all <- (1 + (i-1)*prod(res_vals_lon)):
        (i*prod(res_vals_lon))
      surr_pred <- ncvar_get(surr_file, 'values')[indexes_variable_all,]
      pcs_use <- pc_vals[indexes_variable_all,]
    } else {
      mask <- ncvar_get(surr_file, 'mask')
      plev <- sim_file[['dim']][['plev']][['vals']]
      lon <- NA
      lat <- sim_file[['dim']][['lat']][['vals']]
      lat_surr_rep <- rep(lat, each =  length(plev))
      plev_surr <-  plev
      lat_rep <- lat
      plev <- rep(plev, each = length(lat_rep))
      sim_run <- rep(1:250, each = length(plev_surr) * length(lat)) 
      indexes_variable_all <- (1 + n_lon_vars*(res_vals_lon[2] * res_vals_lon[1]) +
                                 sum(mask[,1:i]) - sum(mask[,i])):
        (n_lon_vars*(res_vals_lon[2] * res_vals_lon[1]) + sum(mask[,1:i]))
      surr_pred_prelim <- ncvar_get(surr_file, 'values')[indexes_variable_all,]
      surr_pred <- matrix(NA, nrow = prod(res_vals_plev), ncol = 250)
      surr_pred[as.logical(mask[,i]), ] <- surr_pred_prelim
      
      pcs_prelim <- pc_vals[indexes_variable_all,]
      pcs_use <- matrix(NA, nrow = prod(res_vals_plev), ncol = 16)
      pcs_use[as.logical(mask[,i]), ] <- pcs_prelim
    }
    nc_close(sim_file)

    #pc values
    data_frame_pcs_list[[length(data_frame_pcs_list) + 1]] <- 
      data.frame('season' = seasons_use[i], lon, lat = lat_surr_rep, plev  = plev_surr, pc = rep(1:(dim(pc_vals)[2]), each = length(lat_surr_rep)),
                 variable = variables_use[i], res = q,
                 value = as.vector(pcs_use))
  }
  nc_close(surr_file)
}



## Plots of principal component vectors
pc_vals_df <- dplyr::bind_rows(data_frame_pcs_list)
pc_vals_df %>%
  filter(!is.na(lon), pc %in% 1:5) %>%
  mutate(variable_pc = variable) %>%
  group_split(variable_pc) %>% 
  map(
    ~ggplot(.,aes(x= lon, y  = lat, fill = value))+
      geom_raster() +
      scale_fill_gradientn(colors = rainbow(10)) + 
      facet_grid(season~pc) +  
      labs(fill = .$variable_pc[1]) + 
      theme(legend.margin=margin(0,0,0,0),
            legend.box.margin=margin(-10,-10,-10,-10), 
            axis.text = element_text(size = 6),
            axis.title =  element_blank(),
            #axis.ticks = element_blank(),
            legend.key.size = unit(.5, "lines"),
            legend.text = element_text(size = 5), 
            legend.title = element_text(size = 6),
            strip.text = element_text(size = 6))
  ) %>%
  plot_grid(plotlist = ., align = 'hv')

plevels <- unique(pc_vals_df$plev)[!is.na(unique(pc_vals_df$plev))]
plevels_df <- data.frame(plev = plevels, 
                         plev_lower = lead(plevels))
plevels_df$plev_lower[is.na(plevels_df$plev_lower)] <- 0

lats <- unique(pc_vals_df$lat)
lat_lag <- lats[2] - lats[1]
lats_df <- data.frame(lat = lats,
                      lat_higher = lats + lat_lag/2, 
                      lat_lower = lats - lat_lag/2)

pc_vals_df %>%
  filter(!is.na(plev), pc %in% 1:5) %>%
  mutate(variable_pc = variable) %>%
  left_join(plevels_df) %>%
  left_join(lats_df) %>%
  group_split(variable_pc) %>% 
  map(
    ~ggplot(., aes(ymin = plev_lower, ymax = plev, xmin = lat_lower, xmax = lat_higher, color = value, fill = value)) + 
      geom_rect() + 
      scale_color_gradientn(colors = rainbow(10)) + 
      scale_fill_gradientn(colors = rainbow(10)) + 
      scale_y_continuous(trans = 'reverse') + 
      facet_grid(season+variable_pc~pc) +  
      labs(fill = .$variable_pc[1]) + 
      theme(legend.margin=margin(0,0,0,0),
            legend.box.margin=margin(-10,-10,-10,-10))
  ) %>%
  plot_grid(plotlist = ., align = 'hv')
  
pc_vals_df %>%
  filter(!is.na(lon), pc == 1) %>%
  mutate(variable_pc = variable) %>%
  ggplot(aes(x= lon, y  = lat, fill = value))+
  geom_raster() +
  scale_fill_gradient2() + 
  facet_grid(season~variable) +  
  theme(legend.margin=margin(0,0,0,0),
        legend.box.margin=margin(-10,-10,-10,-10))

cet_perp_unif <- cetcolor::cet_pal('r2')
save_plot <- pc_vals_df %>%
  filter(!is.na(lon), pc == 1) %>%
  mutate(variable_pc = variable) %>%
  group_split(variable_pc) %>% 
  map({
    ~{    if (.$variable[1] == 'LWCF') {
      theme_use <-       theme(legend.key.size = unit(.3, "lines"),
                               legend.margin=margin(-10,0,-10,-10), 
                               axis.text.x = element_blank(),
                               axis.title.x =  element_blank(),
                               axis.ticks.x = element_blank(),
                               axis.text.y = element_text(size = 6),
                               axis.title.y =  element_text(size = 7),
                               legend.text = element_text(size = 5), 
                               legend.title = element_text(size = 6),
                               strip.text = element_text(size = 6),
                               plot.margin=grid::unit(c(0,2,0,2), "mm"))
    } else if (.$variable[1] != 'Z500') {
      theme_use <- theme(legend.key.size = unit(.3, "lines"),
                         legend.margin=margin(-10,0,-10,-10), 
                         axis.text.x = element_blank(),
                         axis.title.x =  element_blank(),
                         axis.ticks.x = element_blank(),
                         axis.text.y = element_text(size = 6),
                         axis.title.y =  element_text(size = 7),
                         strip.background = element_blank(),
                         strip.text.x = element_blank(),
                         legend.text = element_text(size = 5), 
                         legend.title = element_text(size = 6),
                         strip.text = element_text(size = 6),
                         plot.margin=grid::unit(c(0,2,0,2), "mm"))
    } else {
      theme_use <- theme(legend.key.size = unit(.3, "lines"),
                         legend.margin=margin(-10,0,-10,-10), 
                         axis.text = element_text(size = 6),
                         axis.title =  element_text(size = 7),
                         strip.background = element_blank(),
                         strip.text.x = element_blank(),
                         axis.text.y = element_text(size = 6),
                         axis.title.y =  element_text(size = 7),
                         legend.text = element_text(size = 5), 
                         legend.title = element_text(size = 6),
                         strip.text = element_text(size = 6),
                         plot.margin=grid::unit(c(0,2,0,2), "mm"))
    }
      ggplot(.,aes(x= ifelse(lon > 180, lon - 360, lon), y  = lat, fill = value))+
      geom_raster() +
      geom_polygon(data = map_data('world'),
                   aes(x = long, y = lat, group = group),
                   fill = NA, color = 'gray70', size = .05) + 
      scale_fill_gradient2(low = cet_perp_unif[1], high = cet_perp_unif[length(cet_perp_unif)],
                           mid = 'white') + 
      scale_x_continuous(breaks = c(-180, -90, 0, 90)) +
      scale_y_continuous(breaks = c(-90, -45, 0, 45, 90)) +
      facet_grid(~season) +  
      coord_quickmap() + 
      labs(fill = .$variable_pc[1], x = 'Longitude', y = 'Latitude') + 
      theme_bw() + 
      theme_use}
  }) %>%
  plot_grid(plotlist = ., align = 'v', ncol = 1, rel_heights = c(1,rep(.75, 6),1))
# ggsave(paste0('src/eda/paper_plots/', params[['variable']], '_pc1_lat_lon.png'),
#        width = 6, height = 7.5)
ggsave(plot = save_plot, paste0('src/eda/paper_plots/', params[['variable']], '_pc1_lat_lon.png'),
       width = 4.5, height = 4.5)

pc_vals_df %>%
  filter(!is.na(lon), pc == 1, variable %in% c('PRECT', 'LWCF', 'U200')) %>%
  mutate(variable_pc = variable) %>%
  group_split(variable_pc) %>% 
  map({
    ~{    if (.$variable[1] == 'LWCF') {
      theme_use <-       theme(legend.key.size = unit(.3, "lines"),
                               legend.margin=margin(-10,0,-10,-10), 
                               axis.text.x = element_blank(),
                               axis.title.x =  element_blank(),
                               axis.ticks.x = element_blank(),
                               axis.text.y = element_text(size = 6),
                               axis.title.y =  element_text(size = 7),
                               legend.text = element_text(size = 5), 
                               legend.title = element_text(size = 6),
                               strip.text = element_text(size = 6),
                               plot.margin=grid::unit(c(0,2,0,2), "mm"))
    } else if (.$variable[1] != 'U200') {
      theme_use <- theme(legend.key.size = unit(.3, "lines"),
                         legend.margin=margin(-10,0,-10,-10), 
                         axis.text.x = element_blank(),
                         axis.title.x =  element_blank(),
                         axis.ticks.x = element_blank(),
                         axis.text.y = element_text(size = 6),
                         axis.title.y =  element_text(size = 7),
                         strip.background = element_blank(),
                         strip.text.x = element_blank(),
                         legend.text = element_text(size = 5), 
                         legend.title = element_text(size = 6),
                         strip.text = element_text(size = 6),
                         plot.margin=grid::unit(c(0,2,0,2), "mm"))
    } else {
      theme_use <- theme(legend.key.size = unit(.3, "lines"),
                         legend.margin=margin(-10,0,-10,-10), 
                         axis.text = element_text(size = 6),
                         axis.title =  element_text(size = 7),
                         strip.background = element_blank(),
                         strip.text.x = element_blank(),
                         axis.text.y = element_text(size = 6),
                         axis.title.y =  element_text(size = 7),
                         legend.text = element_text(size = 5), 
                         legend.title = element_text(size = 6),
                         strip.text = element_text(size = 6),
                         plot.margin=grid::unit(c(0,2,0,2), "mm"))
    }
      ggplot(.,aes(x= ifelse(lon > 180, lon - 360, lon), y  = lat, fill = value))+
        geom_raster() +
        geom_polygon(data = map_data('world'),
                     aes(x = long, y = lat, group = group),
                     fill = NA, color = 'gray70', size = .05) + 
        scale_fill_gradient2(low = cet_perp_unif[1], high = cet_perp_unif[length(cet_perp_unif)],
                             mid = 'white') + 
        scale_x_continuous(breaks = c(-180, -90, 0, 90)) +
        scale_y_continuous(breaks = c(-90, -45, 0, 45, 90)) +
        facet_grid(~season) +  
        coord_quickmap() + 
        labs(fill = .$variable_pc[1], x = 'Longitude', y = 'Latitude') + 
        theme_bw() + 
        theme_use}}
  ) %>%
  plot_grid(plotlist = ., align = 'v', ncol = 1, rel_heights = c(1,rep(.75, 1),1))
ggsave(paste0('src/eda/paper_plots/', params[['variable']], '_pc1_lat_lon_reduced.png'),
       width = 4, height = 2)

theme_set(theme_bw())
pc_vals_df %>%
  filter(!is.na(plev), pc ==1) %>%
  mutate(variable_pc = variable) %>%
  left_join(plevels_df) %>%
  left_join(lats_df) %>%
  group_split(variable_pc) %>% 
  map({
    ~{    if (.$variable[1] == 'RELHUM') {
      theme_use <-       theme(legend.key.size = unit(.3, "lines"),
                               legend.margin=margin(-10,0,-10,-10), 
                               axis.text.x = element_blank(),
                               axis.title.x =  element_blank(),
                               axis.ticks.x = element_blank(),
                               axis.text.y = element_text(size = 6),
                               axis.title.y =  element_text(size = 7),
                               legend.text = element_text(size = 5), 
                               legend.title = element_text(size = 6),
                               strip.text = element_text(size = 6),
                               plot.margin=grid::unit(c(0,2,1,2), "mm"))
    } else if (.$variable[1] != 'U') {
      theme_use <- theme(legend.key.size = unit(.3, "lines"),
                         legend.margin=margin(-10,0,-10,-10), 
                         axis.text.x = element_blank(),
                         axis.title.x =  element_blank(),
                         axis.ticks.x = element_blank(),
                         axis.text.y = element_text(size = 6),
                         axis.title.y =  element_text(size = 7),
                         strip.background = element_blank(),
                         strip.text.x = element_blank(),
                         legend.text = element_text(size = 5), 
                         legend.title = element_text(size = 6),
                         strip.text = element_text(size = 6),
                         plot.margin=grid::unit(c(1,2,1,2), "mm"))
    } else {
      theme_use <- theme(legend.key.size = unit(.3, "lines"),
                         legend.margin=margin(-10,0,-10,-10), 
                         axis.text = element_text(size = 6),
                         axis.title =  element_text(size = 7),
                         strip.background = element_blank(),
                         strip.text.x = element_blank(),
                         axis.text.y = element_text(size = 6),
                         axis.title.y =  element_text(size = 7),
                         legend.text = element_text(size = 5), 
                         legend.title = element_text(size = 6),
                         strip.text = element_text(size = 6),
                         plot.margin=grid::unit(c(1,2,0,2), "mm"))
    }
      ggplot(.,aes(
                   ymin = plev_lower/100, ymax = plev/100, xmin = lat_lower, xmax = lat_higher, fill = value, color = value))+
        geom_rect() +
        scale_fill_gradient2(low = cet_perp_unif[1], high = cet_perp_unif[length(cet_perp_unif)],
                             mid = 'white') + 
        scale_color_gradient2(low = cet_perp_unif[1], high = cet_perp_unif[length(cet_perp_unif)],
                             mid = 'white') + 
        scale_y_continuous(trans = 'reverse') +
        scale_x_continuous(breaks = c(-90, -45, 0, 45, 90)) +
        facet_grid(~season) +  
        labs(fill = .$variable_pc[1], color =  .$variable_pc[1], x = 'Latitude', y = 'Pressure\n(mbar)') + 
        theme_bw() + 
        theme_use}}
  ) %>%
  plot_grid(plotlist = ., align = 'v', ncol = 1, rel_heights = c(1,rep(.8, 1),1.05))
ggsave(paste0('src/eda/paper_plots/', params[['variable']], '_pc1_lat_plev.png'),
       width = 9 * (4/9), height = 7.5/2 * (4/9) * 1.2)

if ('RESTOM' %in% unique(pc_vals_df[['variable']])) {
  df_restom <- pc_vals_df %>%
    filter(variable == 'RESTOM')
  ggplot(data = df_restom,
         aes(x = pc , y = value)) + 
    geom_point() + 
    scale_x_continuous(breaks = 1:max(df_restom[['pc']])) +
    labs(x = 'Principal component',
         y = 'Value of PC for top of atmosphere energy balance') + 
    theme(axis.title = element_text(size = 9), legend.position = 'bottom')
  ggsave(paste0('src/eda/paper_plots/', params[['variable']], '_pc1_global.png'),
         width = 6, height = 7.5/2)
}

ggplot(data = data.frame(pc = 1:length(pc_prop_var),
                         prop_var = cumsum(pc_prop_var)), 
       aes(pc, prop_var)) + 
  geom_line()+
  geom_point()+
  scale_y_continuous(limits = c(0,1)) + 
  scale_x_continuous(breaks = 1:length(pc_prop_var)) +
  labs(x = 'Principal component', 
       y = 'Proportion of variance explained by those PCs')
ggsave(paste0('src/eda/paper_plots/', params[['variable']], '_pc_prop_var.png'),
       width = 6, height = 7.5/2)

## Plots of principal component scores of model and surrogate
par(mfrow  =  c(2,2))
plot(pc_scores_model[1,],  main = 'PC1 scores of model output', cex = .5,
     ylab = 'PC1 score')
points(pc_scores_surr[1,], col = 2, pch = 2, cex = .5)
legend('topright', title =  'Type',  legend = c('sim_output', 'surr_pred'), col = 1:2,  pch  = 1:2, pt.cex = c(.5, .5), cex = .4)

if(dim(pc_vals)[2] > 9) {
  plot(pc_scores_model[10,],  main = 'PC10 scores of model output', cex = .5,
       ylab = 'PC10 score')
  points(pc_scores_surr[10,], col = 2, pch = 2, cex = .5)
}

if(dim(pc_vals)[2] > 14) {
  plot(pc_scores_model[15,],  main = 'PC15 scores of model output', cex = .5,
       ylab = 'PC15 score')
  points(pc_scores_surr[15,], col = 2, pch = 2, cex = .5)
}


image.plot(cor(t(pc_scores_surr)), main = 'Correlation of PC scores on surrogate \n(is diagonal on model output)')
par(mfrow  =  c(1,1))

pc_score_values <- data.frame(sim_value = as.vector(pc_scores_model),
                              surr_value = as.vector(pc_scores_surr),
                              pc = rep(1:16, times = 250),
                              sim_run = rep(1:250, each = 16))

ggplot(data = pc_score_values, aes(x = sim_value, surr_value)) +
  geom_point(size = .3) + 
  geom_abline(intercept = 0, slope = 1) + 
  facet_wrap(~pc) + 
  labs(x = 'Normalized principal component\nscore value for E3SM output',
       y = 'Normalized principal component\nscore value for surrogate predictions')
ggsave(paste0('src/eda/paper_plots/', params[['variable']], '_pc_scatter.png'),
       width = 5, height = 4)

surr_pc_mse <- apply(pc_scores_model - pc_scores_surr, 1, 
                     function(x) mean(x^2))
mean_pc_mse <- apply(pc_scores_model - apply(pc_scores_model, 1, mean), 1, 
                     function(x) mean(x^2))

plot(1 - surr_pc_mse/mean_pc_mse,
     main = 'R-squared by Principal Component',
     xlab = 'Principal Component',
     ylab = 'R-squared', 
     ylim = c(0,1))

