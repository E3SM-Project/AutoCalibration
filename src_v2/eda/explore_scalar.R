
library(ncdf4)
pred_file <- nc_open('surrogate_models/pred_ALL_10yr_obs_scalar_24x48_16_SWCF_LWCF_PRECT_PSL_Z500_U200_U850_TREFHT_U_RELHUM_T_RESTOM_0.7_raw.nc')

pred <- ncvar_get(pred_file, 'values')
library(fields)
image.plot(pred)
obs_list <- list()
for (q in c('DJF', 'MAM', 'JJA', 'SON')) {
  file <- nc_open(paste0('data/lat_lon_24x48_', q, '_obs.nc'))
  for (v in c('SWCF', 'LWCF', 'PRECT', 'PSL', 'Z500', 'U200', 'U850', 'TREFHT')) {
    obs_list[[length(obs_list) + 1]] <- ncvar_get(file, v)
  }
  nc_close(file)
}
obs_values <- simplify2array(obs_list)

rmse_list <- list()
for (q in c('DJF', 'MAM', 'JJA', 'SON')) {
  file <- nc_open(paste0('data/lat_lon_10yr_24x48_', q, '.nc'))
  for (v in c('SWCF', 'LWCF', 'PRECT', 'PSL', 'Z500', 'U200', 'U850', 'TREFHT')) {
    obs_values_all <- array(obs_values[,,length(rmse_list)+1],
                            dim = c(dim(obs_values[,,length(rmse_list)+1]), 250))
    values <- ncvar_get(file, v)
    area <- ncvar_get(file, 'area')
    area_norm <- area/sum(area[,,1])
    rmse_list[[length(rmse_list)+1]] <- 
      sqrt(apply(area_norm * (values - obs_values_all)^2, 3, sum))
  }
  nc_close(file)
}
sim_values <- t(simplify2array(rmse_list))

misc_rmse_list <- list()
for (q in c('DJF', 'MAM', 'JJA', 'SON')) {
  file <- nc_open(paste0('data/lat_lon_10yr_24x48_', q, '_miscsim_global_RMSE_RESTOM.7.nc'))
  for (v in c('SWCF', 'LWCF', 'PRECT', 'PSL', 'Z500', 'U200', 'U850', 'TREFHT')) {
    obs_values_all <- array(obs_values[,,length(misc_rmse_list)+1],
                            dim = c(dim(obs_values[,,length(misc_rmse_list)+1]), 250))
    values <- ncvar_get(file, v)
    area <- ncvar_get(file, 'area')
    area_norm <- area/sum(area)
    misc_rmse_list[[length(misc_rmse_list)+1]] <- 
      sqrt(sum(area_norm * (values - obs_values[,,length(misc_rmse_list)+1])^2))
  }
  nc_close(file)
}
misc_rmse <- unlist(misc_rmse_list)

ctrl_rmse_list <- list()
for (q in c('DJF', 'MAM', 'JJA', 'SON')) {
  file <- nc_open(paste0('data/lat_lon_10yr_24x48_', q, '_ctrl.nc'))
  for (v in c('SWCF', 'LWCF', 'PRECT', 'PSL', 'Z500', 'U200', 'U850', 'TREFHT')) {
    obs_values_all <- array(obs_values[,,length(ctrl_rmse_list)+1],
                            dim = c(dim(obs_values[,,length(ctrl_rmse_list)+1]), 250))
    values <- ncvar_get(file, v)
    area <- ncvar_get(file, 'area')
    area_norm <- area/sum(area)
    ctrl_rmse_list[[length(ctrl_rmse_list)+1]] <- 
      sqrt(sum(area_norm * (values - obs_values[,,length(ctrl_rmse_list)+1])^2))
  }
  nc_close(file)
}
ctrl_rmse <- unlist(ctrl_rmse_list)

obs_list <- list()
for (q in c('DJF', 'MAM', 'JJA', 'SON')) {
  file <- nc_open(paste0('data/lat_plev_24x48_', q, '_obs.nc'))
  for (v in c('U', 'RELHUM', 'T')) {
    obs_list[[length(obs_list) + 1]] <- ncvar_get(file, v)
  }
  nc_close(file)
}
obs_values_plev <- simplify2array(obs_list)

rmse_list <- list()
for (q in c('DJF', 'MAM', 'JJA', 'SON')) {
  file <- nc_open(paste0('data/lat_plev_10yr_24x48_', q, '.nc'))
  for (v in c('U', 'RELHUM', 'T')) {
    obs_values_all <- array(obs_values_plev[,,length(rmse_list)+1],
                            dim = c(dim(obs_values_plev[,,length(rmse_list)+1]), 250))
    values <- ncvar_get(file, v)
    area <- ncvar_get(file, 'area')
    area_all <- values
    for (i in 1:250) {
      for (j in 1:37) {
        area_all[,j,i] <-area[,j]
      }
    }
    area_norm <- area_all/sum(area_all[,,1])
    rmse_list[[length(rmse_list)+1]] <- 
      sqrt(apply(area_norm * (values - obs_values_all)^2, 3, sum, na.rm = T))
  }
  nc_close(file)
}
sim_values_plev <- t(simplify2array(rmse_list))

misc_rmse_list <- list()
for (q in c('DJF', 'MAM', 'JJA', 'SON')) {
  file <- nc_open(paste0('data/lat_plev_10yr_24x48_', q, '_miscsim_global_RMSE_RESTOM.7.nc'))
  for (v in  c('U', 'RELHUM', 'T')) {
    obs_values_all <- array(obs_values_plev[,,length(misc_rmse_list)+1],
                            dim = c(dim(obs_values_plev[,,length(misc_rmse_list)+1]), 250))
    values <- ncvar_get(file, v)
    area <- ncvar_get(file, 'area')
    area_all <- values
    for (j in 1:37) {
      area_all[,j] <- area
    }
    area_norm <- area_all/sum(area_all)
    misc_rmse_list[[length(misc_rmse_list)+1]] <- 
      sqrt(sum(area_norm * (values - obs_values_plev[,,length(misc_rmse_list)+1])^2, na.rm = T))
  }
  nc_close(file)
}
misc_rmse_plev <- unlist(misc_rmse_list)

ctrl_rmse_list <- list()
for (q in c('DJF', 'MAM', 'JJA', 'SON')) {
  file <- nc_open(paste0('data/lat_plev_10yr_24x48_', q, '_ctrl.nc'))
  for (v in  c('U', 'RELHUM', 'T')) {
    obs_values_all <- array(obs_values_plev[,,length(ctrl_rmse_list)+1],
                            dim = c(dim(obs_values_plev[,,length(ctrl_rmse_list)+1]), 250))
    values <- ncvar_get(file, v)
    area <- ncvar_get(file, 'area')
    area_all <- values
    for (j in 1:37) {
      area_all[,j] <- area
    }
    area_norm <- area_all/sum(area_all)
    ctrl_rmse_list[[length(ctrl_rmse_list)+1]] <- 
      sqrt(sum(area_norm * (values - obs_values_plev[,,length(ctrl_rmse_list)+1])^2, na.rm = T))
  }
  nc_close(file)
}
ctrl_rmse_plev <- unlist(ctrl_rmse_list)

file <- nc_open(paste0('data/lat_lon_10yr_24x48_ANN.nc'))
FLNT <- ncvar_get(file, 'FLNT')
FSNT <- ncvar_get(file, 'FSNT')
area <- ncvar_get(file, 'area')
area_norm <- area/sum(area[,,1])
sim_values_restom <- apply(area_norm * (FSNT - FLNT),3, sum)
nc_close(file)

file <- nc_open(paste0('data/lat_lon_10yr_24x48_ANN_miscsim_global_RMSE_RESTOM.7.nc'))
FLNT <- ncvar_get(file, 'FLNT')
FSNT <- ncvar_get(file, 'FSNT')
area <- ncvar_get(file, 'area')
area_norm <- area/sum(area)
misc_restom <- sum(area_norm * (FSNT - FLNT))
nc_close(file)

file <- nc_open(paste0('data/lat_lon_10yr_24x48_ANN_ctrl.nc'))
FLNT <- ncvar_get(file, 'FLNT')
FSNT <- ncvar_get(file, 'FSNT')
area <- ncvar_get(file, 'area')
area_norm <- area/sum(area)
ctrl_restom <- sum(area_norm * (FSNT - FLNT))
nc_close(file)




sim_values <- rbind(sim_values, sim_values_plev)
misc_rmse <- c(misc_rmse, misc_rmse_plev)
ctrl_rmse <- c(ctrl_rmse, ctrl_rmse_plev)
#sim_values <- rbind(sim_values, sim_values_plev)
image.plot(sim_values)
image.plot(pred[1:32,])

cor_values <- rep(NA, nrow(sim_values))
for(i in 1:nrow(sim_values)) {
  cor_values[i] <- cor(pred[i, ], sim_values[i, ] )
}

plot(cor_values)
round(matrix(cor_values[1:32], ncol = 4),
      3)
round(matrix(cor_values[-c(1:32)], ncol = 4),
      3)
mean(cor_values)
library(dplyr)
rowMeans(rbind(matrix(cor_values[1:32], ncol = 4), 
      matrix(cor_values[-c(1:32)], ncol = 4))) %>%
  round(3)

plot(pred[1,], sim_values[1,],
     xlab = c())
abline(h = misc_rmse[1])
abline(v = misc_rmse[1])
plot(pred[2,], sim_values[2,])
abline(h = misc_rmse[2])
abline(v = misc_rmse[2])
plot(pred[3,], sim_values[3,])
abline(h = misc_rmse[3])
plot(pred[4,], sim_values[4,])
abline(h = misc_rmse[4])
plot(pred[5,], sim_values[5,])
abline(h = misc_rmse[5])
plot(pred[6,], sim_values[6,])
abline(h = misc_rmse[6])
plot(pred[7,], sim_values[7,])
abline(h = misc_rmse[7])
plot(pred[8,], sim_values[8,])
abline(h = misc_rmse[8])
plot(pred[9,], sim_values[9,])
abline(h = misc_rmse[9])

plot(pred[10,], sim_values[10,])
abline(h = misc_rmse[10])
plot(pred[11,], sim_values[11,])
abline(h = misc_rmse[11])
plot(pred[12,], sim_values[12,])
abline(h = misc_rmse[12])
plot(pred[13,], sim_values[13,])
abline(h = misc_rmse[13])
plot(pred[14,], sim_values[14,])
abline(h = misc_rmse[14])
plot(pred[15,], sim_values[15,])
abline(h = misc_rmse[15])
plot(pred[16,], sim_values[16,])
abline(h = misc_rmse[16])



prop_better <- rep(0, nrow(sim_values))
for (i in 1:nrow(sim_values)) {
  prop_better[i] <- mean(sim_values[i,] > misc_rmse[i])
}
prop_better

(rbind(matrix(prop_better[1:32], ncol = 4), 
               matrix(prop_better[-c(1:32)], ncol = 4))) %>%
  round(3) * 100

plot(prop_better)
plot(cor_values^2, prop_better)
plot(prop_better)

mean(cor_values^2)
plot(cor_values^2)


pred_at_default <- c(10.74300939, 6.8891859, 1.05362501, 282.58402658, 22.76485969, 3.71757487, 1.4467148,
                     1.21295924, 10.54683952, 5.53756204, 0.84202446, 287.17353447, 24.60787941, 2.92037953, 
                     1.30690357, 1.19161441, 14.69988442, 5.76357958, 1.25397973, 285.98774344, 25.83866875, 
                     3.12621151, 1.69091419, 1.08876049, 10.39522488, 5.60103809, 0.97486345, 278.61952404, 
                     23.27226276, 2.71035629, 1.3062318, 1.36547747, 6.01203675, 7.63555497, 2.79405035, 
                     4.07767527, 7.38801723, 2.09716811, 4.02607994, 8.13093724, 2.27649297, 3.87752062, 
                     6.85478172, 1.99002618, 0.19072637)

plot(pred_at_default)

sigma_values <- c( 39.87847139, 13.24022271, 2.06660146, 1018.87158784, 313.68086888, 13.29755089, 5.31828424,
                   19.30164889, 22.98395012, 11.45468033, 1.84592555, 973.54160158, 317.38944926, 11.19410894, 
                   5.10509384, 20.57518546, 30.89797033, 12.79811263, 2.0385041, 966.93116353, 325.54495177, 
                   14.38075791, 5.3478599, 20.45048237, 27.19443608, 11.00642831, 1.85145485, 1067.31151732, 
                   327.36687418, 11.99238599, 5.29065909, 18.96616604, 13.48351723, 31.29495594, 26.88022183, 
                   10.45897617, 31.05813758, 26.94492742, 17.6229091, 30.81113318, 29.19204056, 10.72786154, 
                   30.78687687, 27.58162491, 27.022487492943355)

percent_change_scalar <- c(13.5, 12.7, 5.2, -3.6, 14.8, 13.3, 15.5, 8.2,
                           10.2, 29.4, 4.9, 6.3, 15.2, 8.7, 4.0, -5.5, 
                           7.4, 27.6, -5.6, 4.3, 19.2, -1.9, -8.5, -3.6,
                           8.5, 21.8, 1.4, -4.2, 15.4, 19.5, 4.7, 0.3,
                           -8.0, 2.0, -5.2, -9.0, 2.5, -6.7, 14.1, 4.2, 8.6,
                           -11.5, 3.4, -3.2, NA)

percent_change_full <- c(5.1, 9.7, 9.5, 4.3, 4.0, 7.4, 5.7, -7.2,
                         -0.3, -1.3, 4.1, -6.9, -9.8, -12.8, -11.8, -10.0,
                         -6.2, 0.4, -0.3, -5.3, -7.1, -18.0, -16.1, -2.5,
                         2.0, 10.0, 11.8, -18.0, -15.0, -7.3, 0.7, -10.3, 
                         1.4, -1.7, -0.3, -10.6, 0.3, -3.3, 
                         -6.7, 1.9, 1.9, -10.8, 0.4, -4.0,NA)

plot(percent_change_full, percent_change_scalar,
     xlab = 'Full spatial fields: RMSE percent change compared to default',
     ylab = 'Scalar fields: RMSE percent change compared to default')
abline(a = 0, b = 1)


variables <- c(rep(c('SWCF', 'LWCF', 'PRECT', 'PSL', 'Z500', 'U200', 'U850', 'TREFHT'), times = 4),
               rep(c('U', 'RELHUM', 'T'), times = 4), 
               'RESTOM')
seasons <- c(rep(c('DJF', 'MAM', 'JJA', 'SON'), each = 8), 
             rep(c('DJF', 'MAM', 'JJA', 'SON'), each = 3),
             NA)

var_seas <- paste0(variables, seasons, collapse = '_')

plot(pred_at_default^2/sigma_values^2, 
     xlab = 'Field', ylab = 'Y^2/sigma^2')
plot(pred_at_default^2, sigma_values^2)
abline(a = 0, b = 1)
plot(sigma_values)

data.frame(variables, seasons, vals = pred_at_default^2/sigma_values^2,
           perc_change_scalar = percent_change_scalar,
           perc_change_full = percent_change_full, 
           better = percent_change_full > percent_change_scalar) 
library(dplyr)
data.frame(variables, seasons, 'MSE_def/sigma^2' = pred_at_default^2/sigma_values^2,
           perc_change_scalar = percent_change_scalar,
           perc_change_full = percent_change_full, 
           better = percent_change_full > percent_change_scalar,
           diff = percent_change_scalar - percent_change_full ) %>%
  filter(better) %>%
  select(-better)

data.frame(variables, seasons, vals = pred_at_default^2/sigma_values^2,
           perc_change_scalar = percent_change_scalar,
           perc_change_full = percent_change_full, 
           better = percent_change_full > percent_change_scalar) %>%
  group_by(variables) %>%
  summarize(mean(vals))

plot(pred_at_default^2/sigma_values^2, percent_change_scalar)
plot(pred_at_default^2/sigma_values^2, percent_change_scalar - percent_change_full)
cor(pred_at_default^2/sigma_values^2, percent_change_scalar - percent_change_full, use = 'complete')
plot(pred_at_default^2/sigma_values^2, percent_change_full)

