library(stringr)
library(dplyr)
library(ggplot2)
library(viridis)

saved_files <- 'surrogate_models/nsim_test/'

param_files <- list.files(saved_files, pattern = '.txt')
df_list <- list()

process_parameters <- function(filename, folder) {
  test <- read.delim(paste0(folder, filename))[5:14,]
  names <- str_trim(substr(test, 2 , 34))
  values <- as.double(str_trim(substr(test, 38 , nchar(test) -2)))
  
  file_str <- substr(filename, nchar(filename) - 14, 
                     nchar(filename))
  
  i_rand <- as.integer(strsplit(file_str, '_')[[1]][3])
  n_train <- as.integer(strsplit(file_str, '_')[[1]][2])
  data.frame(names, values, i_rand, n_train)
}

for(i in 1:length(param_files)) {
  nyear <- sapply(strsplit(param_files[i], '_'), function(x) x[3])
  res <- sapply(strsplit(param_files[i], '_'), function(x) x[5])
  df_list[[i]] <- cbind(process_parameters(param_files[i], saved_files), type = paste(nyear, res, sep  = '_'))
}

df <- dplyr::bind_rows(df_list)
df_use <- dplyr::filter(df, !(names %in% c('fit_time', 'score_time', 'test_r2',
                                           'test_neg_median_absolute_error', 
                                           'test_neg_root_mean_squared_error')), n_train > 15)
df_use_wide <- tidyr::pivot_wider(df_use, names_from = 'names', 
                                  values_from = 'values') 
color_df <- data.frame(n_train = c(25, 50, 75, 100,
                                   125, 150, 175, 200, 225),
                       color = viridis(9))

df_use_wide2   <- dplyr::left_join(df_use_wide, color_df) %>%
  dplyr::arrange(n_train, i_rand)
for (i in 1:length(unique(df_use_wide2$type))) {
  type_use <- unique(df_use_wide2$type)[i]
  df_use_wide3 <- df_use_wide2[df_use_wide2$type == type_use,]
  par(xpd = FALSE)
  pairs(df_use_wide3[,c('ice_sed_ai', 'clubb_c1',  
                        'clubb_gamma_coef', 'zmconv_tau', 'zmconv_dmpdz')], 
        col = df_use_wide3$color, 
        pch = 7, 
        main = type_use,
        cex = .5, oma=c(3,3,6,14))
  par(xpd = TRUE)
  legend('topright', title =  'Type', legend = as.character(color_df$n_train),
         col = color_df$color, 
         pch = rep(7, nrow(color_df)),
         pt.cex = rep(.5, nrow(color_df)))
  cat('  \n')
}

# get parameter bounds
sim_file <- ncdf4::nc_open('data/lat_lon_10yr_180x360_JJA.nc')
bounds_df <- data.frame('variable' = sim_file$dim$x$vals,
                        'bnd' = t(ncdf4::ncvar_get(sim_file, 'lhs_bnds')))
bounds_df_t <- t(bounds_df[,-1])
colnames(bounds_df_t) <- bounds_df$variable
for (i in 1:length(unique(df_use_wide2$type))) {
  type_use <- unique(df_use_wide2$type)[i]
  df_use_wide3 <- df_use_wide2[df_use_wide2$type == type_use,]
  par(xpd = FALSE)
  pairs(rbind(df_use_wide3[,c('ice_sed_ai', 'clubb_c1',  
                              'clubb_gamma_coef', 'zmconv_tau', 'zmconv_dmpdz')],
              bounds_df_t), 
        col = c(df_use_wide3$color, rep('white', 2)), 
        pch = 7,
        main = type_use,
        cex = .5, oma=c(3,3,6,14))
  par(xpd = TRUE)
  legend('topright', title =  'Type', legend = as.character(color_df$n_train),
         col = color_df$color, 
         pch = rep(7, nrow(color_df)),
         pt.cex = rep(.5, nrow(color_df)))
  cat('  \n')
}

params_default <- c(500, 2.4, .12, 3600, -.0007)
ggplot(data = df_use_wide2, aes(x = n_train, y = ice_sed_ai, group = interaction(n_train, type), color = type)) + 
  geom_boxplot() + 
  scale_y_continuous(limits = as.data.frame(bounds_df_t)$ice_sed_ai) + 
  geom_hline(yintercept = params_default[1])+
  annotate(geom = 'text', label = '- v2 Default', x = 8, y = params_default[1],# +.000051,
           angle = 90, hjust = 0)+
  labs(title = 'ice_sed_ai', 
       color= 'Type',
       x  = 'Number of perturbed simulations used')+ 
  theme(legend.position = 'bottom')+
  guides(color = guide_legend(nrow = 2)) + 
  theme_bw() + 
  theme(legend.position = 'bottom')
ggsave('src/eda/ice_sed_ai.png', width = 7, height = 4.66)


ggplot(data = df_use_wide2, aes(x = n_train, y = clubb_c1, group = interaction(n_train, type), color = type)) + 
  geom_boxplot() + 
  scale_y_continuous(limits = as.data.frame(bounds_df_t)$clubb_c1) + 
  geom_hline(yintercept = params_default[2])+
  annotate(geom = 'text', label = '- v2 Default', x = 8, y = params_default[2],# +.000051,
           angle = 90, hjust = 0)+
  labs(title = 'clubb_c1', 
       color= 'Type',
       x  = 'Number of perturbed simulations used')+ 
  theme(legend.position = 'bottom')+
  guides(color = guide_legend(nrow = 2)) + 
  theme_bw() + 
  theme(legend.position = 'bottom')
ggsave('src/eda/clubb_c1.png', width = 7, height = 4.66)

ggplot(data = df_use_wide2, aes(x = n_train, y = clubb_gamma_coef, group = interaction(n_train, type), color = type)) + 
  geom_boxplot() + 
  scale_y_continuous(limits = as.data.frame(bounds_df_t)$clubb_gamma_coef) + 
  geom_hline(yintercept = params_default[3])+
  annotate(geom = 'text', label = '- v2 Default', x = 8, y = params_default[3],# +.000051,
           angle = 90, hjust = 0)+
  labs(title = 'clubb_gamma_coef', 
       color= 'Type',
       x  = 'Number of perturbed simulations used')+ 
  theme(legend.position = 'bottom')+
  guides(color = guide_legend(nrow = 2)) + 
  theme_bw() + 
  theme(legend.position = 'bottom')
ggsave('src/eda/clubb_gamma_coef.png', width = 7, height = 4.66)

library(patchwork)
a  <- ggplot(data = df_use_wide2, aes(x = n_train, y = zmconv_tau, group = interaction(n_train, type), color = type)) + 
  geom_boxplot() + 
  scale_y_continuous(limits = as.data.frame(bounds_df_t)$zmconv_tau) + 
  geom_hline(yintercept = params_default[4])+ 
  labs(title = 'zmconv_tau on regular scale')+ 
  theme(legend.position = 'bottom')+
  guides(color = guide_legend(nrow = 2))
b <- ggplot(data = df_use_wide2, aes(x = n_train, y = zmconv_tau, group = interaction(n_train, type), color = type)) + 
  geom_boxplot() + 
  scale_y_continuous(limits = as.data.frame(bounds_df_t)$zmconv_tau, trans = 'log',
                     breaks = c(2000, 5000, 10000, 15000)) + 
  geom_hline(yintercept = params_default[4]) + 
  labs(title = 'zmconv_tau on log scale') + 
  theme(legend.position = 'bottom') +
  guides(color = guide_legend(nrow = 2))
a+b

ggplot(data = df_use_wide2, aes(x = n_train, y = zmconv_tau, group = interaction(n_train, type), color = type)) + 
  geom_boxplot() + 
  scale_y_continuous(limits = as.data.frame(bounds_df_t)$zmconv_tau) + 
  geom_hline(yintercept = params_default[4])+ 
  annotate(geom = 'text', label = '- v2 Default', x = 8, y = params_default[4],
           angle = 90, hjust = 0)+
  labs(title = 'zmconv_tau', 
       color= 'Type',
       x  = 'Number of perturbed simulations used')+ 
  theme(legend.position = 'bottom')+
  guides(color = guide_legend(nrow = 2)) + 
  theme_bw() + 
  theme(legend.position = 'bottom')
ggsave('src/eda/zmconv_tau.png', width = 7, height = 4.66)

ggplot(data = df_use_wide2, aes(x = n_train, y = zmconv_dmpdz, group = interaction(n_train, type), color = type)) + 
  geom_boxplot() + 
  scale_y_continuous(limits = as.data.frame(bounds_df_t)$zmconv_dmpdz ) + 
  geom_hline(yintercept = params_default[5])



ggplot(data = df_use_wide2, aes(x = n_train, y = zmconv_dmpdz, group = interaction(n_train, type), color = type)) + 
  geom_boxplot() + 
  scale_y_continuous(limits = as.data.frame(bounds_df_t)$zmconv_dmpdz) + 
  geom_hline(yintercept = params_default[5])+ 
  annotate(geom = 'text', label = '- v2 Default', x = 8, y = params_default[5],# +.000051,
           angle = 90, hjust = 0)+
  labs(title = 'zmconv_dmpdz', 
       color= 'Type',
       x  = 'Number of perturbed simulations used')+ 
  theme(legend.position = 'bottom')+
  guides(color = guide_legend(nrow = 2)) + 
  theme_bw() + 
  theme(legend.position = 'bottom')
ggsave('src/eda/zmconv_dmpdz.png', width = 7, height = 4.66)
df <- dplyr::filter(df, n_train > 11)
ggplot(data = dplyr::filter(df, names == 'test_r2',
                            n_train > 30)  %>%
         left_join(data.frame('names' =  'test_r2', 'names_label' = 'R-squared on test data')),
       aes(x = n_train, y = values, color = type, group =  interaction(n_train, type))) + 
  geom_boxplot() +
  labs(x =  'Number of perturbed simulations used',
       y = 'R-squared value', 
       color = 'Type',
       title = 'Comparison of number of simulations',
       subtitle = '10yr vs 5yr averages and 24x48 vs 180x360 resolutions') + 
  theme_bw() + 
  theme(legend.position = 'bottom')
ggsave(filename = 'src/eda/r2_comparison.png', width = 7, height = 4.66)
