df <- data.frame(nyr = c(5, 10, 5, 10, 5, 10, 5, 10, 5, 10), target = c('zonal_old', 'zonal_old', 'full', 'full',
                                                                        'scalarmean', 'scalarmean', 'scalar', 'scalar', 'zonal', 'zonal'), 
                 ice_sed_ai = c(1400.00, 1400.00, 1334.52, 1354.28, 388.11, 390.44,
                                350.00, 366.85, 1099.63, 838.73), 
                 clubb_c1 = c(4.05, 3.37, 2.36, 1.32, 5.00, 5.00, 
                              1.00, 1.00, 1.00, 1.00),
                 clubb_gamma_coef = c(0.100, 0.446, 0.100, 0.271, 0.100, 0.100,
                                      0.344, 0.295, .334, 0.323),
                 zmconv_tau = c(12158.61, 5458.19, 2490.86, 2274.36, 4395.19, 4480.07,
                                4352.29, 4366.47, 3112.41,3103.26), 
                 zmconv_dmpdz = c(-0.00010, -0.00054, -0.00046, -0.00052, -0.00014, -0.00014,
                                  -0.00010, -0.00017, -0.00033, -0.00034), 
                 test_r2 = c(0.475, 0.582, 0.372, 0.478, 0.819, 0.864, 0.603, 0.705, 0.477, 0.585),
                 prop_var_explained_16 = c(0.855, 0.909, 0.774, 0.866, NA, NA, NA, NA, 0.938, 0.963),
                 minutes = c(84 + 43/60, 83 + 35/60, 66 + 57/60, 70 + 23/60, 10 + 53/60, 
                             12, 7 + 42/60, 7 + 38/60, 65 + 25/60, 76 + 2/60))
library(GGally)
bounds <- data.frame(parameter = c('ice_sed_ai', 'clubb_c1', 'clubb_gamma_coef', 'zmconv_tau', 
                                   'zmconv_dmpdz'),
                     lower = c(350, 1.0, 0.1, 1800, -0.002),
                     upper = c(1400, 5.0, 0.5, 14400, -0.0001))
df <- df[df$target != 'zonal_old',]  %>%
  dplyr::mutate(target = factor(target, levels = c('full', 'zonal', 'scalar', 'scalarmean'))) %>%
  dplyr::arrange(target)
ggpairs(df, aes(color = as.factor(nyr), shape = target),
        columns = c('ice_sed_ai', 'clubb_c1', 'clubb_gamma_coef', 'zmconv_tau', 
                    'zmconv_dmpdz'), legend = c(2,1),
        upper = list(continuous = 'points'))+
  theme(legend.position = "bottom")

df %>%
  dplyr::select(nyr, target,test_r2, prop_var_explained_16, minutes) %>%
  dplyr::mutate(minutes =  round(minutes, 2))

df
