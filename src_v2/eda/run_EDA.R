# knit report

param_list_to_str <- function(param_list) {
  paste0(param_list, collapse = '_')
}

params_use <- list(nyears = '10yr', resolution = '24x48', n_components = 16, 
                   variable = 'SWCF_LWCF_PRECT_PSL_Z500_U200_U850_TREFHT_U_RELHUM_T_RESTOM_0.7', 
                   season = 'ALL', subtract = 'raw')
rmarkdown::render('src/eda/EDA_general.Rmd',     output_file = paste0('EDA_', param_list_to_str(params_use),'.pdf'),
                  knit_root_dir = '../../',          params = params_use)

