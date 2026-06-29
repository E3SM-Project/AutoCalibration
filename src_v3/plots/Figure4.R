library(tidync)
library(dplyr)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(fmsb)

#h003 <- tidync("H003.nc") %>% hyper_tibble()
h003obs_params <- tidync("H003_rshp_w_obs.nc") %>% activate("D6,D5") %>% hyper_tibble()
params = unique(h003obs_params$input_param)

#reformat data
colnames(h003obs_params)[1] <- "value"
param_data <- dcast(h003obs_params, ens_idx ~ input_param)
ens_idx = param_data$ens_idx

#add max and min to top of dataframe
mins = apply(param_data[,-1], 2, min)
maxes = apply(param_data[,-1], 2 , max)
param_data = rbind(maxes,mins,param_data[,-1])

colnames(param_data) = c('clubb_c1', 'clubb_gamma_coef', 'zmconv_tau', 'zmconv_dmpdz', 'zmconv_micro_dcs',
                                 'zmconv_auto_fac','zmconv_accr_fac','zmconv_ke','nucleate_ice_subgrid', 'p3_nc_autocon_expon',
                                 'p3_qc_accret_expon','cldfrc_dp1', 'p3_embryonic_rain_size','p3_mincdnc' )

# Prepare color
colors_border=c("brown1","brown3","darkred","cadetblue2","deepskyblue","deepskyblue4") #change to variations of blue and red
#colors_in=c( rgb(0.2,0.5,0.5,0.4), rgb(0.8,0.2,0.5,0.4)  )
plty_custom = c(2,2,1,1,2,2)
#op <- par(mar =rep(2,4)) 

# Custom the radarChart !
png(file="spiderplot_params.png",width=650,height=550)
radarchart( param_data[c(1,2,which(ens_idx %in% c("L001","L002","L003","H001","H002","H003"))+2),-1], 
            axistype=1,
            
            #custom polygon
            pcol=colors_border , #pfcol=colors_in , 
            #plwd=c(2,1,1,1,1,2), 
            plty=plty_custom  ,
            
            #custom the grid
            cglcol="grey", cglty=1, axislabcol="grey", #caxislabels=seq(0,20,5), 
  
            cglwd=1.1,
            
            #custom labels
            #vlabels=c('clubb_c1', 'clubb_gamma_coef', 'zmconv_tau', 'zmconv_dmpdz', 'zmconv_micro_dcs',
            #          'zmconv_auto_fac','zmconv_accr_fac','zmconv_ke','nucleate_ice_subgrid', 'p3_nc_autocon_expon',
            #          'p3_qc_accret_expon','cldfrc_dp1', 'p3_embryonic_rain_size','p3_mincdnc' ),
            #paxislabels = c(10, 10, 10, 10),
            vlcex=1
)

# Legend
legend(x=-1.75, y=-0.75, legend = c("H1","H2","H3","L1","L2","L3"), bty = "n", pch=20 , col=colors_border, lty=plty_custom, text.col = "black", cex=0.9, pt.cex=1.6)
#par(op)
dev.off()