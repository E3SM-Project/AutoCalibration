library(tidync)
library(dplyr)
library(ggplot2)

data_path= "H003_rshp_w_obs_20260126.nc" #set data path
h003obs_latlon <- tidync(data_path) %>% hyper_tibble()
h003obs_latlon$lat <- as.numeric(h003obs_latlon$lat) #reformatting lat, lon, lev to numeric vectors
h003obs_latlon$lon<- as.numeric(h003obs_latlon$lon)
h003obs_latlev <- tidync(data_path) %>% activate("D3,D4,D2,D0,D1") %>% hyper_tibble()
h003obs_latlev$lat <- as.numeric(h003obs_latlev$lat) #reformatting lat, lon, lev to numeric vectors
h003obs_latlev$lev<- as.numeric(h003obs_latlev$lev)
h003obs_area <- tidync(data_path) %>% activate("D5,D3") %>% hyper_tibble() 
h003obs_area$lat <- as.numeric(h003obs_area$lat)

h003obs_area_latlev <- expand.grid(lat=sort(unique(h003obs_latlev$lat)), lev=unique(h003obs_latlev$lev))
h003obs_area_latlev[,'area'] <- rep(as.vector((h003obs_area %>% arrange(lon))[1:24,"area"])[[1]], length(unique(h003obs_latlev$lev)))

rmse <- function(x,y, wvec){
  wvec2 = wvec*(y-x)^2
  rmse = sqrt(sum(wvec2, na.rm=TRUE))
  return(rmse)
}

rsq <- function(x,y, wvec){
  #wvec = cos(lat*pi/180)
  sstot = sum(wvec*(x-mean(x, na.rm=TRUE))^2,na.rm=TRUE)
  ssres = sum(wvec*(x-y)^2, na.rm=TRUE)
  
  rsq = 1-ssres/sstot
  return(rsq)
}

seasons = unique(h003obs_latlon$time)
vars = c(colnames(h003obs_latlon)[1:8], colnames(h003obs_latlev)[1:3])
ens = unique(h003obs_latlon$ens_idx)

s="ANN" #only compute for annual summaries, not by season
#--------------------------------------------------------------------------------#
#establish baseline metrics for ctrl vs obs
#--------------------------------------------------------------------------------#

baseline_table <- data.frame(season=rep(s,length(vars)), var=vars)

moddata_latlon = h003obs_latlon %>% filter(product=="obs", time==s, ens_idx=="ctrl")
surdata_latlon = h003obs_latlon %>% filter(product =="mod", time==s, ens_idx=="ctrl")
moddata_latlev = h003obs_latlev %>% filter(product=="obs", time==s, ens_idx=="ctrl") %>% arrange(lat,lev)
surdata_latlev = h003obs_latlev %>% filter(product =="mod", time==s, ens_idx=="ctrl")  %>% arrange(lat,lev)

#remove observations where E3SMv3 simulation has some missing data 
if(nrow(surdata_latlev)<nrow(moddata_latlev)){
  naidx_tmp <- sapply(1:nrow(moddata_latlev),function(x){
  tmp_lat = moddata_latlev[x,'lat']
  tmp_lev = moddata_latlev[x,'lev']
      
  idx_tmp <- intersect(which(surdata_latlev$lat==tmp_lat[[1]]),which(surdata_latlev$lev==tmp_lev[[1]]))
  return(ifelse(length(idx_tmp)>0,idx_tmp,NA))           
  })
    naidx_tmp <- which(is.na(naidx_tmp))
    moddata_latlev = moddata_latlev[-naidx_tmp,]
    wvec_latlev <- h003obs_area_latlev[-naidx_tmp,"area"]
}
  
#compute metrics for each target variable
for(v in vars){
    
  if(v %in% colnames(h003obs_latlon)){
      
    moddata1 = moddata_latlon[,v]
    surdata1 = surdata_latlon[,v]
  
    rtmp1 = rmse(moddata1[[1]],surdata1[[1]],h003obs_area$area)
    rtmp2 = rsq(moddata1[[1]],surdata1[[1]],h003obs_area$area)
    rtmp3 = cor(moddata1,surdata1, use="complete")
    rtmp4 = sum(h003obs_area$area*(surdata1-moddata1), na.rm=TRUE)/sum(h003obs_area$area*moddata1,na.rm=TRUE) #bias
      
  }else{
      
    moddata1 = moddata_latlev[,v]
    surdata1 = surdata_latlev[,v]
      
    rtmp1 = rmse(moddata1[[1]], surdata1[[1]],wvec_latlev)
    rtmp2 = rsq(moddata1[[1]], surdata1[[1]],wvec_latlev)
    rtmp3 = cor(moddata1,surdata1, use="complete")
    rtmp4 = sum(wvec_latlev*(surdata1-moddata1), na.rm=TRUE)/sum(wvec_latlev*moddata1,na.rm=TRUE) #bias
  }
    
  idx = intersect(which(baseline_table$season==s),which(baseline_table$var==v))
  baseline_table[idx,'rmse'] = rtmp1
  baseline_table[idx,'rsq'] = rtmp2
  baseline_table[idx,'cor'] = rtmp3
  baseline_table[idx,'bias'] = rtmp4
}

#--------------------------------------------------------------------------------#
#compute metrics for sim vs obs for all sims in E3SMv3 ensemble
#--------------------------------------------------------------------------------#

rmse_std_table <- expand.grid(vars, ens)
colnames(rmse_std_table)[1:2] <- c("var","ens")
rmse_std_table[,'season'] = s

wvec_latlon = h003obs_area$area

#identiyfing missing data and removing from area vector
datatmp <- h003obs_latlev %>% filter(time=="ANN",ens_idx=="ctrl", product=="mod")
naidx <- sapply(1:888,function(x){
  tmp_lat =h003obs_area_latlev[x,'lat']
  tmp_lev =h003obs_area_latlev[x,'lev']

  idx_tmp <- intersect(which(datatmp$lat==tmp_lat),which(datatmp$lev==tmp_lev))
  return(ifelse(length(idx_tmp)>0,idx_tmp,NA))
})
naidx <- which(is.na(naidx))
rm(datatmp)
wvec_latlev = (h003obs_area_latlev %>% arrange(lat,lev))[-naidx,"area"]


for(e in ens){
  moddata_latlon = h003obs_latlon %>% filter(product=="mod", time==s, ens_idx==e)
  surdata_latlon = h003obs_latlon %>% filter(product =="sur", time==s, ens_idx==e)
  moddata_latlev = h003obs_latlev %>% filter(product=="mod", time==s, ens_idx==e) %>% arrange(lat,lev)
  surdata_latlev = h003obs_latlev %>% filter(product =="sur", time==s, ens_idx==e) %>% arrange(lat,lev)
    
  for(v in vars){
    if(v %in% colnames(h003obs_latlon)){
    
      moddata1 = moddata_latlon[,v][[1]]
      surdata1 = surdata_latlon[,v][[1]]
      rtmp1 = rmse(moddata1,surdata1,wvec_latlon)
      rtmp2 = rsq(moddata1,surdata1,wvec_latlon)
      rtmp3 = cor(moddata1,surdata1, use="complete")
      rtmp4 = sum(wvec_latlon*(surdata1-moddata1), na.rm=TRUE)/sum(wvec_latlon*moddata1,na.rm=TRUE) #bias
        
    }else{
        
      moddata1 = moddata_latlev[,v][[1]]
      surdata1 = surdata_latlev[,v][[1]]
      rtmp1 = rmse(moddata1,surdata1,wvec_latlev)
      rtmp2 = rsq(moddata1,surdata1,wvec_latlev)
      rtmp3 = cor(moddata1,surdata1, use="complete")
      rtmp4 = sum(wvec_latlev*(surdata1-moddata1), na.rm=TRUE)/sum(wvec_latlev*moddata1,na.rm=TRUE) #bias 
    }
    
    ctrl_rmse = baseline_table[intersect(which(baseline_table$season==s),which(baseline_table$var==v)),'rmse']
      
    idx = intersect(intersect(which(rmse_std_table$season==s),which(rmse_std_table$var==v)),which(rmse_std_table$ens==e))
    rmse_std_table[idx,'rmse'] = rtmp1/ctrl_rmse
    rmse_std_table[idx,'rsq'] = rtmp2
    rmse_std_table[idx,'cor'] = rtmp3
    rmse_std_table[idx,'bias'] = rtmp4
    #rmse_std_table[idx,'baseline_cor'] = baseline_table[intersect(which(baseline_table$season==s),which(baseline_table$var==v)),'cor']
   
  }
}


#------------------------------------------------------#
#Figure A1
#------------------------------------------------------#

rmse_std_table$var = factor(rmse_std_table$var, levels = c("LWCF","T","PSL","TREFHT","Z500","U200","PRECT","SWCF","RELHUM","U850","U" ))

metrics_melted <- melt(rmse_std_table %>% filter(season == "ANN", !ens %in% c("ctrl","L001","L002","L003","H001","H002","H003")) %>% select(var,ens,rmse, cor, bias),id.vars = c("var","ens"), variable.name = "metric")
metrics_melted$metric <- factor(metrics_melted$metric, levels=c("bias","rmse","cor"))
ggplot(metrics_melted) + geom_boxplot(aes(x=var, y=value)) + 
  facet_wrap(~metric, scales = "free_y", nrow=3,
             strip.position = "left", 
             labeller = as_labeller(c(bias = "Standardized Global Mean Bias", rmse = "Standardized RMSE", cor="Spatial Correlation") ) ) +
  ylab(NULL) + xlab(NULL) + theme_bw(base_size = 14) +
  theme(strip.background = element_blank(), strip.placement = "outside", 
        axis.text.x = element_text(angle = 45, vjust=0.5))