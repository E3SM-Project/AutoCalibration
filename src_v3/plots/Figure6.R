library(tidync)
library(dplyr)
library(ggplot2)
library(reshape2)

h003obs_latlon <- tidync("H003_rshp_w_obs.nc") %>% hyper_tibble()
h003obs_latlev <- tidync("H003_rshp_w_obs.nc") %>% activate("D2,D4,D5,D0,D1") %>% hyper_tibble()
h003obs_area <- tidync("H003_rshp_w_obs.nc") %>% activate("D3,D2") %>% hyper_tibble() 

h003obs_latlev$lat = as.numeric(h003obs_latlev$lat)
h003obs_latlev$lev= as.numeric(h003obs_latlev$lev)
h003obs_area_latlev <- expand.grid(lat=sort(unique(h003obs_latlev$lat)), lev=unique(h003obs_latlev$lev))
h003obs_area_latlev[,'area'] <- rep(as.vector((h003obs_area %>% arrange(lon))[1:24,"area"])[[1]], length(unique(h003obs_latlev$lev)))

datatmp <- h003obs_latlev %>% filter(time=="ANN",ens_idx=="ctrl", product=="mod")
naidx <- sapply(1:888,function(x){
  tmp_lat =h003obs_area_latlev[x,'lat']
  tmp_lev =h003obs_area_latlev[x,'lev']
  
  idx_tmp <- intersect(which(datatmp$lat==tmp_lat),which(datatmp$lev==tmp_lev))
  return(ifelse(length(idx_tmp)>0,idx_tmp,NA))           
})
naidx <- which(is.na(naidx))
rm(datatmp)

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
vars=c(vars,"Net Cloud Forcing")
ens = unique(h003obs_latlon$ens_idx)


baseline_table <- expand.grid(seasons,vars)
colnames(baseline_table)[1:2] <- c("season","var")

s="ANN"
#for(s in seasons){
  
  moddata_latlon = h003obs_latlon %>% filter(product=="obs", time==s, ens_idx=="ctrl")
  surdata_latlon = h003obs_latlon %>% filter(product =="mod", time==s, ens_idx=="ctrl")
  moddata_latlev = h003obs_latlev %>% filter(product=="obs", time==s, ens_idx=="ctrl") %>% arrange(lat,lev)
  surdata_latlev = h003obs_latlev %>% filter(product =="mod", time==s, ens_idx=="ctrl")  %>% arrange(lat,lev)

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
  
  for(v in vars){
    
    if(v=="Net Cloud Forcing"){
      moddata1 = moddata_latlon[,"LWCF"][[1]] + moddata_latlon[,"SWCF"][[1]]
      surdata1 = surdata_latlon[,"LWCF"][[1]] + surdata_latlon[,"SWCF"][[1]] #obs
      rtmp1 = rmse(moddata1,surdata1,h003obs_area$area)
      rtmp2 = rsq(moddata1,surdata1,h003obs_area$area)
      rtmp3 = cor(moddata1,surdata1, use="complete")
      rtmp4 = sum(h003obs_area$area*(moddata1-surdata1), na.rm=TRUE)/sum(h003obs_area$area*surdata1,na.rm=TRUE) #bias as fraction of observed global mean
    }else{
      
      if(v %in% colnames(h003obs_latlon)){
      
        moddata1 = moddata_latlon[,v]
        surdata1 = surdata_latlon[,v]
      #m = mean(surdata1[[1]],na.rm=TRUE)
      #sd = sd(surdata1[[1]],na.rm=TRUE)
      
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
    }
    
    idx = intersect(which(baseline_table$season==s),which(baseline_table$var==v))
    baseline_table[idx,'rmse'] = rtmp1
    baseline_table[idx,'rsq'] = rtmp2
    baseline_table[idx,'cor'] = rtmp3
    baseline_table[idx,'bias'] = rtmp4
  }
#}

rmse_std_table <- expand.grid(vars, ens)
colnames(rmse_std_table)[1:3] <- c("var","ens")
rmse_std_table[,'season'] = "ANN"

wvec_latlon = h003obs_area$area
wvec_latlev = (h003obs_area_latlev %>% arrange(lat,lev))[-naidx,"area"]

s="ANN"
  surdata_latlon = h003obs_latlon %>% filter(product =="obs", time==s, ens_idx=="ctrl")
  surdata_latlev = h003obs_latlev %>% filter(product =="obs", time==s, ens_idx=="ctrl") %>% arrange(lat,lev)
  
  for(e in ens){
    moddata_latlon = h003obs_latlon %>% filter(product=="mod", time==s, ens_idx==e)
    moddata_latlev = h003obs_latlev %>% filter(product=="mod", time==s, ens_idx==e) %>% arrange(lat,lev)
    
    if(nrow(surdata_latlev)>nrow(moddata_latlev)){
      naidx_tmp <- sapply(1:nrow(surdata_latlev),function(x){
        tmp_lat = surdata_latlev[x,'lat']
        tmp_lev = surdata_latlev[x,'lev']
        
        idx_tmp <- intersect(which(moddata_latlev$lat==tmp_lat[[1]]),which(moddata_latlev$lev==tmp_lev[[1]]))
        return(ifelse(length(idx_tmp)>0,idx_tmp,NA))           
      })
      naidx_tmp <- which(is.na(naidx_tmp))
      surdata_latlev = surdata_latlev[-naidx_tmp,]
      wvec_latlev <- h003obs_area_latlev[-naidx_tmp,"area"]
      
    }
    
    # lat = moddata$lat
    for(v in vars){
      
      if(v=="Net Cloud Forcing"){
        moddata1 = moddata_latlon[,"LWCF"][[1]] + moddata_latlon[,"SWCF"][[1]]
        surdata1 = surdata_latlon[,"LWCF"][[1]] + surdata_latlon[,"SWCF"][[1]] #obs
        rtmp1 = rmse(moddata1,surdata1,wvec_latlon)
        rtmp2 = rsq(moddata1,surdata1,wvec_latlon)
        rtmp3 = cor(moddata1,surdata1, use="complete")
        rtmp4 = sum(wvec_latlon*(moddata1-surdata1), na.rm=TRUE)/sum(wvec_latlon*surdata1,na.rm=TRUE) #bias as fraction of observed global mean
      }else{
        
      
        if(v %in% colnames(h003obs_latlon)){
        
          moddata1 = moddata_latlon[,v][[1]]
          surdata1 = surdata_latlon[,v][[1]] #obs
          rtmp1 = rmse(moddata1,surdata1,wvec_latlon)
          rtmp2 = rsq(moddata1,surdata1,wvec_latlon)
          rtmp3 = cor(moddata1,surdata1, use="complete")
          rtmp4 = sum(wvec_latlon*(moddata1-surdata1), na.rm=TRUE)/sum(wvec_latlon*surdata1,na.rm=TRUE) #bias as fraction of observed global mean
        
        }else{
        
          moddata1 = moddata_latlev[,v][[1]]
          surdata1 = surdata_latlev[,v][[1]]
          rtmp1 = rmse(moddata1,surdata1,wvec_latlev)
          rtmp2 = rsq(moddata1,surdata1,wvec_latlev)
          rtmp3 = cor(moddata1,surdata1, use="complete")
          rtmp4 = sum(wvec_latlev*(moddata1-surdata1), na.rm=TRUE)/sum(wvec_latlev*surdata1,na.rm=TRUE) #bias 
        }
      } #ncf
      
      ctrl_rmse = baseline_table[intersect(which(baseline_table$season==s),which(baseline_table$var==v)),'rmse']
      
      idx = intersect(intersect(which(rmse_std_table$season==s),which(rmse_std_table$var==v)),which(rmse_std_table$ens==e))
      rmse_std_table[idx,'rmse'] = rtmp1/ctrl_rmse
      rmse_std_table[idx,'rsq'] = rtmp2
      rmse_std_table[idx,'cor'] = rtmp3
      rmse_std_table[idx,'bias'] = rtmp4
      rmse_std_table[idx,'baseline_cor'] = baseline_table[intersect(which(baseline_table$season==s),which(baseline_table$var==v)),'cor']
      
    } #vars
    print(ens)
  } #ens
  

metrics_melted <- melt(rmse_std_table %>% filter(season == "ANN", !ens %in% c("ctrl","L001","L002","L003","H001","H002","H003")) %>% select(var,ens,rmse, cor, bias),id.vars = c("var","ens"), variable.name = "metric")
metrics_melted$metric <- factor(metrics_melted$metric, levels=c("bias","rmse","cor"))
metrics_summary <- metrics_melted %>% group_by(var,metric) %>% summarise(ymean = mean(value), ymin = min(value), ymax=max(value))
metrics_summary$var <- factor(metrics_summary$var, levels = c("LWCF","T","PSL","TREFHT","Z500","U200","PRECT","SWCF","RELHUM","U850","U","Net Cloud Forcing"))

rmse_optim <- rmse_std_table %>% filter(season == "ANN", ens %in% c("ctrl","L001","L002","L003","H001","H002","H003")) %>% select(var,ens,bias,rmse, cor)
rmse_optim <- melt(rmse_optim,variable.name = "metric", id.vars = c("var","ens"))
rmse_optim$ens <- factor(rmse_optim$ens, levels = c("H001","H002","H003","L001","L002","L003","ctrl"))
rmse_optim[,'ens'] = recode(rmse_optim$ens,H001="H1",H002="H2",H003="H3", L001="L1",L002="L2",L003="L3")
colors_border=c("brown1","brown3","darkred","cadetblue2","deepskyblue","deepskyblue4","black") #change to variations of blue and red

rmseplot <- ggplot(data=metrics_summary %>% filter(var!="Net Cloud Forcing"), aes(group=1)) + 
  geom_line(aes(x=var,y=ymin),col="lightgrey") + 
  geom_line(aes(x=var,y=ymax),col="lightgrey") +
  geom_ribbon( aes(x = var,ymin = ymin,ymax = ymax), alpha=0.25) +
  facet_wrap(~metric, scales = "free_y", nrow=3,
             strip.position = "left", 
             labeller = as_labeller(c(bias = "Standardized Global Mean Bias", rmse = "Standardized RMSE", cor="Spatial Correlation") ) ) +
  ylab(NULL) + xlab(NULL) + 
  theme_bw(base_size = 14) +
  theme(strip.background = element_blank(), strip.placement = "outside", 
        axis.text.x = element_text(angle = 45, vjust=0.5))

rmseplot + geom_line(data=rmse_optim %>% filter(var!="Net Cloud Forcing"),
                     aes(x=var,y=value,col=ens, lty=ens, group = ens), lwd=1.1, alpha=0.7) +
          scale_color_manual(values=colors_border) +
          scale_linetype_manual(values= c(2,2,1,1,2,2,1)) +
          theme(legend.title = element_blank(), 
                legend.key.width = unit(1.5,"cm"),
                legend.position = "bottom")

