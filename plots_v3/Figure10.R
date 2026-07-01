library(tidync)
library(dplyr)
library(ggplot2)
#library(reshape2)

#------------------------------------------------------------------------------#
#F2010

#setup
s="ANN"
ens = c("L001","H003")
#ens = ens[-which(ens %in% c("ctrl","H001","H002","H003","L001","L002","L003"))]

#f2010
data_path= "H003_rshp_w_obs_20260126.nc" #set data path
h003obs_latlon <- tidync(data_path) %>% hyper_tibble()
h003obs_latlon$lat <- as.numeric(h003obs_latlon$lat) #reformatting lat, lon, lev to numeric vectors
h003obs_latlon$lon<- as.numeric(h003obs_latlon$lon)
h003obs_latlev <- tidync(data_path) %>% activate("D3,D4,D2,D0,D1") %>% hyper_tibble()
h003obs_latlev$lat <- as.numeric(h003obs_latlev$lat) #reformatting lat, lon, lev to numeric vectors
h003obs_latlev$lev<- as.numeric(h003obs_latlev$lev)
vars = c(colnames(h003obs_latlon)[1:8], colnames(h003obs_latlev)[1:3])
vars=c(vars,"Net Cloud Forcing")
f2010data_latlon <- h003obs_latlon %>% filter(ens_idx %in% ens, product=="mod", time==s)
f2010data_latlev <- h003obs_latlev %>% filter(ens_idx %in% ens, product=="mod", time==s)

#f2010 ctrl
f2010ctrl_latlon = h003obs_latlon %>% filter(product=="mod", time==s, ens_idx=="ctrl")
f2010ctrl_latlev = h003obs_latlev %>% filter(product=="mod", time==s, ens_idx=="ctrl") %>% arrange(lat,lev)

rm(h003obs_latlon, h003obs_latlev)

#WCYCL20TR
l001_coupled_latlev <- tidync("v3alt.LR.lowECS001.historical_ANN_198501_201412_merged.nc") %>% hyper_tibble()
l001_coupled_latlon <- tidync("v3alt.LR.lowECS001.historical_ANN_198501_201412_merged.nc") %>% activate("D3,D1") %>% hyper_tibble()
l001_coupled_latlev[,'ens_idx'] = "L001"
l001_coupled_latlon[,'ens_idx'] = "L001"

h003_coupled_latlev <- tidync("v3alt.LR.highECS003.historical_ANN_198501_201412_merged.nc") %>% hyper_tibble()
h003_coupled_latlon <- tidync("v3alt.LR.highECS003.historical_ANN_198501_201412_merged.nc") %>% activate("D3,D1") %>% hyper_tibble()
h003_coupled_latlev[,'ens_idx'] = "H003"
h003_coupled_latlon[,'ens_idx'] = "H003"

ctrl_coupled_latlev <- tidync("v3.LR.historical_0101_ANN_198501_201412_merged.nc") %>% hyper_tibble()
ctrl_coupled_latlon <- tidync("v3.LR.historical_0101_ANN_198501_201412_merged.nc") %>% activate("D3,D1") %>% hyper_tibble()

coupled_latlon <- (rbind(l001_coupled_latlon, h003_coupled_latlon))[,-c(1:2,6,8,13)]
coupled_latlev <- rbind(l001_coupled_latlev, h003_coupled_latlev)

rm(l001_coupled_latlev, l001_coupled_latlon, h003_coupled_latlon, h003_coupled_latlev)

#compute bias by spatial point
#standardize values by ctrl
coupled_latlev_melted = merge(melt(coupled_latlev, id.vars = c("lat","lev","ens_idx")), melt(ctrl_coupled_latlev[,1:5], id.vars = c("lat","lev"), value.name = "value_obs"), all=TRUE)
coupled_latlon_melted = merge(melt(coupled_latlon, id.vars = c("lat","lon","ens_idx")), melt(ctrl_coupled_latlon[,c(3:5,7,9:12,14:15)], id.vars = c("lat","lon"), value.name = "value_obs"), all=TRUE)
coupleddata = merge(coupled_latlev_melted,coupled_latlon_melted, all=TRUE)
coupleddata[,'config'] = "WCYCL20TR"

f2010_latlev_melted = merge(melt(f2010data_latlev[,1:6], id.vars = c("lat","lev","ens_idx")), melt(f2010ctrl_latlev[,1:5], id.vars = c("lat","lev"), value.name = "value_obs"), all=TRUE)
f2010_latlon_melted = merge(melt(f2010data_latlon[,1:11], id.vars = c("lat","lon","ens_idx")), melt(f2010ctrl_latlon[,1:10], id.vars = c("lat","lon"), value.name = "value_obs"), all=TRUE)
f2010data = merge(f2010_latlev_melted,f2010_latlon_melted, all=TRUE)
f2010data[,'config'] = "F2010"

rm(coupled_latlev_melted, coupled_latlon_melted, f2010_latlev_melted, f2010_latlon_melted)

coupleddata[which(coupleddata$variable=="PRECT"),'value'] = coupleddata[which(coupleddata$variable=="PRECT"),'value']*8.64e+7
coupleddata[,'bias_wcycl20tr'] = (coupleddata[,'value'] - coupleddata[,'value_obs'])
#f2010data$value = as.numeric(f2010data$value)
f2010data[,'bias_f2010'] = (f2010data[,'value'] - f2010data[,'value_obs'])

alldata_bias <- merge(coupleddata[,c("lat","variable","ens_idx","lev","lon","bias_wcycl20tr")], f2010data[,c("lat","variable","ens_idx","lev","lon","bias_f2010")], all=TRUE)

colnames(coupleddata)[which(colnames(coupleddata)=="value")] = "value_wcycl20tr"
colnames(f2010data)[which(colnames(f2010data)=="value")] = "value_f2010"
alldata <- merge(coupleddata[,c("lat","variable","ens_idx","lev","lon","value_wcycl20tr")], f2010data[,c("lat","variable","ens_idx","lev","lon","value_f2010")], all=TRUE)

#------------------------------------------------------------------------------#
#Figure 10

alldata_bias$variable <- factor(alldata_bias$variable, levels = c("LWCF","T","PSL","TREFHT","Z500","U200","PRECT","SWCF","RELHUM","U850","U"))
alldata_bias[,'ens_idx'] = recode(alldata_bias$ens_idx,H003="H3", L001="L1")
biasplot <- ggplot(alldata_bias %>% filter(!is.na(ens_idx))) + 
  geom_point(aes(x=bias_f2010, y=bias_wcycl20tr, col=ens_idx)) + 
  #scale_y_continuous(labels = function(x) sprintf("%.3f", x)) +
  facet_wrap(~variable, scales="free") + 
  ylab("WCYCL20TR Standardized Bias") + xlab("F2010 Standardized Bias") +
  labs(col=NULL) +
  geom_abline(slope = 1, color = "darkgrey", linetype = "dashed") +
  theme_minimal(base_size = 14) +theme(legend.position = "bottom",legend.text=element_text(size=14)) 

#add correlation labels
bias_correlation <- alldata_bias %>% group_by(variable) %>% summarise(cor=cor(bias_f2010,bias_wcycl20tr, use="na.or.complete"))
bias_correlation[,"xx"] <- c(11,1.2,239,1.2,24,3,2,20,7,1.2,2.7)
bias_correlation[,"yy"] <- c(-6.5,-1,-160,-1.3,-18,-3.5,1,-20,-3.2,-2.2,-5)

biasplot + geom_label(data=bias_correlation,aes(x=xx, y=yy, label=paste0("r= ",round(cor,2))))
