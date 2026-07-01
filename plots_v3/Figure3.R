library(ggplot2)
#library(dplyr)
library(reshape2)
ecs_weights <- read.csv("ecs_weights.csv")

#all seasons for each target field, RESTOM last for run H01
h001_weights_allseasons <- c(0.28463486, 0.44940442, 0.44362695, 0.23692335, 0.06452651, 0.24141523,
  0.21241179, 0.06664942, 0.4026099,  0.35348677, 0.40840895, 0.21760331,
  0.06389447, 0.2557827,  0.18929332, 0.06155023, 0.40621401, 0.38365648,
  0.51453263, 0.25569847, 0.0730542,  0.15238188, 0.23946893, 0.0614495,
  0.37257219, 0.41538382, 0.44667292, 0.24996075, 0.07207156, 0.18318005,
  0.20379563, 0.06324854, 0.44332203, 0.27166681, 0.13017859, 0.33657317,
  0.25597076, 0.08491052, 0.20775715, 0.29939104, 0.1009223,  0.34033721,
  0.24381293, 0.07840763, 0.02963332)

h001_weights <- c(h001_weights_allseasons[45],0,apply(matrix(h001_weights_allseasons[-45], nrow=4),2,mean))

colnames(ecs_weights) <- c("var","L1","L2", "L3","H1","H2","H3")
ecs_weights$H1 <- h001_weights
col_sums <- apply(ecs_weights[,-1],2,sum)
ecs_weights_scaled <- ecs_weights
for(i in 2:ncol(ecs_weights_scaled)){
  ecs_weights_scaled[,i] <- ecs_weights[,i]/col_sums[i-1]
}
weights_melted <- melt(ecs_weights_scaled, id.vars = "var",variable.name = "ens", value.name = "weight")

colors_border=c("brown1","brown3","darkred","cadetblue2","deepskyblue","deepskyblue4") #change to variations of blue and red
weights_melted$ens <- factor(weights_melted$ens, levels = c("H1","H2","H3","L1","L2","L3"))
weights_melted$weight <- as.numeric(weights_melted$weight)
weights_melted$var <- factor(weights_melted$var, levels = c("LWCF","T","PSL","TREFHT","Z500","U200","PRECT","SWCF","RELHUM","U850","U","RESTOM","dnet_cld_dir"))

labs <- c("LWCF","T","PSL","TREFHT","Z500","U200","PRECT","SWCF","RELHUM","U850","U","RESTOM",expression(lambda))

ggplot(weights_melted) + geom_line(aes(x=var, y=weight, group=ens, col=ens, lty=ens), alpha=0.7, lwd=1.1)+
  scale_color_manual(values=colors_border) +
  scale_linetype_manual(values= c(2,2,1,1,2,2)) +
  scale_x_discrete(labels= labs) +
  ylab(NULL) + xlab(NULL) + 
  theme_bw(base_size = 16) +
  theme(strip.background = element_blank(), strip.placement = "outside", 
        axis.text.x = element_text(angle = 45, vjust=0.5),
        legend.title = element_blank(), 
        legend.position="inside",
        legend.position.inside = c(0.1,0.75),
        legend.key.size = unit(0.5,"cm"),
        legend.key.width = unit(1,"cm"))
