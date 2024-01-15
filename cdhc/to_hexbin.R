#!/usr/bin/Rscript

library(hexbin)
data <- read.table('xrs/hex-temp.csv',sep=',',header=TRUE)
hbin<-hexbin(x=data$x,y=data$y,xbins=100,shape=diff(range(data$x))/diff(range(data$y)))
write.table(data.frame(hcell2xy(hbin),slot(hbin,"count")),row.name=F,file="xrs/hex-temp.csv")
