setwd("~/SIFI/");
library("R.matlab");
library("quantreg");

YpartV = readMat("YpartV.mat");
Xall = readMat("Xall.mat");Xall = Xall$Xall;
pv = readMat("pv.mat");pv=pv$pv;



all = matrix(0, ncol = 2*length(pv), nrow = dim(Xall)[2]) 
all.resid = matrix(0, ncol = length(pv), nrow = dim(Xall)[1]) 


for(i in c(1:length(pv)))
{
  qr.sifi = rq(YpartV$YpartV ~ Xall[,-1], pv[i]);
  Z = summary(qr.sifi);
  all[, (i-1)*2+ c(1:2)] = Z$coefficients[, 1:2];
  #all.resid[, i] = Z$residuals;
}
  
write.table(all, "rq_result.txt", col.names = F, row.names = F, sep ="\t");

