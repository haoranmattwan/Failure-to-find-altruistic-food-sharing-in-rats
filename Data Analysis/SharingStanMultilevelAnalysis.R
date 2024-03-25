#=================================================
#========Concurrent Food vs. Social Choice========
#=================================================

#====================
#====Front Matter====
#====================
#
library(rethinking)
library(readr)
library(ggplot2)
setwd("/Users/gregjensen/Projects/Academia/0 - In Preparation/**Sharing Data (Hackenberg Lab)/Stan Analysis")

Behav <- read_csv("Raw_data.csv")
Behav$Subject <- as.integer(as.factor(Behav$Rat))

rstan_options(auto_write = TRUE);
options(mc.cores = parallel::detectCores());
parallel:::setDefaultClusterOptions(setup_strategy = "sequential")

#===========================================
#====Stan Analysis of Food/Social Choice====
#===========================================
#

pair_machine_mlm_p <- stan_model(file="pair_multilevel_model.stan")
cyc <- 3000

dex <- (Behav$Condition!=5)
dat <- list(N=sum(dex),S=length(unique(Behav$Subject[dex])),subID=Behav$Subject[dex],C=length(unique(Behav$Condition[dex])),condID=Behav$Condition[dex],choice=Behav$Social[dex]+Behav$Food[dex],food=Behav$Food[dex])
dat$condID[dat$condID>5] <- dat$condID[dat$condID>5]-1
choice_output <- sampling(pair_machine_mlm_p, data=dat, iter=cyc*2, warmup=cyc, chains=4, cores=4, control=list(adapt_delta=0.99,max_treedepth=15))
precis(choice_output,depth=3,pars=c("gamma","beta"))
cq <- extract.samples(choice_output)





outputs <- matrix(0,24,5)
for (i in 1:6) {
  dm <- matrix(0,dim(cq$beta)[1])
  for (s in 1:3) {
    d <- inv_logit(cq$beta[,s,i])
    dm <- dm + d/3
    outputs[(s-1)*6 + i,1] <- mean(d)
    outputs[(s-1)*6 + i,2] <- mean(d) - quantile(d,c(.025))
    outputs[(s-1)*6 + i,3] <- quantile(d,c(.975)) - mean(d)
    outputs[(s-1)*6 + i,4:5] <- quantile(d,c(0.1,.9))
  }
  outputs[18+i,1] <- mean(dm)
  outputs[18+i,2] <- mean(dm) - quantile(dm,c(.025))
  outputs[18+i,3] <- quantile(dm,c(.975)) - mean(dm)
  outputs[18+i,4:5] <- quantile(dm,c(0.1,.9))
}

clip <- pipe("pbcopy", "w")                       
write.table(outputs, file=clip,sep="\t")                               
close(clip)



#===============================================
#====Stan Analysis of Response Rates By Type====
#===============================================
#
resp_rate_machine_mlm_p <- stan_model(file="resp_rate_multilevel_model.stan")
cyc <- 3000

dex <- rep(TRUE,length(Behav$Rat))
dat <- list(N=sum(dex),S=length(unique(Behav$Subject[dex])),subID=Behav$Subject[dex],C=length(unique(Behav$Condition[dex])),condID=Behav$Condition[dex],FC=Behav$Food[dex],SC=Behav$Social[dex])
resp_rate_output <- sampling(resp_rate_machine_mlm_p, data=dat, iter=cyc*2, warmup=cyc, chains=4, cores=4, control=list(adapt_delta=0.99,max_treedepth=15))
precis(resp_rate_output,depth=3,pars=c("gamma","beta"))
rq <- extract.samples(resp_rate_output)

outputs <- matrix(0,28,10)
for (i in 1:7) {
  dm <- matrix(0,dim(rq$beta)[1],2)
  for (s in 1:3) {
    d <- exp(rq$beta[,s,i])
    dm[,1] <- dm[,1] + d/3
    outputs[(s-1)*7 + i,1] <- mean(d)
    outputs[(s-1)*7 + i,2] <- mean(d) - quantile(d,c(.025))
    outputs[(s-1)*7 + i,3] <- quantile(d,c(.975)) - mean(d)
    outputs[(s-1)*7 + i,4:5] <- quantile(d,c(0.1,.9))
    d <- exp(rq$beta[,s,i+7])
    dm[,2] <- dm[,2] + d/3
    outputs[(s-1)*7 + i,6] <- mean(d)
    outputs[(s-1)*7 + i,7] <- mean(d) - quantile(d,c(.025))
    outputs[(s-1)*7 + i,8] <- quantile(d,c(.975)) - mean(d)
    outputs[(s-1)*7 + i,9:10] <- quantile(d,c(0.1,.9))
  }
  print(sd(dm[,1]))
  print(sd(dm[,2]))
  outputs[21+i,1] <- mean(dm[,1])
  outputs[21+i,2] <- mean(dm[,1]) - quantile(dm[,1],c(.025))
  outputs[21+i,3] <- quantile(dm[,1],c(.975)) - mean(dm[,1])
  outputs[21+i,4:5] <- quantile(dm[,1],c(0.1,.9))
  outputs[21+i,6] <- mean(dm[,2])
  outputs[21+i,7] <- mean(dm[,2]) - quantile(dm[,2],c(.025))
  outputs[21+i,8] <- quantile(dm[,2],c(.975)) - mean(dm[,2])
  outputs[21+i,9:10] <- quantile(dm[,2],c(0.1,.9))
}

clip <- pipe("pbcopy", "w")                       
write.table(outputs, file=clip,sep="\t")                               
close(clip)



#============================================
#====Stan Analysis of Food Intake By Type====
#============================================
#
intake_machine_mlm_p <- stan_model(file="intake_multilevel_model.stan")
cyc <- 3000

dex <- rep(TRUE,length(Behav$Rat))
dat <- list(N=sum(dex),S=length(unique(Behav$Subject[dex])),subID=Behav$Subject[dex],C=length(unique(Behav$Condition[dex])),condID=Behav$Condition[dex],CP=Behav$Food[dex]*Behav$FoodAmmt[dex],SP=Behav$Sharing[dex],LP=Behav$PelletsLeft[dex])
intake_output <- sampling(intake_machine_mlm_p, data=dat, iter=cyc*2, warmup=cyc, chains=4, cores=4, control=list(adapt_delta=0.99,max_treedepth=15))
precis(intake_output,depth=3,pars=c("gamma","beta"))
iq <- extract.samples(intake_output)

outputs <- matrix(0,28,15)
for (i in 1:7) {
  dm <- matrix(0,dim(iq$beta)[1],3)
  for (s in 1:3) {
    d <- exp(iq$beta[,s,i])
    dm[,1] <- dm[,1] + d/3
    outputs[(s-1)*7 + i,1] <- mean(d)
    outputs[(s-1)*7 + i,2] <- mean(d) - quantile(d,c(.025))
    outputs[(s-1)*7 + i,3] <- quantile(d,c(.975)) - mean(d)
    outputs[(s-1)*7 + i,4:5] <- quantile(d,c(0.1,.9))
    d <- exp(iq$beta[,s,i+7])
    dm[,2] <- dm[,2] + d/3
    outputs[(s-1)*7 + i,6] <- mean(d)
    outputs[(s-1)*7 + i,7] <- mean(d) - quantile(d,c(.025))
    outputs[(s-1)*7 + i,8] <- quantile(d,c(.975)) - mean(d)
    outputs[(s-1)*7 + i,9:10] <- quantile(d,c(0.1,.9))
    d <- exp(iq$beta[,s,i+14])
    dm[,3] <- dm[,3] + d/3
    outputs[(s-1)*7 + i,11] <- mean(d)
    outputs[(s-1)*7 + i,12] <- mean(d) - quantile(d,c(.025))
    outputs[(s-1)*7 + i,13] <- quantile(d,c(.975)) - mean(d)
    outputs[(s-1)*7 + i,14:15] <- quantile(d,c(0.1,.9))
  }
  print(sd(dm[,1]))
  print(sd(dm[,2]))
  print(sd(dm[,3]))
  outputs[21+i,1] <- mean(dm[,1])
  outputs[21+i,2] <- mean(dm[,1]) - quantile(dm[,1],c(.025))
  outputs[21+i,3] <- quantile(dm[,1],c(.975)) - mean(dm[,1])
  outputs[21+i,4:5] <- quantile(dm[,1],c(0.1,.9))
  outputs[21+i,6] <- mean(dm[,2])
  outputs[21+i,7] <- mean(dm[,2]) - quantile(dm[,2],c(.025))
  outputs[21+i,8] <- quantile(dm[,2],c(.975)) - mean(dm[,2])
  outputs[21+i,9:10] <- quantile(dm[,2],c(0.1,.9))
  outputs[21+i,11] <- mean(dm[,3])
  outputs[21+i,12] <- mean(dm[,3]) - quantile(dm[,3],c(.025))
  outputs[21+i,13] <- quantile(dm[,3],c(.975)) - mean(dm[,3])
  outputs[21+i,14:15] <- quantile(dm[,3],c(0.1,.9))
}

clip <- pipe("pbcopy", "w")                       
write.table(outputs, file=clip,sep="\t")                               
close(clip)


