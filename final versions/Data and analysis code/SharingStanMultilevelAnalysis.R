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

Behav <- read_csv("Raw_data.csv")
Behav$Subject <- as.integer(as.factor(Behav$Rat))

rstan_options(auto_write = TRUE);
options(mc.cores = parallel::detectCores());
parallel:::setDefaultClusterOptions(setup_strategy = "sequential")

#===========================================
#====Stan Analysis of Food/Social Choice====
#===========================================
#
# Logistic regression model estimating rates of food choice (as opposed to its complement, social choice).
# Condition 5 is explicitly excluded because preference is undefined when one of the response options is
# unavailable.
#

pair_machine_mlm_p <- stan_model(file="pair_multilevel_model.stan")
cyc <- 3000

dex <- (Behav$Condition!=5)
dat <- list(N=sum(dex),S=length(unique(Behav$Subject[dex])),subID=Behav$Subject[dex],C=length(unique(Behav$Condition[dex])),condID=Behav$Condition[dex],choice=Behav$Social[dex]+Behav$Food[dex],food=Behav$Food[dex])
dat$condID[dat$condID>5] <- dat$condID[dat$condID>5]-1
choice_output <- sampling(pair_machine_mlm_p, data=dat, iter=cyc*2, warmup=cyc, chains=4, cores=4, control=list(adapt_delta=0.99,max_treedepth=15))
precis(choice_output,depth=3,pars=c("gamma","beta"))
cq <- extract.samples(choice_output)

#===============================================
#====Stan Analysis of Response Rates By Type====
#===============================================
#
# Negative binomial regression of the rate of food and social presses made in each condition. All subjects are
# presumed to share a common overdispersion parameter. Condition 5 is included, despite having no data for
# social choices; as such, its posterior estimates will simply reiterate the prior.
#

resp_rate_machine_mlm_p <- stan_model(file="resp_rate_multilevel_model.stan")
cyc <- 3000

dex <- rep(TRUE,length(Behav$Rat))
dat <- list(N=sum(dex),S=length(unique(Behav$Subject[dex])),subID=Behav$Subject[dex],C=length(unique(Behav$Condition[dex])),condID=Behav$Condition[dex],FC=Behav$Food[dex],SC=Behav$Social[dex])
resp_rate_output <- sampling(resp_rate_machine_mlm_p, data=dat, iter=cyc*2, warmup=cyc, chains=4, cores=4, control=list(adapt_delta=0.99,max_treedepth=15))
precis(resp_rate_output,depth=3,pars=c("gamma"))
rq <- extract.samples(resp_rate_output)

#============================================
#====Stan Analysis of Food Intake By Type====
#============================================
#
# Negative binomial regression of the number of pellets consumed, "shared," and left behind in each condition.
# All subjects are presumed to share a common overdispersion parameter. Note that "pellets left behind" is a
# caregory that is undefined, and has no data, for conditions 1, 2, 3, 4, and 6. As such, posterior estimates
# for those conditions merely reiterate the prior.

intake_machine_mlm_p <- stan_model(file="intake_multilevel_model.stan")
cyc <- 3000

dex <- rep(TRUE,length(Behav$Rat))
dat <- list(N=sum(dex),S=length(unique(Behav$Subject[dex])),subID=Behav$Subject[dex],C=length(unique(Behav$Condition[dex])),condID=Behav$Condition[dex],CP=Behav$Food[dex]*Behav$FoodAmmt[dex],SP=Behav$Sharing[dex],LP=Behav$PelletsLeft[dex])
intake_output <- sampling(intake_machine_mlm_p, data=dat, iter=cyc*2, warmup=cyc, chains=4, cores=4, control=list(adapt_delta=0.99,max_treedepth=15))
precis(intake_output,depth=3,pars=c("gamma","beta"))
iq <- extract.samples(intake_output)


