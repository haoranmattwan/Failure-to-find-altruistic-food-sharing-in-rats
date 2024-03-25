// saved as intake_model.stan
//
// This model is slightly more complicated, because it makes a within-subject comparison.
// To this end, we have two pieces of data per session, P and NP. Once again, we'll use
// the condID indexing trick to identify the condition to which each session belongs.
//
data{
	int<lower=1> S;         // Number of sessions
	int<lower=1> C;         // Number of conditions
	int<lower=1> condID[S]; // Condition ID
	int<lower=0> P[S];      // Number of pellets eaten per session
	int<lower=0> NP[S];     // Number of pellets left behind (or 'shared') per session
}
//
// Our statistical model requires specifying several parameters this time, one of which
// is a dispersion variable called "overdisp" that governs the overdispersion of our
// model. Like a standard deviation, this parameter *cannot* be negative, so it's really
// important to include a bound on the parameter. Otherwise, the HMC sampler will wander
// into negative territory and produce an error that will abort the whole process. We
// also have two condition-specific parameters. Here, "intercept" is going to govern the
// "global" rate of how many pellers get left behind. This approach treats "shared" and
// "left behind" as a distinction without a difference, which is to be interpreted 
// qualitatively given knowledge about the conditions. So, on average, we expect a rat in
// condition i to leave behind (intercept[i]) pellets and to consume
// (intercept[i]+self[i]) pellets. As a results, "self" measures the deviation from the
// global leave-behind rate regardless of the reason. However, as we'll see below, we
// shouldn't interpret these values directly, because we are once again working with
// transformed parameters. In fact, given the transform described below, it's accurate to
// say that rats leave behind exp(intercept[i]) pellets and eat exp(intercept[i]+self[i])
// pellets.
//
// To be clear, the choice to frame the analysis in terms of pellets
// left vs. pellets eaten is arbitrary; would could have made sharing the deviation, and
// the resulting model would yield the same inference with respect to each condition.
//
parameters{
	real intercept[C];
	real self[C];
	real<lower=0> overdisp;
}
//
// The negative binomial distribution is the likelihood function of choice in this model,
// which is effectively an overdispersed Poisson distribution. This means it's great for
// data that are discrete counts but that don't have well defined "trials" and are instead
// governed by an underlying "rate" with no well-defined upper limit. In practice, of
// course, the rats have physical constraints on how many pellets they can eat, but so
// long as their displayed rates don't go to complete satiation, this should be a good
// model.
//
// As with the binomial model for the pair_model code, this model relies on transformed
// parameters. Rather than being subject to a logit transform, our underlying "rate"
// parameter As such subjects will *actually* leave behind exp(intercept[i]) pellets, and
// consume exp(intercept[i]+self[i]) pellets. As such, even though we've coded self[i] as
// a difference, it works out to be a multiplicative effect, since exp(a)*exp(b)=exp(a+b).
// Put another way, exp(self[i]) is the factor by which a subject eats more pellets than
// they leave behind.
//
// This also helps to explain the priors. If our prior on intercept is normal(0,3), that
// corresponds to a prediction that the rate at which pellets are shares will cover a
// range of exp(0 +/- 3) for one standard deviation, i.e. anywhere from 1 pellet per 20
// sessions (exp(-3)) to 20 pellets per session (exp(3)). By the time we go out to two
// standard deviations, our upper bound is 400 pellets per session! So it's safe to say
// that the true rate is below that number, but it's also safe to say that estimates of,
// say, a million pellets per session should be ruled out, which this prior accomplishes.
//
// The choice of a prior for overdisp is probably the most arcane choice in this case.
// I've chosen an exponential distribution with a mean of 1 because this has a fairly
// strong regularizing effect, and because, counterintuitively, the *inverse* of this
// parameter governs the amount of overdispersion. As such, the more like a Poisson
// distribution these data are, the larger overdisp will need to get. As such, since these
// data appear overdispersed, I'm confident that I've covered the full range of possible
// values. When in doubt, though, it's worth changing this prior and seeing if it makes a
// difference. Don't take my or anyone else's word that a prior is "the right one," since
// a prior should reflect *your* intuition about how the parameter is distributed, and
// sometimes the only way to build those intuitions is to play around with the model to
// see the implications.
//
model{
	real muP[S];
	real muNP[S];
	intercept ~ normal(0,3);
	self ~ normal(0,3);
	overdisp ~ exponential(1);
	for ( s in 1:S ) {
		muP[s] = intercept[condID[s]] + self[condID[s]];
		muNP[s] = intercept[condID[s]];
	}
	P ~ neg_binomial_2_log(muP,overdisp);
	NP ~ neg_binomial_2_log(muNP,overdisp);
}
