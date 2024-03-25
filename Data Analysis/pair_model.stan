// saved as pair_model.stan
//
// The "data" block of a stan model is responsible for telling the
// model exactly what to expect from the incoming data. Beyond merely
// knowing what the names of the various variables are, you must also
// specify their type. In this case, all of the data are integers ("int").
// Finally, for data that are ranges of values, rather than single numbers,
// This tells Stan how large those ranges are. For example N[S] tells Stan
// that the variable named N is an array that has S different elements.
// 
// It is also useful to specify the bounds of each variable, although this
// is less crucial for the data than it is for bounded parameters. The main
// advantage of setting bounds on the data is that it can help you detect
// when you have included impossible values. For example, suppose you have
// coded your dataset such that cases where you have no data have been
// given the dummy code "-1". Since these are counts of events, and you can't
// 
data{
	int<lower=1> S;         // Number of sessions
	int<lower=1> C;         // Number of conditions
	int<lower=0> N[S];      // Number of choices per session
	int<lower=1> condID[S]; // Condition ID
	int<lower=0> F[S];      // Number of food choices (such that N[s]-F[s]=Social[s])
}
//
// The "parameters" block follows a similar syntax to the "data" block, but
// instead tells Stan about the unknown numerical quantities that it is going to
// need to fit. Here, we have a single parameter variable named "intercept", but
// following the syntax from the previous block, we can see that it is an array
// of length C. Therefore, this models actually has multiple parameters, one for
// each condition, given that there are C conditions in total. These parameters
// are "real" (i.e. numbers with arbitrary decimal precision) and are unbounded,
// so they can be positive or negative.
//
parameters{
	real intercept[C];
}
//
// The "model" block is where the model itself is fit; that is, this is where the
// parameters are connected to the data by way of a likelihood. This is also where
// the parameters are connected with a prior probability distribution that describes
// the values we think they should display prior to having seen the data. When linking
// numerical value to a probability distribution, the "~" operator is used instead of
// "=". "~" should be read as "is a random sample from".
//
// As I argued in our last call, the ideal approach is to use "weakly informative"
// priors. These should nominate all the potential values for the parameters that
// are plausible without too strong a preference among them, while also giving very
// low weight to parameter value that are highly implausible.
//
// In this particular model, we're using a binomial likelihood function that works
// in terms of log-odds units, here named "binomial_logit". The log-odds transformation
// is important, because if we wish to interpret the resulting parameter as a probability,
// we'll need to use the inv_logit transformation to do so. For example, if a subject
// has a 0.5 probability of making a choice, their parameter value in this model will
// be logit(0.5) = 0.0, and to recover the probability, we'll need to reverse that by
// using inv_logit(0.0) = 0.5.
//
// This also helps to explain the choice of prior. Here, we're saying that we think each
// of the parameters in the "intercept" array comes from a normal distribution with a
// mean of 0.0 and a standard deviation of 1.5. That might seem like a strange and
// arbitrary choice, but this approximately corresponds to the case in which probabilities
// from 0.0 to 1.0 are approximately uniform. If you have the rethinking package
// installed, you can see this for yourself in R using the following line of code:
//
// dens(inv_logit(rnorm(10000,mean=0,sd=1.5)))
//
// It's worth not only plotting this distribution, but plotting distributions with other
// parameter values and seeing what distributions they imply. For example, using sd=1.0
// creates a clear bias toward values close to 0.5, whereas sd=2.0 creates a bias toward
// the extreme values and away from 0.5. More broadly, this strategy of using R to
// generate random samples in order to see what your prior implies is a great way to
// make sure that your prior corresponds to the kinds of parameter values that you think
// it should.
//
// The beating heart of this model, however, is the line that connect F (the actual food
// choices of subjects) to the binomial likelihood. This line says that each data points
// in F is the product of a binomial distribution where N is the number of attempts and
// inv_logit(mu) is the probability of success. So that's "mu"? Well, this is a little
// sneaky: "mu" is defined by the code above. It's an array that has the same length
// as F (i.e. S, the count of how many sessions there are in this dataset), and for each
// of its S cases, the value of mu is given the value of one of the parameters in the
// intercept array. Which one, well that's governed by an indexing variable called
//  "condID," which consists of integer values from 1 to C. Since F and my are of the same
// length, Stan knows to match these up case by case, so for any given session's data
// F[i], its probability of a good choice is by inv_logit(mu[i]), and mu got that value
// by looking up which condition case i belonged to using conID[i]. The result is a very
// compact syntax that's pretty easy to deal with once you're used to it.
//
model{
	real mu[S];
	intercept ~ normal(0,1.5);
	for ( s in 1:S ) {
		mu[s] = intercept[condID[s]];
	}
	F ~ binomial_logit(N,mu);
}
