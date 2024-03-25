// saved as pair_multilevel_model.stan
//
data{
	int<lower=1> N;         // Number of observations
	int<lower=1> S;         // Number of subjects
	int<lower=1> subID[N];  // Subject IDs
	int<lower=1> C;         // Number of conditions
	int<lower=1> condID[N]; // Condition ID
	int<lower=0> choice[N]; // Number of choices per session
	int<lower=0> food[N];   // Number of food choices (such that choose[s]-food[s]=social[s])
}
//
transformed data {
	vector[S] u;
	for (s in 1:S) {
		u[s] = 1;
	}
}
//
parameters{
	matrix[C, S] z;                   // beta proxy
	cholesky_factor_corr[C] L_Omega;  // prior correlation
	vector<lower=0>[C] tau;           // prior scale
	row_vector[C] gamma;              // population means
}
//
transformed parameters{
	matrix[S,C] beta;                 // individual coefficients for each subject in each condition
	beta = u * gamma + (diag_pre_multiply(tau,L_Omega) * z)';
}
//
model{
	real mu[N];

	to_vector(z) ~ normal(0,1);
	L_Omega ~ lkj_corr_cholesky(2);
	tau ~ exponential(1.5);
	to_vector(gamma) ~ normal(0,1.5);

	for ( n in 1:N ) {
		mu[n] = beta[subID[n],condID[n]];
	}
	food ~ binomial_logit(choice,mu);
}
