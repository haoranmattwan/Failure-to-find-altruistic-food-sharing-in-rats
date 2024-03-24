// saved as intake_multilevel_model.stan
//
data{
	int<lower=1> N;         // Number of observations
	int<lower=1> S;         // Number of subjects
	int<lower=1> subID[N];  // Subject IDs
	int<lower=1> C;         // Number of conditions
	int<lower=1> condID[N]; // Condition ID
	int<lower=0> CP[N];     // Number of pellets consumed per session
	int LP[N];     // Number of pellets left behind per session
	int SP[N];     // Number of pellets "shared" per session
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
	matrix[3*C, S] z;                   // beta proxy
	cholesky_factor_corr[3*C] L_Omega;  // prior correlation
	vector<lower=0>[3*C] tau;           // prior scale
	row_vector[3*C] gamma;              // population means
	real<lower=0> overdisp;
}
//
transformed parameters{
	matrix[S,3*C] beta;                 // individual coefficients for each subject in each condition
	beta = u * gamma + (diag_pre_multiply(tau,L_Omega) * z)';
}
//
model{
	to_vector(z) ~ normal(0,1);
	L_Omega ~ lkj_corr_cholesky(2);
	tau ~ exponential(1.5);
	to_vector(gamma) ~ normal(0,1.5);
	overdisp ~ exponential(1);

	for ( n in 1:N ) {
		CP[n] ~ neg_binomial_2_log( beta[subID[n],condID[n]] ,overdisp);
		if (LP[n] >= 0) {
			LP[n] ~ neg_binomial_2_log( beta[subID[n],condID[n]+C] ,overdisp);
		}
		if (SP[n] >= 0) {
			SP[n] ~ neg_binomial_2_log( beta[subID[n],condID[n]+(2*C)] ,overdisp);
		}
	}
}
