// saved as intake_multilevel_model.stan
//
data{
	int<lower=1> N;         // Number of observations
	int<lower=1> S;         // Number of subjects
	int<lower=1> subID[N];  // Subject IDs
	int<lower=1> C;         // Number of conditions
	int<lower=1> condID[N]; // Condition ID
	int<lower=0> P[N];      // Number of pellets eaten per session
	int<lower=0> NP[N];     // Number of pellets left behind (or 'shared') per session
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
	matrix[2*C, S] z;                   // beta proxy
	cholesky_factor_corr[2*C] L_Omega;  // prior correlation
	vector<lower=0>[2*C] tau;           // prior scale
	row_vector[2*C] gamma;              // population means
	real<lower=0> overdisp;
}
//
transformed parameters{
	matrix[S,2*C] beta;                 // individual coefficients for each subject in each condition
	beta = u * gamma + (diag_pre_multiply(tau,L_Omega) * z)';
}
//
model{
	real muP[N];
	real muNP[N];

	to_vector(z) ~ normal(0,1);
	L_Omega ~ lkj_corr_cholesky(2);
	tau ~ exponential(1.5);
	to_vector(gamma) ~ normal(0,1.5);
	overdisp ~ exponential(1);

	for ( n in 1:N ) {
		muP[n] = beta[subID[n],condID[n]] + beta[subID[n],(condID[n]+C)];
		muNP[n] = beta[subID[n],condID[n]];
	}
	P ~ neg_binomial_2_log(muP,overdisp);
	NP ~ neg_binomial_2_log(muNP,overdisp);
}
