data{

  int N;
  real x[N];
}

parameters{

  positive_ordered[2] alpha;
  real<lower=0> beta_1;
  real<lower=0> beta_2;

  // Mixture Prob
  real<lower=0, upper=1> lambda;
}


model{

  alpha[1] ~ normal(5, 1);
  beta_1 ~ normal(25, 2);
  alpha[2] ~ normal(10, 1);
  beta_2 ~ normal(10, 2);

  lambda ~ beta(0.5, 0.5);

  for (n in 1:N){
    target += log_mix(lambda,
                     beta_lpdf(x[n] | alpha[1], beta_1),
                     beta_lpdf(x[n] | alpha[2], beta_2)
                     );
  }
}