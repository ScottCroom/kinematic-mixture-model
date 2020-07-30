functions{

   real sigmoid(real m, real mu, real sigma) {
    return exp(-(m-mu)/sigma)/(1+exp(-(m-mu)/sigma));
  }

}

data{
  int N;
  real x[N];
  real mass[N];
}

parameters{

  real<lower=0, upper=1> locSR;
  real<lower=locSR, upper=1> locFR;

  // Mixture Prob
  //real<lower=0, upper=1> lambda;
  real mu;
  real<lower=0> sigma;

  // Let PhiSR and PhiFR be equal to a * b log_m
  real a_FR;
  real a_SR;
  real b_FR; 
  real b_SR;
}

transformed parameters{

  real<lower=0> phiSR[N];
  real<lower=0> phiFR[N];
  real<lower=0, upper=1> lambda[N];

  for (n in 1:N){

  lambda[n] = sigmoid(mass[n], mu, sigma);
  phiSR[n] = a_SR + b_SR * mass[n];
  phiFR[n] = a_FR + b_FR * mass[n];

  }
}

model{

  locSR ~ normal(0.07, 0.1);
  locFR ~ normal(0.5, 1.0);
  //phiSR ~ pareto(1, 0.1);
  //phiFR ~ pareto(1, 1);
  a_SR ~ normal(40, 5);
  a_FR ~ normal(10, 5);
  b_SR ~ normal(0, 10);
  b_FR ~ normal(0, 10);


  mu ~ normal(0, 1);
  sigma ~ normal(0, 2);

  for (n in 1:N){

    
    //lambda = sigmoid(mass[n], mu, sigma);
    target += log_mix(lambda[n],
                     beta_lpdf(x[n] | locFR * phiFR[n], phiFR[n] * (1-locFR)),
                     beta_lpdf(x[n] | locSR * phiSR[n], phiSR[n] * (1-locSR))
                     );
  }
}

generated quantities{

  vector[N] x_tilde;
  //real mixture_prob;

  for (i in 1:N){

    

    //Random numbers- bernoulli trial with prob. lamda to see if we have an FR or SR
    if (bernoulli_rng(lambda[i])) {
      x_tilde[i] = beta_rng(locFR * phiFR[i], phiFR[i] * (1-locFR));
      }
    else {
      x_tilde[i] = beta_rng(locSR * phiSR[i], phiSR[i] * (1-locSR));
      }
    }
}