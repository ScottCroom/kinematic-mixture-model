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
  real<lower=0> phiSR;
  real<lower=0> phiFR;

  // Mixture Prob
  //real<lower=0, upper=1> lambda;
  real mu;
  real<lower=0> sigma;
}

// transformed parameters{

//   real<lower=0, upper=1> lambda;
//   lambda = sigmoid(mass, mu, sigma);
// }

model{

  locSR ~ normal(0.07, 0.1);
  locFR ~ normal(0.5, 1.0);
  phiSR ~ pareto(1, 0.1);
  phiFR ~ pareto(1, 1);
  

  mu ~ normal(0, 1);
  sigma ~ normal(0, 2);

  for (n in 1:N){

    real lambda; 
    lambda = sigmoid(mass[n], mu, sigma);
    target += log_mix(lambda,
                     beta_lpdf(x[n] | locFR * phiFR, phiFR * (1-locFR)),
                     beta_lpdf(x[n] | locSR * phiSR, phiSR * (1-locSR))
                     );
  }
}

generated quantities{

  vector[N] x_tilde;
  real lambda;

  for (i in 1:N){

    lambda = sigmoid(mass[i], mu, sigma);

    //Random numbers- bernoulli trial with prob. lamda to see if we have an FR or SR
    if (bernoulli_rng(lambda)) {
      x_tilde[i] = beta_rng(locFR * phiFR, phiFR * (1-locFR));
      }
    else {
      x_tilde[i] = beta_rng(locSR * phiSR, phiSR * (1-locSR));
      }
    }
}