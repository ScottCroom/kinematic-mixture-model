functions{

   real sigmoid(real m, real mu, real sigma) {
    return 1.0 /(1.0 + exp(-(m-mu)/sigma));
  }

}

data{
  int N;
  real x[N];
  real mass[N];
}

parameters{


  // Mixture Prob
  //real<lower=0, upper=1> lambda;
  real mu;
  real log_sigma;

  // Let PhiSR and PhiFR be equal to a * b log_m
  real<lower=0> phiSR;
  real<lower=0> phiFR;

  //real<lower=0, upper=1> offset;

  // real b_FR; 
  // real b_SR;

  real c_FR; 
  real c_SR;

  real d_FR; 
  real d_SR;

  real e_FR; 
  real e_SR;
}

transformed parameters{

  // real<lower=0> phiSR[N];
  // real<lower=0> phiFR[N];
  real<lower=0, upper=1> lambda[N];

  // real locSR[N];
  // real locFR[N];

  ordered[2] loc[N];

  real tmp_FR;
  real tmp_SR;

  for (n in 1:N){


  lambda[n] = sigmoid(mass[n], mu, exp(log_sigma));
  // phiSR[n] = a_SR + b_SR * mass[n];
  // phiFR[n] = a_FR + b_FR * mass[n];
  tmp_SR = c_SR + d_SR * mass[n] + e_SR * mass[n] .* mass[n];
  tmp_FR = c_FR + d_FR * mass[n] + e_FR * mass[n] .* mass[n];
  loc[n, 1] = sigmoid(tmp_SR, 0.0, 1.0); // Slow Rotators
  loc[n, 2] = sigmoid(tmp_FR, 0.0, 1.0); // Fast rotators

  }
}

model{

  //locSR ~ normal(0.07, 0.1);
  //locFR ~ normal(0.5, 0.1);
  phiSR ~ cauchy(0, 100);
  phiFR ~ cauchy(0, 10);
  // a_SR ~ normal(40, 5);
  // a_FR ~ normal(10, 5);
  // b_SR ~ normal(0, 10);
  // b_FR ~ normal(0, 10);

  c_SR ~ normal(-2.75, 0.2);
  c_FR ~ normal(0.2, 0.3);
  d_SR ~ normal(0, 3);
  d_FR ~ normal(0, 3);
  e_SR ~ normal(0, 3);
  e_FR ~ normal(0, 3);

  // loc[, 1] ~ beta(1, 10);
  // loc[, 2] ~ beta(10, 10);

  //offset ~ uniform(0, 1);


  mu ~ normal(0, 1);
  log_sigma ~ normal(0, 2);

  for (n in 1:N){

    
    //lambda = sigmoid(mass[n], mu, sigma);
    // print("~~~~~~~~~~~~~~~~~~~")
    // print(n, " x[n] ", x[n], " locFR: ", locFR[n], " phiFR: ", phiFR, " lambda: ", lambda[n], "beta: ", beta_lpdf(x[n] | locFR[n] * phiFR, phiFR * (1-locFR[n])))
    // print(n, " x[n] ", x[n], " locSR: ", locSR[n], " phiSR: ", phiSR, " lambda: ", lambda[n], "beta: ", beta_lpdf(x[n] | locSR[n] * phiSR, phiSR * (1-locSR[n])))
    // print("~~~~~~~~~~~~~~~~~~~")
    target += log_mix(lambda[n],
                     beta_lpdf(x[n] | loc[n, 1] * phiSR, phiSR * (1-loc[n, 1])),
                     beta_lpdf(x[n] | loc[n, 2] * phiFR, phiFR * (1-loc[n, 2]))
                     );

     // Add in the Jacobian of the transformation
     //target += log(fabs(d_SR + 2. * mass[n] * e_SR));
     //target += log(fabs(d_FR + 2. * mass[n] * e_FR));
  }
}

generated quantities{

  vector[N] x_tilde;
  vector[N] SR_flag;

  // vector[N_pred] predicted_x;
  // vector[N_pred] SR_flag;
  
  //real pred_lam

  //PPC checks
  for (i in 1:N){

    //Random numbers- bernoulli trial with prob. lamda to see if we have an FR or SR
    if (bernoulli_rng(lambda[i])) {
      x_tilde[i] = beta_rng(loc[i, 1] * phiSR, phiSR * (1-loc[i, 1]));
      SR_flag[i] = 0;
      }

    else {
      x_tilde[i] = beta_rng(loc[i, 2] * phiFR, phiFR * (1-loc[i, 2]));
      SR_flag[i] = 1;
      }
    }

  // PRedictions

  // for (j in 1:N){



  //   pred_lam = sigmoid(mass_pred[j], mu, exp(log_sigma));
  //   // phiSR[n] = a_SR + b_SR * mass[n];
  //   // phiFR[n] = a_FR + b_FR * mass[n];
  //   if (bernoulli_rng(pred_lam)) {
  //   tmp_SR = c_SR + d_SR * mass_pred[j] + e_SR * mass_pred[j] .* mass_pred[j];
    
  //   loc_SR = sigmoid(tmp_SR, 0.0, 1.0); // Slow Rotators
  //    // Fast rotators

  //   else {
  //     tmp_FR = c_FR + d_FR * mass_pred[j] + e_FR * mass_pred[j] .* mass_pred[j];
  //   }

  // }
  }
