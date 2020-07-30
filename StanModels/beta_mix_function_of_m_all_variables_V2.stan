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

  //real<lower=0> B_SR;

  real intercept_SR;
  real m_SR;

  real intercept_FR;
  real m_FR;

  real c_FR; 
  real c_SR;

  real d_FR; 
  real d_SR;

}

transformed parameters{

  // real<lower=0> phiSR[N];
  // real<lower=0> phiFR[N];
  real<lower=0, upper=1> lambda[N];

  // real locSR[N];
  // real locFR[N];

  vector[N] A_FR;
  vector[N] A_SR;
  vector[N] B_FR;
  vector[N] B_SR;

  real tmp_FR;
  real tmp_SR;
  real tmp_B_FR;
  real tmp_B_SR;

  for (n in 1:N){


  lambda[n] = sigmoid(mass[n], mu, exp(log_sigma));
  tmp_SR = c_SR + d_SR * mass[n];
  tmp_FR = c_FR + d_FR * mass[n];

  tmp_B_FR = intercept_FR + m_FR * mass[n];
  tmp_B_SR = intercept_SR + m_SR * mass[n];

  A_SR[n] = tmp_SR; // Slow Rotators
  A_FR[n] = tmp_FR; // Fast rotators
  B_FR[n] = tmp_B_FR;
  B_SR[n] = tmp_B_SR;
  }
}

model{

  //locSR ~ normal(0.07, 0.1);
  //locFR ~ normal(0.5, 0.1);
  intercept_SR ~ normal(100, 10);
  m_SR ~ normal(0, 5);
 
  intercept_FR ~ normal(4.4, 1);
  m_FR ~ normal(-2, 1);

  c_SR ~ normal(6.9, 3);
  c_FR ~ normal(4.5, 3);

  d_SR ~ normal(2, 1);
  d_FR ~ normal(-2, 1);

  mu ~ normal(0, 1);
  log_sigma ~ normal(0, 2);

  for (n in 1:N){

    //lambda = sigmoid(mass[n], mu, sigma);
    // print("~~~~~~~~~~~~~~~~~~~")
    // print(n, " x[n] ", x[n], " locFR: ", locFR[n], " phiFR: ", phiFR, " lambda: ", lambda[n], "beta: ", beta_lpdf(x[n] | locFR[n] * phiFR, phiFR * (1-locFR[n])))
    // print(n, " x[n] ", x[n], " locSR: ", locSR[n], " phiSR: ", phiSR, " lambda: ", lambda[n], "beta: ", beta_lpdf(x[n] | locSR[n] * phiSR, phiSR * (1-locSR[n])))
    // print("~~~~~~~~~~~~~~~~~~~")
    target += log_mix(lambda[n],
                     beta_lpdf(x[n] | A_SR[n], B_SR[n]),
                     beta_lpdf(x[n] | A_FR[n], B_FR[n])
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
      x_tilde[i] = beta_rng(A_SR[i], B_SR[i]);
      SR_flag[i] = 0;
      }

    else {
      x_tilde[i] = beta_rng(A_FR[i], B_FR[i]);
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
