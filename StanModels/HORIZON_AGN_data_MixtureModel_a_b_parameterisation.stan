functions{
  real sigmoid(real m, real mu, real sigma) {
    return 1.0 /(1.0 + exp(-(m-mu)/sigma));
  }

}

data{
  // Number of observations
  int N;

  // Lambda r and mass values
  real lambda_r[N];
  real mass[N];


  // prior locations
}


parameters{

  // The parameters which control the dependence of mixture probability on mass
  real mu;
  real log_sigma;


  // The parameters of 'a', which is a linear function of mass (i.e. two params) for the SRs
  real intercept_SR;
  real m_SR;
  real m_SR_2;

  // The parameters of 'a', which is a linear function of mass (i.e. two params) for the FRs
  real intercept_FR;
  real m_FR;
  real m_FR_2;

  // The parameters of 'b', which is a linear function of mass (i.e. two params) for the FRs
  real c_FR; 
  real d_FR; 
  real e_FR;

  // The parameters of 'b', which is a linear function of mass (i.e. two params) for the SRs
  real c_SR;
  real d_SR;
  real e_SR;

}

transformed parameters{

  // Micture probability
  real<lower=0, upper=1> lambda[N];

  // A and B values for the FR and SR beta functions
  // These are a linear fuction of mass
  vector[N] A_FR;
  vector[N] A_SR;
  vector[N] B_FR;
  vector[N] B_SR;


  for (n in 1:N){

  // Lambda is a sigmoid function of mass, with parameters mu and sigma
  lambda[n] = sigmoid(mass[n], mu, exp(log_sigma));

  // 'a' and 'b' are linear functions of mass
  A_SR[n] = exp(c_SR + d_SR * mass[n] + e_SR * mass[n] * mass[n]);
  A_FR[n] = exp(c_FR + d_FR * mass[n] + e_FR * mass[n] * mass[n]);

  B_FR[n] = exp(intercept_FR + m_FR * mass[n] + m_FR_2 * mass[n] * mass[n]);
  B_SR[n] = exp(intercept_SR + m_SR * mass[n] + m_SR_2 * mass[n] * mass[n]);
  }
}
model{


  intercept_SR ~ normal(log(100), 0.5);
  m_SR ~ normal(0, 3);
  m_SR_2 ~ normal(0, 1);
 
  intercept_FR ~ normal(log(15.0), 3);
  m_FR ~ normal(0, 3);
  m_FR_2 ~ normal(0, 1);

  c_SR ~ normal(log(9.74), 3);
  c_FR ~ normal(0, 3);

  d_SR ~ normal(log(1.49), 3);
  d_FR ~ normal(0, 3);

  e_SR ~ normal(0, 1);
  e_FR ~ normal(0, 1);

  mu ~ normal(0, 1);
  log_sigma ~ normal(0, 2);

  for (n in 1:N){

    //lambda = sigmoid(mass[n], mu, sigma);
    // print("~~~~~~~~~~~~~~~~~~~")
    // print(n, " x[n] ", x[n], " locFR: ", locFR[n], " phiFR: ", phiFR, " lambda: ", lambda[n], "beta: ", beta_lpdf(x[n] | locFR[n] * phiFR, phiFR * (1-locFR[n])))
    // print(n, " x[n] ", x[n], " locSR: ", locSR[n], " phiSR: ", phiSR, " lambda: ", lambda[n], "beta: ", beta_lpdf(x[n] | locSR[n] * phiSR, phiSR * (1-locSR[n])))
    // print("~~~~~~~~~~~~~~~~~~~")
    target += log_mix(lambda[n],
                     beta_lpdf(lambda_r[n] | A_SR[n], B_SR[n]),
                     beta_lpdf(lambda_r[n] | A_FR[n], B_FR[n])
                     );

  }
}

generated quantities{

  vector[N] x_tilde;
  vector[N] SR_flag;


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
