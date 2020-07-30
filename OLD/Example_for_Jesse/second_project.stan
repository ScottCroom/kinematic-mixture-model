data{

  // The data block contains our measurements we'll pass to the model

  int<lower=0> N;  // The number of data points we have. We also tell Stan that N must be greater than 0 by using a lower bound of 0
  vector[N] x; // A vector of data points of length N
  vector[N] y;
}

parameters {

  // The parameters block contans things we want to estimate using our model

  real m; // The slope og the line
  real c; // The intercept
  real<lower=0> sigma; // The intrinsic scatter of the line
}

model {

  // The model block is where the data and parameters are linked together!

  // The '~' symbol is the basis of all stan models. We say that the thing on the left hand side follows the probability distribution on the right hand side
  // These are our prior distributions we place on our parameters, m, c and sigma

  m ~ normal(0, 5);
  c ~ normal(0, 5); 
  sigma ~ normal(0, 1); // This gives sigma a Gaussian prior too- but since we've said above it can't be negative, this is really a "half" Gaussian. 

  //Now the bit where our model meets the data
  //We say that our data, y, is distributed according to one of the in-built probability distributions in stan, with paramters on the right hand side
  //Here, we say y follows a Gaussian distribution with mean m*x + c and s.d. sigma. When stan is run, it'll estimate the parameters for us! 
  y ~ normal(m * x + c, sigma);


  // That's it!
}

