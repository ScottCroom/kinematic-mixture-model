data{

  // The data block contains our measurements we'll pass to the model

  int<lower=0> N;  // The number of data points we have. We also tell Stan that N must be greater than 0 by using a lower bound of 0
  real x[N]; // A vector of data points of length N
}

parameters {

  // The parameters block contans things we want to estimate using our model

  real mu; // The mean of the data
  real<lower=0> sigma; // The standard deviation of the data. Also say that it can't be negative
}

model {

  // The model block is where the data and parameters are linked together!

  // The '~' symbol is the basis of all stan models. We say that the thing on the left hand side follows the probability distribution on the right hand side
  // These are our prior distributions we place on our parameters, mu and sigma
  // Here I'm giving mu an Gaussian prior, of location 10 and s.d. 5 
  mu ~ normal(10, 5);
  sigma ~ normal(0, 5); // This gives sigma a Gaussian prior too- but since we've said above it can't be negative, this is really a "half" Gaussian. 


  //Now the bit where our model meets the data
  //We say that our data, x, is distributed according to one of the in-built probability distributions in stan, with paramters on the right hand side
  //Here, we say x follows a Gaussian distribution with mean mu and s.d. sigma. When stan is run, it'll estimate the parameters for us! 
  x ~ normal(mu, sigma);


  // That's it!
}


generated quantities{

  // The "generated quantities" block allows us to create "fake" data from our model to see if it looks like our real model 
  // We make some new data, which I'll call x_tilde. Hopefully this will look like x if our model is any good! 
  // This is known as doing a "posterior predictive check".

  real x_tilde;  // This is another way to declare a variable in stan.

  // This is how we make fake data- the normal_rng generates random numbers from a normal distribution with mean mu and scale sigma. 
  // all stan distributions have an "_rng" method too. So you can do "beta_rng" or "student_t_rng" too. 
  // We'll 

  x_tilde = normal_rng(mu, sigma);

}
