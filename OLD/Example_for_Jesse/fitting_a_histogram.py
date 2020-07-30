import numpy as np
import matplotlib.pyplot as plt
import pystan

"""
An introduction to "stan" and probabilsitic programming.
Here I'll make some fake data. First we'll make some gaussian data and then fit a normal distribution to it. Next we'll fit a straight line to some x/y data. We'll then add uncertainties and then move away from gaussian distributions/linear models!
"""



"""
First project- measuring the mean and standard deviation of some normally distributed data by fitting a Gaussian
"""

# Make our x values- 1000 values
N_datapoints = 1000

true_mean = 12.0
true_standard_deviation = 3.0
x = np.random.randn(N_datapoints) * true_standard_deviation + true_mean

# We now have some data, x, which is a Gaussian of mean 12 and s.d. 3. We'll use Stan to infer these values. This may seem overly simple but it shows how stan works very nicely. 

# We write our stan model in a separate file- first_project.stan.

# We now use the python interface to stan- "pystan" to load our stan model
stan_model = pystan.StanModel(file='first_project.stan')

# Pystan needs a dictionary of data. The names here must match the names in your "model block" in the stan code
data = dict(N=N_datapoints, x=x)

# Now we run the model 
fit = stan_model.sampling(data=data, iter=1000, chains=4)

# here our our results
print(fit)

# My favourite way to deal with stan output is using a pandas dataframe
results = fit.to_dataframe()

# Plot the posteriors of mu and sigma
fig, axs = plt.subplots(ncols=2)
axs[0].hist(results['mu'], bins='fd', color='0.5')
axs[0].axvline(true_mean, c='r')
axs[0].set_xlabel('mu')
axs[1].hist(results['sigma'], bins='fd', color='0.5')
axs[1].axvline(true_standard_deviation, c='r')
axs[1].set_xlabel('Sigma')
fig.suptitle("Posteriors")

# And plot our data with the 'fake' data on top of it
fig, ax = plt.subplots()
ax.hist(x, bins='fd', color='red', density='True')
ax.hist(results['x_tilde'], bins='fd', histtype='step', color='k', linewidth=3.0, density=True)



"""
Second project- fitting a straight line
"""
