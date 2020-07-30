import numpy as np
import matplotlib.pyplot as plt
import pystan

"""
An introduction to "stan" and probabilsitic programming.
Here I'll make some fake data. First we'll make some gaussian data and then fit a normal distribution to it. Next we'll fit a straight line to some x/y data. We'll then add uncertainties and then move away from gaussian distributions/linear models!
"""



"""
Second project- fitting a straight line
"""

# Make our x values- 1000 values
N_datapoints = 1000
x = np.random.rand(N_datapoints) * 5 - 10

true_slope = 0.5
true_intercept = -2.0
intrinsic_scatter = 0.3
y = true_slope * x + true_intercept + np.random.randn(N_datapoints) * intrinsic_scatter


# The stan model
stan_model = pystan.StanModel(file='second_project.stan')

# Pystan needs a dictionary of data. The names here must match the names in your "model block" in the stan code
data = dict(N=N_datapoints, x=x, y=y)

# Now we run the model 
fit = stan_model.sampling(data=data, iter=1000, chains=4)

# here our our results
print(fit)

# My favourite way to deal with stan output is using a pandas dataframe
results = fit.to_dataframe()

fig, ax = plt.subplots()
ax.scatter(x, y, c='k', s=20)

# Now make some x values for our straight line
# These are just equally spaced between -5 and 5
xx = np.linspace(x.min(), x.max(), 100)
ax.scatter(xx, xx*results['m'].mean() + results['c'].mean(), c='r')

import corner
corner.corner(results[['m', 'c', 'sigma']], truths=[true_slope, true_intercept, intrinsic_scatter])