import numpy as np 
import matplotlib.pyplot as plt 

import scipy.stats as stats
import corner
import pystan

a1=5
b1=30
a2=10
b2=15
mixture_prob=0.5

N1 = 1000
N2 = int(mixture_prob * N1 /( 1- mixture_prob))

x1 = stats.beta(a=a1, b=b1).rvs(N1)
x2 = stats.beta(a=a2, b=b2).rvs(N2)

X = np.append(x1, x2)



data=dict(N=len(X), x=X)
sm = pystan.StanModel(file='beta_mix.stan')
fit = sm.sampling(data=data, iter=1000, chains=4)

df = fit.to_dataframe()

alpha_1, alpha_2, beta_1, beta_2, lamda = np.mean(df[['alpha[1]', 'alpha[2]', 'beta_1', 'beta_2', 'lambda']])
best_dist1 =  stats.beta(a=alpha_1, b=beta_1)
best_dist2 =  stats.beta(a=alpha_2, b=beta_2)

xx = np.linspace(0.0, 1.0, 10000)

plt.style.use('publication')
fig, ax = plt.subplots()
ax.hist(X, bins='fd', color='dodgerblue', density=0.1)

for j in range(1000):
    random_num = np.random.randint(2000)
    alpha_1, alpha_2, beta_1, beta_2, lamda = df.loc[df.index[random_num], ['alpha[1]', 'alpha[2]', 'beta_1', 'beta_2', 'lambda']]
    dist1 =  stats.beta(a=alpha_1, b=beta_1)
    dist2 =  stats.beta(a=alpha_2, b=beta_2)
    ax.plot(xx, lamda * best_dist1.pdf(xx), c='firebrick', linewidth=0.5, alpha=0.05)
    ax.plot(xx, (1-lamda) * best_dist2.pdf(xx), c='seagreen', linewidth=0.5, alpha=0.05)

ax.plot(xx, lamda * best_dist1.pdf(xx), c='k', linewidth=3.0, linestyle='dashed')
ax.plot(xx, (1-lamda) * best_dist2.pdf(xx), c='k', linewidth=3.0, linestyle='dashed')



fig = corner.corner(df[['alpha[1]', 'alpha[2]', 'beta_1', 'beta_2', 'lambda']], labels=[r'$\alpha_1$', r'$\alpha_2$', r'$\beta_1$', r'$\beta_2$', r'$\lambda$'], truths=[a1, a2, b1, b2, mixture_prob], show_titles=True
)                                                                      
plt.show()                                               
