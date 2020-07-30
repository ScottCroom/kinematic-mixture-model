import numpy as np 
import matplotlib.pyplot as plt 

import scipy.stats as stats
import corner
import pystan



def sigmoid(mass, mu, sigma): 
    return np.exp(-(mass-mu)/sigma)/(1+np.exp(-(mass-mu)/sigma)) 

N1 = 1000

log_m = np.random.rand(N1)*(12-9) + 9

lSR = 0.06
phiSR = 100
lFR = 0.55
phiFR = 7
mu = 3.1
sigma = 1
truths = [lSR, lFR, phiSR, phiFR, mu, sigma]
labels = ['locSR', 'locFR', 'phiSR', 'phiFR', 'mu', 'sigma']

X = []
x_FR = []
x_SR = []
for i in range(N1):
    lamda = sigmoid(log_m[i] - np.mean(log_m), mu, sigma)

    # Now draw a random number and see if this gal is a slow rotator (number < lamda) or a FR (number > lamda)
    if bool(np.random.binomial(1, lamda)):

        value = stats.beta(a=lFR*phiFR, b=phiFR*(1-lFR)).rvs(1)[0]
        x_FR.append(value)

    else:
        value = stats.beta(a=lSR*phiSR, b=phiSR*(1-lSR)).rvs(1)[0]
        x_SR.append(value)

    X.append(value)

prop = len(x_FR)/(len(X))

fig, ax=plt.subplots()
ax.hist(X, bins='fd', density=True)
xx = np.linspace(0.0, 1.0, 1000)
ax.plot(xx, (1-prop)*stats.beta(a=lSR*phiSR, b=phiSR*(1-lSR)).pdf(xx))
ax.plot(xx, prop*stats.beta(a=lFR*phiFR, b=phiFR*(1-lFR)).pdf(xx))    
ax.set_xlabel(r'$\lambda_r$')
fig.savefig('Plots/fake_data.pdf')


fig2, ax2 = plt.subplots()
mm = np.linspace(log_m.min(), log_m.max(), 1000)
ax2.plot(mm, sigmoid(mm - mm.mean(), mu, sigma))
ax2.set_ylabel("P(fast rotator)")
ax2.set_xlabel('log(M*)')
fig2.savefig('Plots/lamda_M_dependence.pdf')
plt.close('all')

def get_init_values():

    return dict(locSR=np.abs(lSR + np.random.randn()*0.01),
        locFR=np.abs(lFR + np.random.randn()*0.01), 
        phiSR=np.abs(phiSR + np.random.randn()),
        phiFR=np.abs(phiFR + np.random.randn()),
        mu=mu + np.random.randn(),
        sigma=np.abs(sigma + np.random.randn())
        )


data=dict(N=len(X), x=X, mass=log_m - np.mean(log_m))
sm = pystan.StanModel(file='beta_mix_function_of_m.stan')
fit = sm.sampling(data=data, iter=1000, chains=4, init=get_init_values)


df = fit.to_dataframe()
posterior_predctive_checks = df.loc[:, 'x_tilde[1]':'x_tilde[1000]']

fig3, ax3 = plt.subplots()
ax3.hist(X, bins='fd', density=True)     
for i in range(1, 1000):  
    ax3.hist(posterior_predctive_checks.iloc[i], density=True, edgecolor='k', facecolor='None', histtype='step', alpha=0.01) 
ax3.set_title('Posterior Predictive Checks')
fig3.savefig('Plots/PPC.pdf')
plt.show()


fig4 = corner.corner(df.loc[:, labels], labels=labels, show_titles=True, truths=truths) 
fig4.savefig('Plots/corner.pdf')
# df = fit.to_dataframe()

# alpha_1, alpha_2, beta_1, beta_2, lamda = np.mean(df[['alpha[1]', 'alpha[2]', 'beta_1', 'beta_2', 'lambda']])
# best_dist1 =  stats.beta(a=alpha_1, b=beta_1)
# best_dist2 =  stats.beta(a=alpha_2, b=beta_2)

# xx = np.linspace(0.0, 1.0, 10000)

# plt.style.use('publication')
# fig, ax = plt.subplots()
# ax.hist(X, bins='fd', color='dodgerblue', density=0.1)

# for j in range(1000):
#     random_num = np.random.randint(2000)
#     alpha_1, alpha_2, beta_1, beta_2, lamda = df.loc[df.index[random_num], ['alpha[1]', 'alpha[2]', 'beta_1', 'beta_2', 'lambda']]
#     dist1 =  stats.beta(a=alpha_1, b=beta_1)
#     dist2 =  stats.beta(a=alpha_2, b=beta_2)
#     ax.plot(xx, lamda * best_dist1.pdf(xx), c='firebrick', linewidth=0.5, alpha=0.05)
#     ax.plot(xx, (1-lamda) * best_dist2.pdf(xx), c='seagreen', linewidth=0.5, alpha=0.05)

# ax.plot(xx, lamda * best_dist1.pdf(xx), c='k', linewidth=3.0, linestyle='dashed')
# ax.plot(xx, (1-lamda) * best_dist2.pdf(xx), c='k', linewidth=3.0, linestyle='dashed')



# fig = corner.corner(df[['alpha[1]', 'alpha[2]', 'beta_1', 'beta_2', 'lambda']], labels=[r'$\alpha_1$', r'$\alpha_2$', r'$\beta_1$', r'$\beta_2$', r'$\lambda$'], truths=[a1, a2, b1, b2, mixture_prob], show_titles=True
# )                                                                      
# plt.show()                                               
