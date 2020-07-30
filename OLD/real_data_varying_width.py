import numpy as np 
import matplotlib.pyplot as plt 
import pandas_tools as P   
import scipy.stats as stats
import corner
import pystan

def sigmoid(mass, mu, sigma): 
    return np.exp(-(mass-mu)/sigma)/(1+np.exp(-(mass-mu)/sigma)) 


data_df = P.load_FITS_table_in_pandas('jvds_stelkin_cat_v011_mge_seeing_corrected_kh19_v161019_private.fits')

# Starting Guesses
lSR = 0.06
phiSR = 100
lFR = 0.55
phiFR = 7
mu = 3.1
sigma = 1

labels = ['locSR', 'locFR', 'phiSR', 'phiFR', 'mu', 'sigma']

def get_init_values():

    return dict(locSR=np.abs(lSR + np.random.randn()*0.01),
        locFR=np.abs(lFR + np.random.randn()*0.01), 
        a_SR=np.abs(phiSR + np.random.randn()),
        a_FR=np.abs(phiFR + np.random.randn()),
        b_SR=np.random.randn(),
        b_FR=np.random.randn(),
        mu=mu + np.random.randn(),
        sigma=np.abs(sigma + np.random.randn())
        )


X_all = data_df['LAMBDAR_RE_EO']
log_m_all = data_df['LMSTAR']

finite_mask = (np.isfinite(X_all) & (np.isfinite(log_m_all)))
mass_mask = log_m_all>9.5

overall_mask = finite_mask & mass_mask

X = X_all[overall_mask]
log_m = log_m_all[overall_mask]

data=dict(N=len(X), x=X, mass=log_m - np.mean(log_m))
sm = pystan.StanModel(file='beta_mix_function_of_m_varying_width.stan')
fit = sm.sampling(data=data, iter=1000, chains=4, init=get_init_values)


df = fit.to_dataframe()
posterior_predctive_checks = df.loc[:, 'x_tilde[1]':'x_tilde[1000]']

# # Bit of a hack here- get the fraction of SRs by guessing lambda_r < 0.1
# prop = (X<0.2).sum()/len(X) 
# lSR = df['locSR'].mean()
# phiSR = df['phiSR'].mean()
# lFR = df['locFR'].mean()
# phiFR = df['phiFR'].mean()
# mu = df['mu'].mean()
# sigma = df['sigma'].mean()

# best_SR_beta = stats.beta(a=lSR*phiSR, b=phiSR*(1-lSR))
# best_FR_beta = stats.beta(a=lFR*phiFR, b=phiFR*(1-lFR))


# plt.style.use('publication')
# fig, ax=plt.subplots()
# ax.hist(X, bins='fd', density=True, facecolor='lightgrey', edgecolor='k', linewidth=2.0, histtype='stepfilled')
# xx = np.linspace(0.0, 1.0, 1000)
# ax.plot(xx, (prop)*best_SR_beta.pdf(xx), c='firebrick', linewidth=3.0)
# ax.plot(xx, (1 - prop)*best_FR_beta.pdf(xx), c='dodgerblue', linewidth=3.0)  
# for i in range(1000):
#     random = np.random.randint(0, 2000)
#     lSR = df.loc[df.index[random], 'locSR']
#     phiSR = df.loc[df.index[random], 'phiSR']
#     lFR = df.loc[df.index[random], 'locFR']
#     phiFR = df.loc[df.index[random], 'phiFR']
#     ax.plot(xx, (prop)*stats.beta(a=lSR*phiSR, b=phiSR*(1-lSR)).pdf(xx), c='firebrick', linewidth=1.0, alpha=0.01)
#     ax.plot(xx, (1 - prop)*stats.beta(a=lFR*phiFR, b=phiFR*(1-lFR)).pdf(xx), c='dodgerblue', linewidth=1.0, alpha=0.01)  
# ax.set_xlabel(r'$\lambda_r$')
# fig.savefig('Plots/RealData/histograms.pdf', bbox_inches='tight')



# fig2, ax2 = plt.subplots()
# mm = np.linspace(log_m.min(), log_m.max(), 1000)
# for i in range(100):
#     random = np.random.randint(0, 2000)
#     ax2.plot(mm, 1 - sigmoid(mm - mm.mean(), df.loc[df.index[random], 'mu'].mean(), df.loc[df.index[random], 'sigma'].mean()), c='k', alpha=0.1)
# ax2.plot(mm, 1 - sigmoid(mm - mm.mean(), df['mu'].mean(), df['sigma'].mean()), c='firebrick', linewidth=3.0)

# ax2.set_ylabel("P(Slow rotator)")
# ax2.set_xlabel('log(M*)')
# fig2.savefig('Plots/RealData/lamda_M_dependence.pdf', bbox_inches='tight')





fig3, ax3 = plt.subplots()
counts, bins, patches = ax3.hist(X, bins='fd', density=True, facecolor='dodgerblue')
ax3.hist(posterior_predctive_checks.values.ravel(), bins=bins, density=True, edgecolor='firebrick', facecolor='None', histtype='step', linewidth=3.0)
for i in range(1, 1000):  
    ax3.hist(posterior_predctive_checks.iloc[i], bins=bins, density=True, edgecolor='k', facecolor='None', histtype='step', alpha=0.01) 
ax3.set_title('Posterior Predictive Checks')
ax3.set_xlabel(r'$\lambda_r$')
fig3.savefig('Plots/RealData/PPC.pdf', bbox_inches='tight')


# fig5, ax5 = plt.subplots()
# SR_prob = 1 - sigmoid(log_m - log_m.mean(), df['mu'].mean(), df['sigma'].mean())
# ax5.scatter(log_m, X, c='None', edgecolors='k', linewidth=3.0)
# im = ax5.scatter(log_m, X, c=SR_prob.values, edgecolors='None', cmap='RdYlBu', vmin=0, vmax=1)
# fig5.colorbar(im, label=r"P(SR)")
# ax5.set_ylabel(r'$\lambda_r$') 
# ax5.set_xlabel(r'M*')
# ax5.set_yscale('log')
# fig5.savefig('Plots/RealData/M_lamda_scatter.pdf', bbox_inches='tight')

# fig6, ax6 = plt.subplots()
# SR_prob = best_SR_beta.pdf(X)
# ax6.scatter(log_m, X, c='None', edgecolors='k', linewidth=3.0)
# im = ax6.scatter(log_m, X, c=SR_prob, edgecolors='None', cmap='RdYlBu', vmin=0, vmax=1)
# ax6.set_ylabel(r'$\lambda_r$') 
# ax6.set_xlabel(r'M*')
# ax6.set_yscale('log')
# fig6.colorbar(im, label=r"P(drawn from SR beta dist)")
# fig6.savefig('Plots/RealData/M_lamda_scatter2.pdf', bbox_inches='tight')


# plt.close('all')
# plt.style.use('default')
# fig4 = corner.corner(df.loc[:, labels], labels=labels, show_titles=True) 
# fig4.savefig('Plots/RealData/corner.pdf', bbox_inches='tight')
# plt.close('all')