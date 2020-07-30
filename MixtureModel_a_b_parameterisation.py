import numpy as np 
import matplotlib.pyplot as plt 
import pandas_tools as P   
import scipy.stats as stats
import corner
import pystan
import pandas as pd
import yaml

# def sigmoid(mass, mu, sigma): 
#     return 1/(1+np.exp(-(mass-mu)/sigma))
import argparse

parser = argparse.ArgumentParser(description='Fit a mixture model to some data')
parser.add_argument("config_filename", help='location of the config file')

args = parser.parse_args()
config_filename = args.config_filename

with open("config_filename", "r") as f:
    config = yaml.safe_load(f)

data_filename = 



# data_df = P.load_FITS_table_in_pandas('Data/jvds_stelkin_cat_v012_mge_seecorr_kh20_v190620_private.fits')


# X_all = data_df['LAMBDAR_RE']
# log_m_all = data_df['LMSTAR']

# finite_mask = (np.isfinite(X_all) & (np.isfinite(log_m_all)))
# mass_mask = log_m_all>9.5
# galaxy_flag_mask = (data_df.GAL_FLAG == 0) | (data_df.GAL_FLAG == 4) #(gal_flag is a flag based on my own visual inspection of the maps and removes some of the weird outliers)

# overall_mask = finite_mask & mass_mask & galaxy_flag_mask

# X = X_all[overall_mask]
# log_m = log_m_all[overall_mask]

# N=len(X)

# # Starting Guesses

# poly = np.polyfit(log_m - log_m.mean(), X, deg=2)   


# # lSR = 0.06
# # phiSR = 7
# # lFR = 0.55
# # phiFR = 100
# mu = 0.0
# sigma = np.log(0.3)

# labels = ['B_FR', 'B_SR', 'c_FR', 'c_SR', 'd_FR', 'd_SR', 'mu', 'sigma']

# def get_init_values():

#     return dict(
#         #phiSR=np.abs(phiSR),
#         #phiFR=np.abs(phiFR),
#         intercept_SR=100 + np.random.randn(),
#         m_SR = 0.0 + np.random.randn(),

#         intercept_FR = 4.4 + np.random.randn()*0.1,
#         m_FR = -1.3 + np.random.randn()*0.1,

#         c_SR = 7 + np.random.randn()*0.1,
#         c_FR = 4.5 + np.random.randn()*0.1,

#         d_SR = 2.0 + np.random.randn()*0.1,#np.random.randn()*0.1,
#         d_FR = -2.0 + np.random.randn()*0.1,#np.random.randn()*0.1,

#         mu = mu, #+ np.random.randn(),
#         sigma = sigma,


#         )





# data=dict(N=N, x=X, mass=log_m - np.mean(log_m))
# sm = pystan.StanModel(file='beta_mix_function_of_m_all_variables_V2.stan')
# fit = sm.sampling(data=data, iter=1000, chains=4, init=get_init_values, control=dict(adapt_delta=0.99))


# df = fit.to_dataframe()
# posterior_predctive_checks = df.loc[:, 'x_tilde[1]':f'x_tilde[{N}]']
# posterior_predctive_SR_flag = df.loc[:, 'SR_flag[1]':f'SR_flag[{N}]']

# lambdas = df.loc[:, 'lambda[1]':f'lambda[{N}]']

# N_ppcs = posterior_predctive_checks.shape[0]



# # A three panel plot with the FR/SR KDEs, the p(SR) as a function of mass and the histogram of PPCs



# # Posterior Predictive Checks
# ## Histograms of real data and of our PPCs:

# fig, axs = plt.subplots(ncols=3)
# print("Making Histogram PPC plot...")
# #plt.style.use('publication')
# _, bins, _ = axs[2].hist(X, bins=15, histtype='step', linewidth=10.0, edgecolor='firebrick', zorder=10)
# _, bins, _ = axs[2].hist(X, bins=15, facecolor='lightgrey', linewidth=0.0)
# for i in range(N_ppcs):
#     axs[2].hist(posterior_predctive_checks.iloc[i], bins=bins, histtype='step', linewidth=0.5, alpha=0.1, color='k')
# #fig.savefig('Different_parameterisation/PPC_histograms.pdf')
# axs[2].set_xlabel(r'$\lambda_R$')


# # print("...Done")

# print("Making the mass vs SR probability plot...")
# #fig2, ax2 = plt.subplots()
# mm = np.linspace(log_m.min(), log_m.max(), 1000)
# for i in range(100):
#     random = np.random.randint(0, 2000)
#     axs[1].plot(mm, sigmoid(mm - mm.mean(), df.loc[df.index[random], 'mu'].mean(), np.exp(df.loc[df.index[random], 'log_sigma']).mean()), c='k', alpha=0.1)
# axs[1].plot(mm, sigmoid(mm - mm.mean(), df['mu'].mean(), np.exp(df['log_sigma']).mean()), c='firebrick', linewidth=3.0)
# axs[1].set_xlabel(r'$\log_{10}(M_*/M_{\odot})$')
# axs[1].set_ylabel(r'$p(\mathrm{SR})$')
# #fig2.savefig('Different_parameterisation/p_FR_as_a_function_of_mass.pdf')
# print("...Done")


# #### Use the PPC checks from the Stan model

# print("Making lamda/epsilon plot...")
# def get_PPC_draws(index, mask_value):

#     x = log_m.loc[posterior_predctive_SR_flag.values[index, :] == mask_value]
#     y = posterior_predctive_checks.values[index, posterior_predctive_SR_flag.values[index, :] == mask_value] 

#     return x, y



# # Posterior Predictive Checks on the 2D lambda/eps plot
# all_ys_SRs=[] 
# all_xs_SRs=[] 

# all_ys_FRs=[] 
# all_xs_FRs=[] 


# for i in range(N): 
#     x, y = get_PPC_draws(i, 0) 
#     all_xs_SRs.extend(list(x.values)) 
#     all_ys_SRs.extend(list(y)) 

#     x, y = get_PPC_draws(i, 1) 
#     all_xs_FRs.extend(list(x.values)) 
#     all_ys_FRs.extend(list(y)) 


# random_samples = np.random.choice(np.arange(len(all_xs_SRs)), size=10000)
# all_xs_SRs = np.array(all_xs_SRs)[random_samples]
# all_ys_SRs = np.array(all_ys_SRs)[random_samples]

# random_samples = np.random.choice(np.arange(len(all_xs_FRs)), size=10000)
# all_xs_FRs = np.array(all_xs_FRs)[random_samples]
# all_ys_FRs = np.array(all_ys_FRs)[random_samples]


# # And now plot them on the lambda-epsilon plot

# import seaborn as sns

# fig, ax = plt.subplots()

# axs[0] = sns.kdeplot(all_xs_FRs, all_ys_FRs, cmap='Blues', shade=True, shade_lowest=False, label='FRs/ORs', ax=axs[0])
# axs[0] = sns.kdeplot(all_xs_SRs, all_ys_SRs, cmap='Reds', shade=True, shade_lowest=False, label='SRs/NORs', ax=axs[0])
# axs[0].scatter(log_m, X, c='k', alpha=0.1, s=10, linewidths=0)
# axs[0].set_xlim(9.5, 11.8)
# axs[0].set_ylim(0.0, 1.0)
# axs[0].set_xlabel(r'$\log_{10}(M_*/M_{\odot})$')
# axs[0].set_ylabel(r'$\lambda_R$')
# axs[0].legend(fontsize=15)
# #fig.savefig('Different_parameterisation/PPC_on_lam_mass_plot.pdf', bbox_inches='tight')
# print("...Done")

# for ax in axs:
#     ax.set_aspect('equal', 'box')
#     ax.tick_params(which='both', axis='both', direction='in')

# fig.savefig("three_panel_combined.pdf", bbox_inches='tight')


# # Make some helper functions for predicting things at any mass/lambda_r
# from scipy.special import betainc
# from tqdm import tqdm
# def get_beta_loc(mass, SR_or_FR):
#     if SR_or_FR == 'SR':
#         linear_combination = df.c_SR + df.d_SR * mass# + df.e_SR * mass**2 
#     else:
#         linear_combination = df.c_FR + df.d_FR * mass# + df.e_FR * mass**2 
#     return linear_combination

# def get_beta_variance(mass, SR_or_FR):
#     if SR_or_FR == 'SR':
#         return df.intercept_SR + df.m_SR * mass
#     else:
#         return df.intercept_FR + df.m_FR * mass# + df.e_SR * mass**2 


# def get_probability_from_mass(mass):

#     return sigmoid(mass, df.mu, np.exp(df.log_sigma))

# def get_beta_distributions(mass, SR_or_FR):

#     #lam = get_probability_from_mass(mass)
#     a = get_beta_loc(mass, SR_or_FR)
#     b = get_beta_variance(mass, SR_or_FR)

#     return stats.beta(a=a, b=b)


# def plot_2d_map(ms, xs, SR_or_FR):

#     probability_density = np.zeros((len(ms), len(xs)))


#     for i, m in enumerate(tqdm(ms)):
#         for j, x in enumerate(xs):
#             betas = get_beta_distributions(m, SR_or_FR)
#             mixture_prob = get_probability_from_mass(m)
#             if SR_or_FR == 'SR':
#                 probability_density[i, j] = (mixture_prob * betas.pdf(x)).mean()
#             else:
#                 probability_density[i, j] = ((1 - mixture_prob) * betas.pdf(x)).mean()
#     return probability_density


# ms = np.linspace(9.5, 12, 100)
# xs = np.linspace(0.0, 1.0, 100)

# p_SR = plot_2d_map(ms - log_m.mean(), xs, SR_or_FR='SR')
# p_FR = plot_2d_map(ms - log_m.mean(), xs, SR_or_FR='FR')


# p_SR[~np.isfinite(p_SR)] = 0.0
# p_FR[~np.isfinite(p_FR)] = 0.0

# fig, ax = plt.subplots()
# ax.scatter(log_m, X, s=5, alpha=1,  color='0.2', zorder=10)
# CS = ax.contour(ms, xs, (p_SR/p_FR).T, colors='k', levels=[0.1, 1, 10, 50, 90, 110, 150])
# ax.contour(ms, xs, (p_SR/p_FR).T, colors='firebrick', levels=[1], linewidths=3.0)
# contours = ax.contourf(ms, xs, (p_SR/p_FR).T, cmap='Oranges', levels=CS.levels)  
# fig.colorbar(contours, ax=ax, label=r'$p_{\mathrm{SR}}/p_{\mathrm{FR}}$')
# ax.set_xlabel(r'$\log_{10}(M_*/M_{\odot})$')
# ax.set_ylabel(r'$\lambda_R$')
# fig.savefig('Different_parameterisation/p_SR_over_p_FR.pdf', bbox_inches='tight')



# def plot_mass_histogram_with_fit(X, mass_lower, mass_upper):

#     fig, ax = plt.subplots()

#     mask = (log_m > mass_lower) & (log_m < mass_upper)
#     average_mass = (mass_upper + mass_lower)/2.0
#     average_mass_rescaled = average_mass - log_m.mean()

#     ax.hist(X[mask], bins='fd', density=True)

#     fraction = sigmoid(average_mass_rescaled, df['mu'].mean(), np.exp(df['log_sigma']).mean())
#     zz = np.linspace(0.0, 1.0, 1000)

#     a = get_beta_loc(average_mass_rescaled, 'FR').mean()
#     b = get_beta_variance(average_mass_rescaled, 'FR').mean()
#     FR = stats.beta(a=a, b=b)

#     a = get_beta_loc(average_mass_rescaled, 'SR').mean()
#     b = get_beta_variance(average_mass_rescaled, 'SR').mean()
#     SR = stats.beta(a=a, b=b)

#     ax.plot(zz, (1 - fraction) * FR.pdf(zz), c='dodgerblue')
#     ax.plot(zz, fraction * SR.pdf(zz), c='crimson')
#     ax.plot(zz, (1 - fraction) * FR.pdf(zz) + fraction * SR.pdf(zz), c='k')

#     return fig

# # # fig5, ax5 = plt.subplots()
# # # SR_prob = 1 - sigmoid(log_m - log_m.mean(), df['mu'].mean(), df['sigma'].mean())
# # # ax5.scatter(log_m, X, c='None', edgecolors='k', linewidth=3.0)
# # # im = ax5.scatter(log_m, X, c=SR_prob.values, edgecolors='None', cmap='RdYlBu', vmin=0, vmax=1)
# # # fig5.colorbar(im, label=r"P(SR)")
# # # ax5.set_ylabel(r'$\lambda_r$') 
# # # ax5.set_xlabel(r'M*')
# # # ax5.set_yscale('log')
# # # fig5.savefig('Plots/RealData/M_lamda_scatter.pdf', bbox_inches='tight')

# # # fig6, ax6 = plt.subplots()
# # # SR_prob = best_SR_beta.pdf(X)
# # # ax6.scatter(log_m, X, c='None', edgecolors='k', linewidth=3.0)
# # # im = ax6.scatter(log_m, X, c=SR_prob, edgecolors='None', cmap='RdYlBu', vmin=0, vmax=1)
# # # ax6.set_ylabel(r'$\lambda_r$') 
# # # ax6.set_xlabel(r'M*')
# # # ax6.set_yscale('log')
# # # fig6.colorbar(im, label=r"P(drawn from SR beta dist)")
# # # fig6.savefig('Plots/RealData/M_lamda_scatter2.pdf', bbox_inches='tight')


# # # plt.close('all')
# # # plt.style.use('default')
# # # fig4 = corner.corner(df.loc[:, labels], labels=labels, show_titles=True) 
# # # fig4.savefig('Plots/RealData/corner.pdf', bbox_inches='tight')
# # # plt.close('all')