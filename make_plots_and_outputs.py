import numpy as np
import matplotlib.pyplot as plt
import pandas_tools as P
import seaborn as sns
import scipy.stats as stats
import corner
import pandas as pd
import yaml
import os
import argparse


"""
After fitting a mixture model on some input mass/lambda_r data using the provided Stan files, make plots and outputs 
"""

parser = argparse.ArgumentParser(description='Make plots and table from a mixture model fit to some data')
parser.add_argument("config_filename", help='location of the config file')

args = parser.parse_args()
config_filename = args.config_filename

with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

data_filename = config['data_filename']
lambda_r_column_name = config['lambda_r_column_name']
mass_column_name = config['mass_column_name']

data_df = P.load_FITS_table_in_pandas(data_filename)

# Apply the selection mask
data_df = data_df[eval(config['mask'])]
N_rows = len(data_df)
print(f"After applying the given mask, we have {N_rows} rows of data")

# Get the columns
lambda_r = data_df[lambda_r_column_name]
log_m = data_df[mass_column_name]



## Get the results
samples = pd.read_csv(f"{config['results_folder']}/Chains/{config['outfile_stem']}_samples.csv")
posterior_predctive_checks = samples.loc[:, 'x_tilde[1]':f'x_tilde[{N_rows}]']
posterior_predctive_SR_flag = samples.loc[:, 'SR_flag[1]':f'SR_flag[{N_rows}]']
lambdas = samples.loc[:, 'lambda[1]':f'lambda[{N_rows}]']
N_ppcs = posterior_predctive_checks.shape[0]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######## Posterior Predictive Checks
# A three panel plot with the FR/SR KDEs, the p(SR) as a function of mass and the histogram of PPCs

fig, axs = plt.subplots(ncols=3, figsize=(13, 4))
print("Making Histogram PPC plot...")
if config['publication_style']:
    plt.style.use('publication')
else:
    plt.style.use('default')
_, bins, _ = axs[2].hist(lambda_r, bins=15, histtype='step', linewidth=10.0, edgecolor='k', zorder=1)
_, bins, _ = axs[2].hist(lambda_r, bins=15, facecolor='lightgrey', linewidth=0.0)
for i in range(N_ppcs):
    axs[2].hist(posterior_predctive_checks.iloc[i], bins=bins, histtype='step', linewidth=0.5, alpha=0.1, color='orange', zorder=10)
axs[2].set_xlabel(r'$\lambda_R$')


print("Making the mass vs SR probability plot...")

def sigmoid(mass, mu, sigma): 
    return 1/(1+np.exp(-(mass-mu)/sigma))

mm = np.linspace(log_m.min(), log_m.max(), 1000)
for i in range(100):
    random = np.random.randint(0, 2000)
    axs[1].plot(mm, sigmoid(mm - mm.mean(), samples.loc[samples.index[random], 'mu'].mean(), np.exp(samples.loc[samples.index[random], 'log_sigma']).mean()), c='k', alpha=0.1)
axs[1].plot(mm, sigmoid(mm - mm.mean(), samples['mu'].mean(), np.exp(samples['log_sigma']).mean()), c='firebrick', linewidth=3.0)
axs[1].set_xlabel(r'$\log_{10}(M_*/M_{\odot})$')
axs[1].set_ylabel(r'$p(\mathrm{SR})$')
print("...Done")


#### Use the PPC checks from the Stan model
print("Making lamda/epsilon plot...")
def get_PPC_draws(index, mask_value):

    x = log_m.loc[posterior_predctive_SR_flag.values[index, :] == mask_value]
    y = posterior_predctive_checks.values[index, posterior_predctive_SR_flag.values[index, :] == mask_value] 

    return x, y


# Posterior Predictive Checks on the 2D lambda/eps plot
all_ys_SRs=[] 
all_xs_SRs=[] 

all_ys_FRs=[] 
all_xs_FRs=[] 


for i in range(N_rows): 
    x, y = get_PPC_draws(i, 0) 
    all_xs_SRs.extend(list(x.values)) 
    all_ys_SRs.extend(list(y)) 

    x, y = get_PPC_draws(i, 1) 
    all_xs_FRs.extend(list(x.values)) 
    all_ys_FRs.extend(list(y)) 


random_samples = np.random.choice(np.arange(len(all_xs_SRs)), size=10000)
all_xs_SRs = np.array(all_xs_SRs)[random_samples]
all_ys_SRs = np.array(all_ys_SRs)[random_samples]

random_samples = np.random.choice(np.arange(len(all_xs_FRs)), size=10000)
all_xs_FRs = np.array(all_xs_FRs)[random_samples]
all_ys_FRs = np.array(all_ys_FRs)[random_samples]


# And now plot them on the lambda-epsilon plot
axs[0] = sns.kdeplot(all_xs_FRs, all_ys_FRs, cmap='Blues', shade=True, shade_lowest=False, label='FRs/ORs', ax=axs[0])
axs[0] = sns.kdeplot(all_xs_SRs, all_ys_SRs, cmap='Reds', shade=True, shade_lowest=False, label='SRs/NORs', ax=axs[0])
axs[0].scatter(log_m, lambda_r, c='k', alpha=0.1, s=10, linewidths=0)
axs[0].set_xlim(9.5, 11.8)
axs[0].set_ylim(0.0, 1.0)
axs[0].set_xlabel(r'$\log_{10}(M_*/M_{\odot})$')
axs[0].set_ylabel(r'$\lambda_R$')
axs[0].legend(fontsize=15)

print("...Done")

for ax in axs:
    ax.set_box_aspect(1)
    ax.tick_params(which='both', axis='both', direction='in')
fig.tight_layout()
fig.savefig(f"{config['results_folder']}/Plots/{config['outfile_stem']}_three_panel_combined.pdf", bbox_inches='tight')


# # # Make some helper functions for predicting things at any mass/lambda_r
# # from scipy.special import betainc
# # from tqdm import tqdm
# # def get_beta_loc(mass, SR_or_FR):
# #     if SR_or_FR == 'SR':
# #         linear_combination = samples.c_SR + samples.d_SR * mass# + samples.e_SR * mass**2 
# #     else:
# #         linear_combination = samples.c_FR + samples.d_FR * mass# + samples.e_FR * mass**2 
# #     return linear_combination

# # def get_beta_variance(mass, SR_or_FR):
# #     if SR_or_FR == 'SR':
# #         return samples.intercept_SR + samples.m_SR * mass
# #     else:
# #         return samples.intercept_FR + samples.m_FR * mass# + samples.e_SR * mass**2 


# # def get_probability_from_mass(mass):

# #     return sigmoid(mass, samples.mu, np.exp(samples.log_sigma))

# # def get_beta_distributions(mass, SR_or_FR):

# #     #lam = get_probability_from_mass(mass)
# #     a = get_beta_loc(mass, SR_or_FR)
# #     b = get_beta_variance(mass, SR_or_FR)

# #     return stats.beta(a=a, b=b)


# # def plot_2d_map(ms, xs, SR_or_FR):

# #     probability_density = np.zeros((len(ms), len(xs)))


# #     for i, m in enumerate(tqdm(ms)):
# #         for j, x in enumerate(xs):
# #             betas = get_beta_distributions(m, SR_or_FR)
# #             mixture_prob = get_probability_from_mass(m)
# #             if SR_or_FR == 'SR':
# #                 probability_density[i, j] = (mixture_prob * betas.pdf(x)).mean()
# #             else:
# #                 probability_density[i, j] = ((1 - mixture_prob) * betas.pdf(x)).mean()
# #     return probability_density


# # ms = np.linspace(9.5, 12, 100)
# # xs = np.linspace(0.0, 1.0, 100)

# # p_SR = plot_2d_map(ms - log_m.mean(), xs, SR_or_FR='SR')
# # p_FR = plot_2d_map(ms - log_m.mean(), xs, SR_or_FR='FR')


# # p_SR[~np.isfinite(p_SR)] = 0.0
# # p_FR[~np.isfinite(p_FR)] = 0.0

# # fig, ax = plt.subplots()
# # ax.scatter(log_m, X, s=5, alpha=1,  color='0.2', zorder=10)
# # CS = ax.contour(ms, xs, (p_SR/p_FR).T, colors='k', levels=[0.1, 1, 10, 50, 90, 110, 150])
# # ax.contour(ms, xs, (p_SR/p_FR).T, colors='firebrick', levels=[1], linewidths=3.0)
# # contours = ax.contourf(ms, xs, (p_SR/p_FR).T, cmap='Oranges', levels=CS.levels)  
# # fig.colorbar(contours, ax=ax, label=r'$p_{\mathrm{SR}}/p_{\mathrm{FR}}$')
# # ax.set_xlabel(r'$\log_{10}(M_*/M_{\odot})$')
# # ax.set_ylabel(r'$\lambda_R$')
# # fig.savefig('Different_parameterisation/p_SR_over_p_FR.pdf', bbox_inches='tight')



# # def plot_mass_histogram_with_fit(X, mass_lower, mass_upper):

# #     fig, ax = plt.subplots()

# #     mask = (log_m > mass_lower) & (log_m < mass_upper)
# #     average_mass = (mass_upper + mass_lower)/2.0
# #     average_mass_rescaled = average_mass - log_m.mean()

# #     ax.hist(X[mask], bins='fd', density=True)

# #     fraction = sigmoid(average_mass_rescaled, df['mu'].mean(), np.exp(df['log_sigma']).mean())
# #     zz = np.linspace(0.0, 1.0, 1000)

# #     a = get_beta_loc(average_mass_rescaled, 'FR').mean()
# #     b = get_beta_variance(average_mass_rescaled, 'FR').mean()
# #     FR = stats.beta(a=a, b=b)

# #     a = get_beta_loc(average_mass_rescaled, 'SR').mean()
# #     b = get_beta_variance(average_mass_rescaled, 'SR').mean()
# #     SR = stats.beta(a=a, b=b)

# #     ax.plot(zz, (1 - fraction) * FR.pdf(zz), c='dodgerblue')
# #     ax.plot(zz, fraction * SR.pdf(zz), c='crimson')
# #     ax.plot(zz, (1 - fraction) * FR.pdf(zz) + fraction * SR.pdf(zz), c='k')

# #     return fig

# # # # fig5, ax5 = plt.subplots()
# # # # SR_prob = 1 - sigmoid(log_m - log_m.mean(), df['mu'].mean(), df['sigma'].mean())
# # # # ax5.scatter(log_m, X, c='None', edgecolors='k', linewidth=3.0)
# # # # im = ax5.scatter(log_m, X, c=SR_prob.values, edgecolors='None', cmap='RdYlBu', vmin=0, vmax=1)
# # # # fig5.colorbar(im, label=r"P(SR)")
# # # # ax5.set_ylabel(r'$\lambda_r$') 
# # # # ax5.set_xlabel(r'M*')
# # # # ax5.set_yscale('log')
# # # # fig5.savefig('Plots/RealData/M_lamda_scatter.pdf', bbox_inches='tight')

# # # # fig6, ax6 = plt.subplots()
# # # # SR_prob = best_SR_beta.pdf(X)
# # # # ax6.scatter(log_m, X, c='None', edgecolors='k', linewidth=3.0)
# # # # im = ax6.scatter(log_m, X, c=SR_prob, edgecolors='None', cmap='RdYlBu', vmin=0, vmax=1)
# # # # ax6.set_ylabel(r'$\lambda_r$') 
# # # # ax6.set_xlabel(r'M*')
# # # # ax6.set_yscale('log')
# # # # fig6.colorbar(im, label=r"P(drawn from SR beta dist)")
# # # # fig6.savefig('Plots/RealData/M_lamda_scatter2.pdf', bbox_inches='tight')


# # # # plt.close('all')
# # # # plt.style.use('default')
# # # # fig4 = corner.corner(samples.loc[:, labels], labels=labels, show_titles=True) 
# # # # fig4.savefig('Plots/RealData/corner.pdf', bbox_inches='tight')
# # # # plt.close('all')