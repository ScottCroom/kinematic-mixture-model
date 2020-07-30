import numpy as np 
import matplotlib.pyplot as plt 
import pandas_tools as P   
import scipy.stats as stats
import corner
import pystan

def sigmoid(mass, mu, sigma): 
    return np.exp(-(mass-mu)/sigma)/(1+np.exp(-(mass-mu)/sigma)) 


data_df = P.load_FITS_table_in_pandas('jvds_stelkin_cat_v011_mge_seeing_corrected_kh19_v201119_private.fits')


data_df['SR'] = np.nan
data_df.loc[data_df.loc[:, 'FR_MC16']==1.0, 'SR'] = 0.0 
data_df.loc[data_df.loc[:, 'SR_MC16']==1.0, 'SR'] = 1.0 



X_all = data_df['LAMBDAR_RE_EO']
log_m_all = data_df['LMSTAR']

finite_mask = (np.isfinite(X_all) & (np.isfinite(log_m_all)))
mass_mask = log_m_all>9.5
clean_selection = data_df.GAL_FLAG == 0 

overall_mask = finite_mask & mass_mask & clean_selection

X = X_all[overall_mask]
log_m = log_m_all[overall_mask]
SR_flag = data_df.loc[overall_mask, 'SR']


FR_mask = (SR_flag == 0)
SR_mask = (SR_flag == 1)

#Plot the joint plots
sns.jointplot(log_m_all[FR_mask], X_all[FR_mask], kind='kde', stat_func=None) 
sns.jointplot(log_m_all[SR_mask], X_all[SR_mask], kind='kde', stat_func=None) 


import scipy.stats as stats

FR_mvnorm = stats.multivariate_normal(mean=[0, 0], cov=np.cov(log_m[FR_mask], X[FR_mask]))                                                    
randoms = FR_mvnorm.rvs(10000)                                                                    

# sns.jointplot(randoms[:, 0], randoms[:, 1])                                                       
# sns.jointplot(randoms[:, 0], randoms[:, 1], kind='kde', stat_func=None)                           
norm = stats.norm()
x_unif = norm.cdf(randoms)
h = sns.jointplot(x_unif[:, 0], x_unif[:, 1], kind='hex', stat_func=None)