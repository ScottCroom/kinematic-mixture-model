import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt 
import pandas_tools as P   
import scipy.stats as stats
import pandas as pd
import scipy.optimize as opt

def sigmoid(mass, mu, sigma): 
    return 1/(1+np.exp(-(mass-mu)/sigma))


data_df = P.load_FITS_table_in_pandas('Data/jvds_stelkin_cat_v012_mge_seecorr_kh20_v190620_private.fits')


X_all = data_df['LAMBDAR_RE']
log_m_all = data_df['LMSTAR']

finite_mask = (np.isfinite(X_all) & (np.isfinite(log_m_all)))
mass_mask = log_m_all>9.5
galaxy_flag_mask = (data_df.GAL_FLAG == 0) | (data_df.GAL_FLAG == 4) #(gal_flag is a flag based on my own visual inspection of the maps and removes some of the weird outliers)

overall_mask = finite_mask & mass_mask & galaxy_flag_mask

X = X_all[overall_mask]
log_m = log_m_all[overall_mask]


def two_beta(x, *args):
    a1, a2, b1, b2, k1, k2 = args
    ret = k1*stats.beta.pdf(x, a=a1 ,b=b1)
    ret += k2*stats.beta.pdf(x, a=a2 ,b=b2)
    return ret


counts, bins = np.histogram(X, bins=20, density=True)
bin_centres = bins[:-1] + np.ediff1d(bins)


avals_FR = np.zeros(7)
avals_SR = np.zeros(7)

bvals_FR = np.zeros(7)
bvals_SR = np.zeros(7)
zz = np.linspace(0.0, 1.0, 1000)

for i in range(7):
    mask = (log_m > 9.5 + i*0.3) & (log_m < 9.8 + i * 0.3)

    counts, bins = np.histogram(X[mask], bins=bins, density=True)

    params=[10, 5, 5, 10, 1, 5]
    fitted_params,_ = opt.curve_fit(two_beta, bin_centres, counts, p0=params)
    a1, a2, b1, b2, k1, k2 = fitted_params

    avals_FR[i] = a1
    avals_SR[i] = a2

    bvals_FR[i] = b1
    bvals_SR[i] = b2

    plt.figure()
    plt.hist(X[mask], bins=bins)
    plt.plot(zz, k1*stats.beta(a1, b1).pdf(zz))
    plt.plot(zz, k2*stats.beta(a2, b2).pdf(zz))