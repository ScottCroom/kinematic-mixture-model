import numpy as np 
import matplotlib.pyplot as plt 
import pandas_tools as P

df = P.load_FITS_table_in_pandas('/Users/samvaughan/Science/SAMI/JVDS_Kinematics/lam_eps_plot/Data/jvds_stelkin_cat_v012_mge_v310820_private.fits')

mask = (np.isfinite(df.LAMBDAR_RE_EO)) & (np.isfinite(df.LMSTAR)) & (df.LMSTAR > 9.5) & ((df.GAL_FLAG == 0) | (df.GAL_FLAG == 4))

df = df.loc[mask]

# Weight things by the 

weighting_per_halo = np.unique(df.HALOMASS_WEIGHT)/np.unique(df.HALOMASS_WEIGHT).min()


ideal_numbers = np.round((df.groupby('HALOMASS_WEIGHT')['CATID'].count() * weights * 1./weights[0])).sum()