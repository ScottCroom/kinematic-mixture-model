import numpy as np 
import matplotlib.pyplot as plt 
import pandas_tools as P   
import scipy.stats as stats
import corner
import pystan
import pandas as pd
import yaml
import os
import argparse

"""
Run a mixture model on some input mass/lambda_r data using the provided Stan files. 
"""

parser = argparse.ArgumentParser(description='Fit a mixture model to some data')
parser.add_argument("config_filename", help='location of the config file')

args = parser.parse_args()
config_filename = args.config_filename

with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

data_filename = config['data_filename']
lambda_r_column_name = config['lambda_r_column_name']
mass_column_name = config['mass_column_name']

### Make the output folders
if not os.path.exists(f"{config['results_folder']}"):
    os.makedirs(f"{config['results_folder']}")
if not os.path.exists(f"{config['results_folder']}/Outputs"):
    os.makedirs(f"{config['results_folder']}/Outputs")
if not os.path.exists(f"{config['results_folder']}/Chains"):
    os.makedirs(f"{config['results_folder']}/Chains")
if not os.path.exists(f"{config['results_folder']}/Plots"):
    os.makedirs(f"{config['results_folder']}/Plots")

### Load the data
data_df = P.load_FITS_table_in_pandas(data_filename)


# Apply the selection mask
data_df = data_df[eval(config['mask'])]
N_rows = len(data_df)
print(f"After applying the given mask, we have {N_rows} rows of data")

if config['subsample']:
    data_df = data_df.sample(config['n_samples'])
    N_rows = len(data_df)
    print(f"After subsampling, we have {N_rows} rows of data")
    print(f"Saving subsample to {config['results_folder']}/Outputs/subsample.csv")
    data_df.to_csv(f"{config['results_folder']}/Outputs/subsample.csv", index=False)

# Get the columns
lambda_r = data_df[lambda_r_column_name]
log_m = data_df[mass_column_name]

## Initial guesses from the config file
def get_init_values():
    return config['init_values']
# labels = ['B_FR', 'B_SR', 'c_FR', 'c_SR', 'd_FR', 'd_SR', 'mu', 'sigma']


### Do the fitting
data=dict(N=N_rows, lambda_r=lambda_r.values, mass=(log_m - np.mean(log_m)).values)
sm = pystan.StanModel(file=config['stan_model_file'])
fit = sm.sampling(data=data, iter=config['iterations'], chains=config['chains'], init=get_init_values, control=config['stan_options'])

print("Checking all diagnostic parameters. This may take a while...")
# Don't check on our predicted quantities- these sometimes give a n_eff of nan
diagnostics = pystan.check_hmc_diagnostics(fit, pars=fit.model_pars[:-2])
print(f"\t...Done: {diagnostics}")

### Save the chain
samples = fit.to_dataframe()
samples.to_csv(f"{config['results_folder']}/Chains/{config['outfile_stem']}_samples.csv")
