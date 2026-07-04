from Supporting_functions import *
import scipy

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy.optimize as sp
import pandas as pd
import yfinance as yf
import time
from scipy.linalg import sqrtm

# Implementation of SCP for the Kurtosis problem. Calls kurtosis_scp_solve from supporting functions.
# we test if we have the same solution with different initial conditions and we compare its portfolio with the markovitz one
n_assets=3
Ytrain, Ytest = download_finance_data(n_assets=n_assets)
mu, sigma = compute_moments(Ytrain)

D2, L2, S4 = compute_coefficients(Ytrain, n_assets)
S2 = D2.T @ D2 @ L2


n=len(mu[0])




all_x0 = [
    np.array([0.0, 0.0, 0.0]),
    np.array([1/3, 1/3, 1/3]),
    np.array([0.2, 0.4, 0.4])
]
all_gk = []
all_xscp = []
for x0 in all_x0:
    x_scp, _, _, _, _, gk = kurtosis_scp_solve(L2, S2, S4, mu, x0)
    all_xscp.append(x_scp)
    all_gk.append(gk)



fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Convergence de g pour différentes initialisations (sans affichage des valeurs sur le graphe)
for i, (gk, x_init) in enumerate(zip(all_gk, all_x0)):

    if len(gk) > 1:
        gk = gk[1:]
    label = f"x0={np.array2string(x_init, precision=2, separator=',', suppress_small=True)}"
    axes[0].plot(gk, marker='o', label=label)
axes[0].axhline(2.61, color='red', linestyle='--', linewidth=2, label='g* = 2.61')
axes[0].set_yscale('log')
axes[0].set_xlabel('Iteration', fontsize=16)
axes[0].set_ylabel('g', fontsize=16)
axes[0].set_title('SCP: g convergence', fontsize=18)
axes[0].tick_params(axis='both', labelsize=14)
axes[0].grid(True)
axes[0].legend(fontsize=12, loc='upper right')

# 2. Histogramme comparatif des portefeuilles (pour la deuxième initialisation)
x_scp=all_xscp[0]
x_marko = markovitz_portfolio(mu, sigma).flatten()
asset_names = list(Ytrain.columns)
bar_width = 0.35
indices = np.arange(n)
axes[1].bar(indices, x_scp, bar_width, label='SCP', color='royalblue')
axes[1].bar(indices + bar_width, x_marko, bar_width, label='Markowitz', color='orange')
axes[1].set_xlabel('Asset', fontsize=16)
axes[1].set_ylabel('Portfolio weight', fontsize=16)
axes[1].set_title('Portfolio weights', fontsize=18)
axes[1].set_xticks(indices + bar_width/2)
axes[1].set_xticklabels(asset_names, fontsize=14, rotation=30)
axes[1].tick_params(axis='y', labelsize=14)
axes[1].legend(fontsize=14)
axes[1].grid(True, axis='y')

plt.tight_layout()
plt.savefig('SCP_convergence_and_portfolio_comparison.pdf', bbox_inches='tight')
plt.show()

