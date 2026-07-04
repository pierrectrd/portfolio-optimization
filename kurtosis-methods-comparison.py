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

# Calculates the solution for Kurtosis and Markowitz, then the metrics. Compares the two Kurtosis methods.
# Plots the optimum on the same graph to see the difference between the three methods and their portfolio composition.

n_assets = 3

Ytrain, Ytest = download_finance_data(n_assets=n_assets)
mu, sigma = compute_moments(Ytrain)      # mu: means, sigma: covariances
D2, L2, S4 = compute_coefficients(Ytrain, n_assets)
S2 = D2.T @ D2 @ L2
rmin = 1.6 / 252.0

n_samples = 1000
c_mean, c_var = montecarlo_sim(n_assets, n_samples, Ytest)

# SLSQP method
t0 = time.time()
x_sp, z_sp, g_sp, X_sp = kurtosis_solve(L2, S2, S4, mu)
t1 = time.time()
x_sp = x_sp.reshape(-1, 1)

print("===== SLSQP Results (Q14) =====")
print(f"\nSLSQP time (s): {t1 - t0:.3f}")
print(f"g*: {g_sp}")
print(f"Optimal portfolio x*: {x_sp}")


s_tr_sp, s_te_sp = compute_metrics(x_sp, Ytrain, Ytest)
portfolio_composition_plot(Ytest, x_sp, "SLSQP Method")

# SCP method
np.random.seed(1)
x0 = np.random.dirichlet(np.ones(len(mu[0]))) # ensures sum = 1
t0 = time.time()
x_cvx, z_cvx, g_cvx, X_cvx,res,gk = kurtosis_scp_solve(L2, S2, S4, mu, x0)
t1 = time.time()
n = len(mu[0])
x_cvx = x_cvx.reshape(-1, 1)

print("===== SCP Results (Q13) =====")
print(f"\nSCP time (s): {t1 - t0:.3f}")
print(f"g*: {g_cvx}")
print(f"Optimal portfolio x*: {x_cvx}")

s_tr_cvx, s_te_cvx = compute_metrics(x_cvx, Ytrain, Ytest)
portfolio_composition_plot(Ytest, x_cvx, "SCP Method")


#Markovitz method

x_markov = markovitz_portfolio(mu, sigma)
s_tr_mar, s_te_mar = compute_metrics(x_markov, Ytrain, Ytest)
portfolio_composition_plot(Ytest, x_markov, "Markowitz Method")



#comparaison with markovitz and kurtosis

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax = np.ravel(ax)



cax0 = ax[0].scatter(c_var, c_mean, c=c_mean / c_var, cmap='Spectral')
ax[0].scatter(s_te_sp.iloc[1], s_te_sp.iloc[0], marker='*', s=2 ** 9, color='tab:red', label='SLSQP Computed solution')
ax[0].scatter(s_te_mar.iloc[1], s_te_mar.iloc[0], marker='*', s=2 ** 9, color='tab:orange', label='Markowitz Computed solution')
ax[0].scatter(s_te_cvx.iloc[1], s_te_cvx.iloc[0], marker='*', s=2 ** 9, color='tab:green', label='Markowitz Computed solution')
ax[0].set_ylabel('Return [%]', fontsize=20)
ax[0].set_title('Efficient Frontier: Markowitz vs SLSQP vs SCP', fontsize=24)
ax[0].tick_params(axis='both', labelsize=20)
ax[0].grid(True)
ax[0].legend(fontsize=20)
plt.tight_layout()
plt.savefig('Markowitz_vs_SLSQP.pdf', bbox_inches='tight')
plt.show()








