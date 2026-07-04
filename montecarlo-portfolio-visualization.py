from Supporting_functions import *
import scipy

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy.optimize as sp
import pandas as pd
import yfinance as yf
import time


#plotting the MonteCarlo results to visualize the solution. First on YTrain, we can see that it is the optimum as wished. 
#Then on the Ytest, we can directly see if it is an optimum or not.
#The portfolio composition are plotted as well.

beta = 0.5
alpha=1.6/252
phi=scipy.stats.norm.ppf(beta)



for n_assets in [3,20]:
    print(f"\n=== Results for n_assets = {n_assets} ===")
    Ytrain, Ytest = download_finance_data(n_assets=n_assets)
    mu, sigma = compute_moments(Ytrain)
    x = markovitz_portfolio_probabilistic(mu, sigma, phi, alpha, beta)
    print("Optimum Portfolio x*  :", x)

    s_tr, s_te = compute_metrics(x, Ytrain, Ytest)
    
    n_samples = 1000

    c_mean, c_var = montecarlo_sim(n_assets, n_samples, Ytrain)
    scatter_plot_port(c_mean, c_var, s_tr.iloc[0], s_tr.iloc[1], f"Montecarlo on Ytrain -- Markovitz Probabilistic (n={n_assets})")
    portfolio_composition_plot(Ytrain, x, f"Markovitz Probabilistic-Ytrain-n{n_assets}")

    c_mean, c_var = montecarlo_sim(n_assets, n_samples, Ytest)
    scatter_plot_port(c_mean, c_var, s_te.iloc[0], s_te.iloc[1], f"Montecarlo on Ytest -- Markovitz Probabilistic (n={n_assets})")
    portfolio_composition_plot(Ytest, x, f"Markovitz Probabilistic-Ytest-n{n_assets}")
 