from Supporting_functions import *
import scipy

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy.optimize as sp
import pandas as pd
import yfinance as yf
import time


#Test of the Markoviitz Probabilistic different assets and beta can be changed

beta = 0.49
alpha=1.6/252
phi=scipy.stats.norm.ppf(beta)

Ytrain, Ytest = download_finance_data(n_assets=3)
mu, sigma = compute_moments(Ytrain)
rmin=1.6/252     
x = markovitz_portfolio_probabilistic(mu, sigma,phi,alpha,beta)


for n_assets in [3,5,10,15,20]:
    Ytrain, Ytest = download_finance_data(n_assets=n_assets)
    mu, sigma = compute_moments(Ytrain)
    rmin=1.6/252     
    x = markovitz_portfolio_probabilistic(mu, sigma,phi,alpha,beta)
    
    print("Pour n_ assets=",n_assets)
    s_tr, s_te = compute_metrics(x, Ytrain, Ytest) 



