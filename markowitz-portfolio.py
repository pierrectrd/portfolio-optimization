from Supporting_functions import *


import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy.optimize as sp
import pandas as pd
import yfinance as yf
import time

#for different number assets to see the different result with the metrics given by the subject
for n_assets in [3,5,10,15,20]:
    Ytrain, Ytest = download_finance_data(n_assets=n_assets)
    mu, sigma = compute_moments(Ytrain)
    rmin=1.6/252     
    x = markovitz_portfolio(mu, sigma)
    
    print("Pour n_ assets=",n_assets)
    s_tr, s_te = compute_metrics(x, Ytrain, Ytest)

