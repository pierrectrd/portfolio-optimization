from Supporting_functions import *
import scipy

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy.optimize as sp
import pandas as pd
import yfinance as yf
import time

#plot the time series of our dat, helps to see the evolution of our assets
#we used cumssum to add each day the gain or lose of the asset, it is easier to read and to see the evolution

Ytrain, Ytest = download_finance_data(n_assets=3)

# ---  chaque asset a sa couleur, test en pointillé ---
plt.figure(figsize=(12, 6))
for asset in Ytrain.columns:
    ytrain_cum = Ytrain[asset].cumsum()
    ytest_cum = Ytest[asset].cumsum() + ytrain_cum.iloc[-1]
    plt.plot(Ytrain.index, ytrain_cum, label=f'Train: {asset}')
    plt.plot(Ytest.index, ytest_cum, '--', label=f'Test: {asset}')
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Cumulative time series', fontsize=16)
    plt.title('Cumulative time series of assets (train: solid, test: dashed)', fontsize=18)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('time-series data of the assets.pdf',bbox_inches='tight')
plt.show()
