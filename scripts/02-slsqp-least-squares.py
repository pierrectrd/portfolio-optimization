
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy.optimize as sp
import pandas as pd
import yfinance as yf
import time
import matplotlib

font = {'family' : 'sans',
        'size'   : 12}
matplotlib.rc('font', **font)


n = 20
m = 30
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)


def objective(x):
        return np.sum((A @ x - b) ** 2)


constraints = [(0, 1)] * n


res = sp.minimize(objective, x0=np.zeros(n), bounds=constraints, method='SLSQP')

print("status:", res.success)
print("optimal value", res.fun)
print("optimal var", res.x)


## Q2.3
n_values = np.linspace(20, 200, 10, dtype=int)
m_values = np.linspace(30, 300, 10, dtype=int)
times = []

for n, m in zip(n_values, m_values):
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        constraints = [(0,1)]*n

        start = time.time()
        res = sp.minimize(objective, x0=np.zeros(n), bounds=constraints, method='SLSQP')
        end = time.time()
        times.append(end - start)
        

labels=[f"({n},{m})"for n,m in zip(n_values, m_values)]


plt.figure(figsize=(8,5))
plt.plot(n_values, times, marker='o', label='Computation time')
plt.xticks(n_values,labels,fontsize=10)
plt.xlabel('(n,m)',fontsize=12)
plt.ylabel('Computation time (s)',fontsize=12)
plt.title('SLSQP solve time as a function of n and m',fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Temps_SLSQP.pdf',bbox_inches='tight')

