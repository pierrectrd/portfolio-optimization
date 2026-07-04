
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy.optimize as sp
import pandas as pd
import yfinance as yf
import time

import matplotlib



n = 20
m = 30
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)


x = cp.Variable(n)
constraints = [x<=1,x >= 0]

obj = cp.Minimize(cp.sum_squares(A@x - b))


prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)


## Q2.2
n_values = np.linspace(20, 200, 10, dtype=int)
m_values = np.linspace(30, 300, 10, dtype=int)
times = []

for n, m in zip(n_values, m_values):
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        x = cp.Variable(n)
        constraints = [x <= 1, x >= 0]
        obj = cp.Minimize(cp.sum_squares(A @ x - b))
        prob = cp.Problem(obj, constraints)

        start = time.time()
        prob.solve()
        end = time.time()
        times.append(end - start)
        
#plot the time resolution for this method.
labels=[f"({n},{m})"for n,m in zip(n_values, m_values)]


plt.figure(figsize=(8,5))
plt.plot(n_values, times, marker='o', label='Computation time')
plt.xticks(n_values,labels,fontsize=10)
plt.xlabel('(n,m)',fontsize=12)
plt.ylabel('Computation time (s)',fontsize=16)
plt.title('CVXPY solve time as a function of n and m',fontsize=18)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Temps_CVXPY.pdf',bbox_inches='tight')




