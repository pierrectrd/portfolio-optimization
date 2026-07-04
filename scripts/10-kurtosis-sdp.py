from Supporting_functions import *
import scipy

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
 
import cvxpy as cp
import scipy.optimize as sp
import pandas as pd
import yfinance as yf
import time
from scipy.linalg import sqrtm

n_assets = 3 

Ytrain, Ytest = download_finance_data(n_assets=n_assets)
mu, sigma = compute_moments(Ytrain)      # mu : espérances, sigma : covariances
D2, L2, S4 = compute_coefficients(Ytrain, n_assets)
S2 = D2.T @ D2 @ L2
rmin = 1.6 / 252.0      

 
 
def solve_kurtosis_problem(D2, L2, S4, mu, rmin):

    mu=np.asarray(mu).reshape(-1)
    n = len(mu)
    x = cp.Variable(n)
    x=cp.reshape(x, (n, 1))
    X = cp.Variable((n, n), PSD=True)
    z = cp.Variable(L2.shape[0])
    g = cp.Variable()
 


    Sigma4 = S2 @ S4 @ S2.T
    Sigma4_sqrt = sqrtm(Sigma4)
 

    constraints = [
        cp.bmat([[X, x], [x.T, np.ones((1,1))]]) >> 0, 
        z == L2 @ cp.vec(X),
        cp.SOC(g, Sigma4_sqrt @ z),  
        cp.sum(x) == 1,  
        x >= 0,  
        mu.T @ x >= rmin  
    ]
 

    objective = cp.Minimize(g)
    problem = cp.Problem(objective, constraints)
    problem.solve()
 
    return x.value, X.value, z.value, g.value
 

 
t0 = time.time()
x_opt, X_opt, z_opt, g_opt = solve_kurtosis_problem(D2, L2, S4, mu, rmin)
print("Optimal portfolio weights:", x_opt)
print("Optimal g value:", g_opt)
t1 = time.time()

print("Time of calculation :", t1 - t0, "seconds")


residu = np.linalg.norm(X_opt- x_opt.reshape(-1,1) @ x_opt.reshape(1,-1))
print("Norm of X - xx^T :", residu)




