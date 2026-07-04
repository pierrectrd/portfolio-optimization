from Supporting_functions import *
from sqp_solver import * #we use the function coded in sqp_solver
import scipy

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy.optimize as sp
import pandas as pd
import yfinance as yf
import time


# Implementation of SLSQP and CVXPY on a non-convex problem.
# The program compares the two methods in terms of speed, number of iterations, and constraint satisfaction.


n = 20
m = 30
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

 # Objective and constraints for SciPy
def objective(x):
    return 0.5 * np.linalg.norm(A @ x - b) ** 2

cons = [{'type': 'ineq', 'fun': lambda x: np.sum(x**2) - 0.5}]
bounds = [(0, 1)] * n
x0 = np.clip(np.random.rand(n), 0, 1)
if np.sum(x0**2) < 0.5:
    x0[0] = np.sqrt(0.5)
    
    
# Functions for SQP
def f(x):
    return 0.5 * np.linalg.norm(A @ x - b) ** 2
def grad_f(x):
    return A.T @ (A @ x - b)
def hess_f(x):
    return A.T @ A

def g(x):
    
    return np.concatenate([-x, x-1, [0.5 - np.sum(x**2)]])
def Jg(x):
    
    J = np.zeros((2*n+1, n))
    J[:n, :] = -np.eye(n)
    J[n:2*n, :] = np.eye(n)
    J[2*n, :] = -2*x
    return J



# --- Solution with SQP ---
x0_sqp = np.clip(np.random.rand(n), 0, 1)
if np.sum(x0_sqp**2) < 0.5:
    x0_sqp[0] = np.sqrt(0.5)

t0 = time.time()
x_sqp, residuals, feas, iter_sqp  = SQP(x0_sqp,f, grad_f, hess_f, g, Jg, max_iter=50)
t1 = time.time()
t_sqp = t1 - t0


# --- Solution with SLSQP ---
t0 = time.time()
res = sp.minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-6})
t1 = time.time()
t_slsqp = t1 - t0

# --- Display results ---
print("--- SQP (CVXPY) ---")
print(f"Time: {t_sqp:.3f} s, iterations: {iter_sqp}, objective: {0.5 * np.linalg.norm(A @ x_sqp - b) ** 2:.6f}, constraint: {np.sum(x_sqp**2) - 0.5:.3e}")
print("--- SLSQP (SciPy) ---")
print(f"Time: {t_slsqp:.3f} s, iterations: {res.nit}, objective: {res.fun:.6f}, constraint: {np.sum(res.x**2) - 0.5:.3e}")




