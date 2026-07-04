
from Supporting_functions import *
import scipy

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy.optimize as sp
import pandas as pd
import yfinance as yf
import time

#Implementation of the sqp method on the simple Non convex problem 


n = 20
m = 30
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)




# Fonctions pour SQP
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


max_iter=50

def SQP(x0,f,grad_f,hess_f,g,Jg,max_iter): #ici pas de h car pas de contraintes d'egalité
    x = x0.copy()
    n = x.size

        
    constraints = []


    residuals = []
    feas = []
    for k in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        g_val = g(x)
        Jg_val = Jg(x)
        dx = cp.Variable(n)
        constraints = [g_val + Jg_val @ dx <= 0]
        obj = cp.Minimize(0.5 * cp.quad_form(dx, hess) + grad @ dx)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        if dx.value is None:
            print(f"QP problem not solved at iteration {k}")
            break
        x = x + dx.value
        residuals.append(np.linalg.norm(dx.value))
        feas.append(np.max(g(x)))
        if np.linalg.norm(dx.value) < 1e-6:
            print(f"Convergence reached at iteration {k}")
            break
    return x, residuals, feas,k


x0 = np.clip(np.random.rand(n), 0, 1)


x_sol, residuals, feas,k = SQP(x0, f,grad_f,hess_f,g,Jg,max_iter=max_iter)



plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(residuals, marker='o')
plt.yscale('log')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Step norm', fontsize=16)
plt.title('SQP convergence', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(feas, marker='x')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Max constraint (should be <=0)', fontsize=12)
plt.title('Constraint satisfaction', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.savefig('SQP methode.pdf',bbox_inches='tight')

print("\nFinal constraints (should be <= 0):\n", g(x_sol))
print("\nFinal objective value:\n", f(x_sol))