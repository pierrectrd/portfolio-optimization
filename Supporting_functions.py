import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import scipy.optimize as sp
import scipy
import time




#%% Fonctions fournies 

def download_finance_data(n_assets=10):
    # Date range
    start = '2016-01-01'
    end = '2019-12-30'
    # Tickers of assets
    assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'MMC', 'JPM',
              'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
                'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA']
    assets.sort()
    # Downloading data
    if n_assets>23:
        print('Warning: max number of assets is limited to 23')
        n_assets = 23
    # Training
    training_data = yf.download(assets[:n_assets], start=start, end=end, group_by="ticker")
    # Testing
    testing_data = yf.download(assets[:n_assets], start='2020-01-01', end='2020-12-30', group_by="ticker")
    Y = dict()
    dates_train = None
    # Compute the monthly returns:
    for ast in assets[:n_assets]:
        qq = training_data[ast]['Close']
        Y[ast] = [100*(qq[ii]- qq[ii-1])/qq[ii-1] for ii in range(1,len(qq))]
        if dates_train is None:
            dates_train = qq.index[1:]
    training_df = pd.DataFrame(data=Y,index=dates_train)
    Y = dict()
    dates_test = None
    # Compute the monthly returns:
    for ast in assets[:n_assets]:
        qq = testing_data[ast]['Close']
        Y[ast] = [100 * (qq[ii] - qq[ii - 1]) / qq[ii - 1] for ii in range(1, len(qq))]
        if dates_test is None:
            dates_test = qq.index[1:]
    testing_df = pd.DataFrame(data=Y,index=dates_test)
    return training_df, testing_df

def compute_moments(y_data):
    # Defining initial inputs
    mu = y_data.mean().to_numpy().reshape(1, -1)
    sigma = y_data.cov().to_numpy()
    return mu, sigma

def compute_metrics(x, training_df, testing_df):
    # Calculating Annualized Portfolio Stats
    var0 = x * (training_df.cov() @ x)
    var0 = var0.sum().to_frame().T
    std0 = np.sqrt(var0* 252)
    ret0 = training_df.mean().to_frame().T @ x * 252
    var = x * (testing_df.cov() @ x)
    var = var.sum().to_frame().T
    std = np.sqrt(var* 252)
    ret = testing_df.mean().to_frame().T @ x * 252
    stats_training  = pd.concat([ret0, std0, var0], axis=0)
    stats_testing = pd.concat([ret, std, var], axis=0)
    #stats = pd.concat([training_metrics, testing_metrics], axis=1)
    stats_training.index = ['Return', 'Std. Dev.', 'Variance']
    stats_testing.index = ['Return', 'Std. Dev.', 'Variance']
    print('Training set: 2016 -- 2019')
    print(stats_training)
    print('Testing set: 2021')
    print(stats_testing)
    return stats_training, stats_testing


def montecarlo_sim(num_assets,n_samples, Y):
    ####################################
    # Montecarlo Simulation
    ####################################
    # Montecarlo simulation of portfolio weights
    rs = np.random.RandomState(seed=123)
    s1 = rs.dirichlet([0.1] * num_assets, n_samples)
    s2 = rs.dirichlet([0.25] * num_assets, n_samples)
    s3 = rs.dirichlet([0.5] * num_assets, n_samples)
    s4 = rs.dirichlet([0.75] * num_assets, n_samples)
    s5 = rs.dirichlet([1.0] * num_assets, n_samples)
    s6 = rs.dirichlet([1.5] * num_assets, n_samples)
    s7 = rs.dirichlet([2.0] * num_assets, n_samples)
    s8 = rs.dirichlet([3.0] * num_assets, n_samples)
    sample = np.concatenate([np.identity(num_assets), s1, s2, s3, s4, s5, s6, s7, s8], axis=0)
    # Calculating mean, standard deviation and square root kurtosis of each portfolio
    m = sample.shape[0]
    M_1 = np.mean(Y.to_numpy(), axis=0).reshape(1, -1)
    M_2 = Y.cov().to_numpy()
    c_mean = 252 * M_1 @ sample.T
    c_var = np.zeros(m)
    #c_kurt = np.zeros(m)
    for i in range(0, m):
        c_var[i] =  (252 * sample[i] @ M_2 @ sample[i].T) ** (0.5)
        #c_kurt[i] = (np.kron(sample[i], sample[i]) @ Sigma_4 @ np.kron(sample[i], sample[i]).T) ** (1 / 4)
    return c_mean, c_var


def scatter_plot_port(c_mean, c_var, ret, std,title):
    ####################################
    # Plotting Portfolios
    ####################################
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax = np.ravel(ax)
    # Plotting Portfolios in mean-standard deviation plane
    cax0 = ax[0].scatter(c_var, c_mean, c=c_mean / c_var, cmap='Spectral')
    ax[0].scatter(std, ret, marker='*', s=2 ** 9, color='tab:red', label='Computed solution')
    ax[0].set_title(title, fontsize=24)
    ax[0].set_xlabel('Standard deviation [%]', fontsize=22)
    ax[0].set_ylabel('Return [%]', fontsize=22)
    ax[0].tick_params(axis='both', labelsize=22)
    ax[0].grid(True)
    ax[0].legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f'{title}.pdf', bbox_inches='tight')
    plt.show()
    return


def portfolio_composition_plot(Y,x,title):
    ####################################
    # Plotting Portfolios Composition
    ####################################
    import riskfolio as rp
    # Building the portfolio object
    df = dict()
    for i, elements in enumerate(Y):
        df[elements] = x[i]
    dw = pd.DataFrame(data=df)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax = np.ravel(ax)
    rp.plot_pie(w=dw, title=title, others=0.05, nrow=25, ax=ax[0])
    # Move the legend to the right, outside the plot
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, title='Assets')
    ax[0].tick_params(axis='both', labelsize=15)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'{title}.pdf', bbox_inches='tight')
    plt.show()
    
    
def compute_coefficients(Y, n):
    ####################################
    # Auxiliary functions
    ####################################
    # Function that calculates D_2
    def duplication_matrix(n):
        out = np.zeros((int(n * (n + 1) / 2), n ** 2))
        for j in range(1, n + 1):
            for i in range(j, n + 1):
                u = np.zeros((int(n * (n + 1) / 2), 1))
                u[round((j - 1) * n + i - ((j - 1) * j) / 2) - 1] = 1.0
                E = np.zeros((n, n))
                E[i - 1, j - 1] = 1.0
                E[j - 1, i - 1] = 1.0
                out += u @ E.reshape(-1, 1).T
        return out.T
    # Function that calculates L_2
    
    
    
    def duplication_elimination_matrix(n):
        out = np.zeros((int(n * (n + 1) / 2), n ** 2))
        for j in range(n):
            e_j = np.zeros((1, n))
            e_j[0, j] = 1.0
            for i in range(j, n):
                u = np.zeros((int(n * (n + 1) / 2), 1))
                row = round(j * n + i - ((j + 1) * j) / 2)
                u[row] = 1.0
                e_i = np.zeros((1, n))
                e_i[0, i] = 1.0
                out += np.kron(u, np.kron(e_j, e_i))
        return out
    # Function that calculates S_4
    def kurt_matrix(Y):
        P = Y.to_numpy()
        T, n = P.shape
        mu = np.mean(P, axis=0).reshape(1, -1)
        mu = np.repeat(mu, T, axis=0)
        x = P - mu
        ones = np.ones((1, n))
        z = np.kron(ones, x) * np.kron(x, ones)
        S4 = 1 / T * z.T @ z
        return S4
    return duplication_matrix(n), duplication_elimination_matrix(n), kurt_matrix(Y)


#%% Fonctions du TP


def markovitz_portfolio(mu, sigma):
    n = len(mu[0])
    x = cp.Variable((n,1))
    rmin = 1.6/252

    obj = cp.Minimize(cp.quad_form(x, sigma))

    constraints = [mu @ x >= rmin, x >= 0, cp.sum(x) == 1]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return x.value


def markovitz_portfolio_probabilistic(mu, sigma,phi,alpha,beta):
    n = len(mu[0])
    x = cp.Variable((n,1))
    rmin = 1.6/252
    obj = cp.Minimize(cp.quad_form(x, sigma))

    constraints = [mu @ x +phi*cp.quad_form(x, sigma) >= alpha, x >= 0, cp.sum(x) == 1]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return x.value


def kurtosis_solve(L2, S2, S4, mu): #Avec SLSQP
    mu = np.array(mu).flatten()
    n = mu.shape[0]
    p = n * (n + 1) // 2  
    

    Sigma_4_sqrt = sqrtm(S2 @ S4 @ S2.T)

    rmin = 1.6 / 252.0      



    def split_vars(v):
        # v = [x, z, g, X.flatten()]
        x = v[:n]
        z = v[n:n+p]
        g = v[n+p]
        X = v[n+p+1:].reshape((n, n))
        return x, z, g, X


    def kurtosis_objective(v):
        x, z, g, X = split_vars(v)
        return g


    def constr_sum_to_one(v):
        x, z, g, X = split_vars(v)
        return np.sum(x) - 1.0

    def constr_min_return(v):
        x, z, g, X = split_vars(v)
        return float(mu @ x - rmin)

    def cons_z(v):
        x, z, g, X = split_vars(v)
        return g - np.linalg.norm(Sigma_4_sqrt @ z)

    def L2z(v):
        x, z, g, X = split_vars(v)
        return z - L2 @ X.flatten()

    def XXx(v):
        x, z, g, X = split_vars(v)
        return (np.outer(x, x) - X).flatten()

    # --- Définition des contraintes SLSQP ---
    constraints = [
        {'type': 'eq',  'fun': constr_sum_to_one},
        {'type': 'ineq','fun': constr_min_return},
        {'type': 'ineq','fun': cons_z},
        {'type': 'eq',  'fun': L2z},
        {'type': 'eq',  'fun': XXx}
    ]

    bounds = [(0.0, None)] * n + [(None, None)] * (p + 1 + n*n)

    np.random.seed(1)
    x0 = np.random.dirichlet(np.ones(n))
    z0 = np.zeros(p)
    g0 = 0.0
    X0 = np.outer(x0, x0)
    v0 = np.concatenate([x0, z0, [g0], X0.flatten()])

 
    res = sp.minimize(
        fun=kurtosis_objective,
        x0=v0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    x_star, z_star, g_star, X_star = split_vars(res.x)
    
    return x_star, z_star, g_star, X_star
    
    
    
    
def kurtosis_scp_solve(L2, S2, S4, mu, x0): #avec CXSVP
    feas=[]
    g=[]
    n = len(mu[0])
    rmin = 1.6/252
    max_iter = 200
    Sigma_4_sqrt = sqrtm(S2 @ S4 @ S2.T)

    
    xk = x0
    Xk = np.outer(xk, xk)
    zk = L2@Xk.reshape(n*n,1)
    
    gk = 0.0
    for k in range(max_iter):
        dx = cp.Variable(n)
        dX = cp.Variable((n, n),symmetric=True)
        dz = cp.Variable((zk.shape[0],1))
        dg = cp.Variable()

        x_col = xk.reshape(-1, 1)
        dx_col = cp.reshape(dx, (n, 1))

        objective = cp.Minimize(gk + dg)

        constraints = [
        Xk + dX == x_col @ x_col.T + x_col @ dx_col.T + dx_col @ x_col.T,
        zk + dz == L2 @ cp.reshape(cp.vec(Xk + dX), (n*n, 1)),
        cp.norm(Sigma_4_sqrt @ (zk + dz), 2) <= gk + dg,
        cp.sum(xk + dx) == 1,
        xk + dx >= 0,
        mu @ (xk + dx) >= rmin,
        Xk + dX >> 0
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        dx_val = np.array(dx.value).reshape(-1)
        dX_val = np.array(dX.value)
        dz_val = np.array(dz.value)
        dg_val = float(dg.value)

        add = dx_val
        xk = xk + dx_val
        Xk = Xk + dX_val
        zk = zk + dz_val
        gk = gk + dg_val


        feas.append(np.linalg.norm(add))
        g.append(gk)



        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Problème QP non résolu à l'itération {k} (status={prob.status})")
            break
        if dx.value is None or dX.value is None or dz.value is None or dg.value is None:
            print(f"Problème QP non résolu à l'itération {k} (valeurs None)")
            break

        if np.linalg.norm(add) < 1e-6:
            print(f"Convergence atteinte à l'itération {k}")
            break
    return xk, zk, gk, Xk,feas,g




