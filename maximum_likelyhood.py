import numpy as np
from numpy.linalg import slogdet, inv
from save_and_plot import  make_all_plots, load, save
from alg import AdProxGrad, ProxGrad, NPG
from scipy.linalg import eigh
# Control section
# - If data is not loaded then it will be created.
# - If results is not loaded then the algorithms will run.
######################################################
LOAD_DATA = False
SAVE_DATA = True
LOAD_RESULTS = False
SAVE_RESULTS = True
PLOT = True
######################################################

seed = 6
n = 100
lb = 1e-1
ub = 10
M = 50
N = 20000

def run_maximum_likelyhood(n = n, lb = lb, ub = ub, M = M, seed = seed, N = N):
    name = "maximum_likelyhood"
    np.random.seed(seed)
    name_instance =  f"max_loglh_n={n}_lb={str(lb).replace('.', ',')}_ub={str(ub).replace('.', ',')}_M={M}_seed={seed}"
    X0 = np.diag(lb * np.ones(n))
    tol = 1e-6

    if LOAD_DATA:
        loaded_data = load(name_instance, name, saving_obj=1)
        Y  = loaded_data["Y"]
    else:
        y = np.random.randn(n) * 10
        def generate_Y(y, n, M):
            Y = np.zeros((n, n))
            for _ in range(M):
                y_ = y + np.random.randn(n)
                Y += np.outer(y_, y_)
            return Y / M
        Y = generate_Y(y, n, M)

    if SAVE_DATA:
        data = {"Y":Y}
        save(data, name_instance, name,saving_obj=1)

    def f(X):
        return -slogdet(X)[1] + np.trace(X @ Y)

    def df(X):
        return -inv(X) + Y

    def oracle_f(X):
        return f(X), df(X)

    def g(X):
        return 0.0
    def Symmetric(matrix):
        return np.triu(matrix) + np.triu(matrix, 1).T
    def prox_g(X, alpha):
        la, U = eigh(Symmetric(X))
        la_ = np.clip(la, lb, ub)
        return U @ ((la_ * U).T)

    def Run(algo, params = None, version = 2):
        return algo(oracle_f, g, prox_g, X0, maxit = N, tol = tol, stop = "res", lns_init = False, verbose = True, 
                                ver = version, track = ["res", "obj", "grad", "steps","time"], fixed_step = 0.001,tuning_params= params)

    results = {}
    if LOAD_RESULTS:
        loaded_results = load(name_instance, name, saving_obj=2)
        for key, history in loaded_results.items():
            results[key] = history
    else:          
        x1, history1 = Run(AdProxGrad)
        results["AdPG"] = history1
        
        for param in [[1.1, 0.5], [1.2, 0.5]]:
            x2, history2 = Run(ProxGrad, param)
            results["PG-LS" + str(param)] = history2

        # In the cases where there're more than 1 set of parameters to tune, 
        # you need to add "+str(param)" to the name of algs in the dictionary results otherwise all results would not be saved correctly.

        for param in [[0.1, 5.7, 0.7, 0.69]]:
            x3, history3 = Run(NPG, param, version=1)
            results["NPG1"] = history3

        for param in [[0.1, 5.7, 0.99, 0.98]]:
            x4, history4 = Run(NPG, param, version=2)
            results["NPG2"] = history4

    if SAVE_RESULTS and LOAD_RESULTS == False:
        save(results,name_instance,name, saving_obj=2)

    parameters = {"additional_info":"" , "size":f"(n = {n}, l = {lb}, u = {ub}, M = {M})", "seed": seed}
    make_all_plots(results, name_instance, name, plot=PLOT)
    return results, name_instance, name, parameters 

if __name__ == "__main__":
    run_maximum_likelyhood()