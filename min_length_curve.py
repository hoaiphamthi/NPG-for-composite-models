import numpy as np
from save_and_plot import load, save, make_all_plots
from alg import AdProxGrad, ProxGrad, NPG

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

seed = 1
m, n = 2000,10000
N = 50000


def run_min_len_curve(m = m, n = n, seed = seed, N = N):
    def f(x):
        return np.sqrt(1 + x[0]**2) + np.sum(np.sqrt(1 + (x[1:] - x[:-1])**2))

    def df(x):
        elements = np.insert( np.sqrt(1 + (x[1:] - x[:-1])**2),0,np.sqrt(1 + x[0]**2))
        return -np.diff(np.append(np.insert(np.diff(x),0,x[0]) * (1 / elements),0))

    def oracle_f(x):
        return f(x), df(x)

    def g(x):
        return 0.0

    def prox_g(x, alpha):
        return x - A.T.dot(P.dot(A.dot(x) - b))

    name = "min_len_curve"
    np.random.seed(seed)
    name_instance = f"{name}_m={m}_n={n}_seed={seed}"
    tol = 1e-6

    if LOAD_DATA:
        loaded_data = load(name_instance, name, saving_obj=1)
        A  = loaded_data["A"]
        b  = loaded_data["b"]
        P  = loaded_data["P"]
        x_feas = loaded_data["x_feas"]
        x0 = loaded_data["x0"]
    else:
        x_feas = np.random.randn(n)
        A = np.random.randn(m, n)
        b = A.dot(x_feas)
        P = np.linalg.inv(A.dot(A.T))
        x0 = prox_g(np.random.randn(n), 1)

    if SAVE_DATA:
        data = {"A":A, "b":b, "x0":x0,"P":P,"x_feas":x_feas}
        save(data, name_instance, name,saving_obj=1)


    def Run(algo, params = None, version = 2):
        return algo(oracle_f, g, prox_g, x0, maxit = N, tol = tol, stop = "res", lns_init = True, verbose = True, 
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

    make_all_plots(results, name_instance, name, plot=PLOT)
    parameters = {"additional_info": "", "size":f"({m}, {n})", "seed": seed}
    return results, name_instance, name, parameters 

if __name__ == "__main__":
    run_min_len_curve()