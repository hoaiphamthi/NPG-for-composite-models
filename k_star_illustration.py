import numpy as np
from save_and_plot import save, load, make_all_plots
from alg import AdProxGrad, ProxGrad, NPG, NPG_quad, AdaPG
import matplotlib.pyplot as plt
######################################################
LOAD_DATA = False
SAVE_DATA = False
LOAD_RESULTS = False
SAVE_RESULTS = True
PLOT = True
######################################################

seed = None
m = 2
n = 2

def run_lasso(m = m, n = n, seed = seed):
    name = "k_star_illustration"
    np.random.seed(seed)
    tol = 1e-12
    N = 10000
    tau = 0.01
    name_instance = "" #f"{name}_n={n}_m={m}_tau={str(tau).replace('.',',')}_seed={seed}"

    if LOAD_DATA:
        loaded_data = load(name_instance, name, saving_obj=1)
        A  = loaded_data["A"]
        b  = loaded_data["b"]
        x0 = loaded_data["x0"]
        xopt = loaded_data["xopt"]

    else:
        A =  np.diag([1,30]) 
        b = np.ones(n) 
        x0 = np.zeros(n)

    lamda = 0.1
    if SAVE_DATA:
        data = {"A":A, "b":b, "x0":x0, "xopt": xopt}
        save(data, name_instance, name,saving_obj=1)

    def f(x):
        Ax_b = A.dot(x) - b
        return 0.5*(Ax_b.T.dot(Ax_b))
    def df(x):
        return A.T.dot(A.dot(x) - b)

    def oracle_f(x):
        return f(x), df(x)

    def g(x):
        return lamda*np.sum(np.abs(x))

    def prox_g(x, stepsize):
        return np.sign(x)*np.maximum(np.zeros(np.shape(x)),np.abs(x)-stepsize*lamda)


    def Run(algo, **kwargs):
        return algo(oracle_f, g, prox_g, x0, maxit = N, tol = tol, stop = "res", lns_init = True, verbose = True, 
                    track = ["res", "obj", "grad", "steps","time"], fixed_step = 0.001,  **kwargs)

    results = {}
    if LOAD_RESULTS:
            loaded_results = load(name_instance, name, saving_obj=2)
            for key, history in loaded_results.items():
                results[key] = history
    else :  

        x, history = Run(AdProxGrad)
        results["AdPG"] = history

        x, history = Run(NPG, ver = 1, params = [0.1, 6, 0.7, 0.1], e = 1.1)
        results["NPG1"  + r' $( c_0 = 0.7, c_1 = 0.1, \gamma_k = \gamma_1$'] = history
        x, history = Run(NPG, ver = 1, params = [0.1, 6, 0.7, 0.69], e = 1.1)
        results["NPG1"  + r' $( c_0 = 0.7, c_1 = 0.69, \gamma_k = \gamma_1$'] = history
        x, history = Run(NPG, ver = 1, params = [0.1, 6, 0.7, 0.1], e = 2.5)
        results["NPG1"  + r' $( c_0 = 0.7, c_1 = 0.1, \gamma_k = \gamma_2$'] = history        
        x, history = Run(NPG, ver = 1, params = [0.1, 6, 0.7, 0.69], e = 2.5)
        results["NPG1"  + r' $( c_0 = 0.7, c_1 = 0.69, \gamma_k = \gamma_2$'] = history        


    if SAVE_RESULTS and LOAD_RESULTS == False:
        save(results,name_instance,name, saving_obj=2)

    make_all_plots(results, name_instance, name, plot=PLOT, legend=False)
    parameters = {"additional_info":f"lambda = {lamda:.2f}" , "size":f"({m}, {n})", "seed": seed}

    return results, name_instance, name, parameters 

if __name__ == "__main__":
    run_lasso()



