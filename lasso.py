import numpy as np
from save_and_plot import save, load, make_all_plots
from alg import AdProxGrad, ProxGrad, NPG, NPG_quad, AdaPG
import matplotlib.pyplot as plt
######################################################
LOAD_DATA = False
SAVE_DATA = True
LOAD_RESULTS = False
SAVE_RESULTS = True
PLOT = True
######################################################

seed = 2
m = 2048
n = 4*m

def run_lasso(m = m, n = n, seed = seed):
    name = "lasso"
    np.random.seed(seed)
    tol = 1e-6
    N = 15000
    tau = 0.01
    name_instance = f"{name}_n={n}_m={m}_tau={str(tau).replace('.',',')}_seed={seed}"

    if LOAD_DATA:
        loaded_data = load(name_instance, name, saving_obj=1)
        A  = loaded_data["A"]
        b  = loaded_data["b"]
        x0 = loaded_data["x0"]
        xopt = loaded_data["xopt"]

    else:
        A = np.random.randn(m,n)
        xopt = np.random.randn(n) * np.random.binomial(1, 0.05, (n))
        b = A.dot(xopt) + np.random.normal(0,0.1,m)
        x0 = np.random.normal(size=n) 

    lamda = tau*np.max(A.T @ b)

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


    def Run(algo, params = None, version = 2):
        return algo(oracle_f, g, prox_g, x0, maxit = N, tol = tol, stop = "res", lns_init = True, verbose = True, 
                                ver = version, track = ["res", "obj", "grad", "steps","time", "mse"], fixed_step = 0.001,tuning_params= params, xopt = xopt)

    def compare_signals(xopt, x):
        plt.figure(2)
        plt.clf()        
        plt.subplot(211)    
        plt.plot(xopt)
        plt.title('Original x',  fontsize=16)
        plt.subplot(212)
        plt.plot(x)
        plt.title('Reconstructed x',  fontsize=16)
        plt.tight_layout()
        plt.show()

    results = {}
    if LOAD_RESULTS:
        loaded_results = load(name_instance, name, saving_obj=2)
        for key, history in loaded_results.items():
            results[key] = history
    else:          
        x1, history1 = Run(AdProxGrad)
        results["AdPG"] = history1
        
        x2, history2 = Run(AdaPG, params=[3/2, 3/4])
        results["AdaPG" + str([3/2, 3/4])] = history2

        for param in [[1.1, 0.5], [1.2, 0.5]]:
            x3, history3 = Run(ProxGrad, param)
            results["PG-LS" + str(param)] = history3

        # In the cases where there're more than 1 set of parameters to tune, 
        # you need to add "+str(param)" to the name of algs in the dictionary results otherwise all results would not be saved correctly.
        
        for param in [[0.1, 5.7, 0.7, 0.69]]:
            x4, history4 = Run(NPG, param, version=1)
            results["NPG1"] = history4

        for param in [[0.1, 5.7, 0.99, 0.98]]:
            x5, history5 = Run(NPG, param, version=2)
            results["NPG2"] = history5

        for param in [[0.1, 5.7, 0.99, 0.98]]:
            x6, history6 = Run(NPG_quad, param)
            results["NPG_quad"] = history6

    if SAVE_RESULTS and LOAD_RESULTS == False:
        save(results,name_instance,name, saving_obj=2)

    make_all_plots(results, name_instance, name, plot_mse=True, plot=PLOT)
    if PLOT and not LOAD_RESULTS: compare_signals(xopt, x5)
    parameters = {"additional_info":f"lambda = {lamda:.2f}" , "size":f"({m}, {n})", "seed": seed}
    return results, name_instance, name, parameters 

if __name__ == "__main__":
    run_lasso()



