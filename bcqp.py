import numpy as np
from save_and_plot import save, load, make_all_plots
from alg import AdProxGrad, ProxGrad, NPG, NPG_quad, AdaPG
import matplotlib.pyplot as plt

# Control section
# - If data is not loaded then it will be created.
# - If results is not loaded then the algorithms will run.
######################################################
LOAD_DATA = False
SAVE_DATA = False
LOAD_RESULTS = True
SAVE_RESULTS = True
PLOT = False
######################################################

seed = 1
n = 1000  
r = 10
N = 15000
def run_bcqp(n=n, r = r, seed=seed, N = N):
    name = "bcqp"
    np.random.seed(seed)
    tol = 1e-6
    name_instance = f"{name}_n={n}_r = {r}_seed={seed}"

    if LOAD_DATA:
        loaded_data = load(name_instance, name, saving_obj=1)
        Q = loaded_data["Q"]
        c = loaded_data["c"]
        x0 = loaded_data["x0"]
    else:
        U = np.random.randn(n, n)
        Q = U.T @ np.diag(np.random.uniform(-1, r, n)) @ U  # Indefinite matrix with eigenvalues in [-1, r]
        c = np.random.randn(n)
        x0 = np.zeros(n)

    if SAVE_DATA:
        data = {"Q": Q, "c": c, "x0": x0}
        save(data, name_instance, name, saving_obj=1)

    # ============================================================
    # Define objective, gradient and proximal operator
    # ============================================================
    def f(x):
        return 0.5 * x.T @ Q @ x + c.T @ x

    def df(x):
        return Q @ x + c

    def oracle_f(x):
        return f(x), df(x)

    def g(x):
        return 0 

    def prox_g(x, stepsize):
        """Projection onto the box [-1, 1]^n"""
        return np.clip(x, -1, 1)

    def Run(algo, **kwargs):
        return algo(oracle_f, g, prox_g, x0, maxit=N, tol=tol, stop="res",mlns_init=True, verbose=False,
            track=["res", "obj", "grad", "steps", "time"],mfixed_step=1e-3,  **kwargs)

    results = {}
    if LOAD_RESULTS:
        loaded_results = load(name_instance, name, saving_obj=2)
        for key, history in loaded_results.items():
            results[key] = history
    else:
        x2, h2 = Run(NPG, params=[0.1, 5.7, 0.99, 0.98], ver=2)
        results["NPG2"] = h2

        x3, h3 = Run(NPG_quad, params=[0.1, 5.7, 0.99, 0.98])
        results["NPG_quad"] = h3

        x4, h4 = Run(AdProxGrad)
        results["AdPG"] = h4

        x5, h5 = Run(AdaPG, params=[3/2, 3/4])
        results["AdaPG"] = h5

        for param in [[1.1, 0.5], [1.2, 0.5]]:
            x6, h6 = Run(ProxGrad, params=param)
            results["PG-LS" + str(param)] = h6

    if SAVE_RESULTS and not LOAD_RESULTS:
        save(results, name_instance, name, saving_obj=2)

    make_all_plots(results, name_instance, name, plot=PLOT)
    parameters = {"additional_info": f"", "size": f"({n}, {r})", "seed": seed}
    return results, name_instance, name, parameters


if __name__ == "__main__":
    run_bcqp()