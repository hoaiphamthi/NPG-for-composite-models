import numpy as np
import time
import warnings

# Suppress RuntimeWarning: divide by zero
warnings.filterwarnings("ignore", category=RuntimeWarning)


def mse(x1,x2):
    return np.mean((x1 - x2)**2)

def linesearch_initial(oracle_f, g, prox_g, x0, alpha, incr = 10, dcr = 0.5, veryLargeSize = 10):
    obj_x, grad_fx0 = oracle_f(x0)
    largestep = True
    for i in range(1, 101):
        x1 = prox_g(x0 - alpha * grad_fx0, alpha)
        obj_x1, grad_fx1 = oracle_f(x1)
        if i == 1 and np.linalg.norm(x0 - x1) < 10**-6: 
            print("Congrats: initial x0 is a solution")
        else:
            L = np.linalg.norm(grad_fx1 - grad_fx0) / np.linalg.norm(x1 - x0)
            if alpha * L > 2:
                largestep = False
                alpha *= dcr
            else:
                if alpha * L <= 2 and largestep:
                    alpha *= incr
                    if alpha > veryLargeSize:
                        return alpha, x1, grad_fx0, grad_fx1, obj_x, obj_x1, i
                else:
                    return alpha, x1, grad_fx0, grad_fx1, obj_x, obj_x1, i

def linesearch(oracle_f, prox_g, x, fx, grad_fx, alpha, counter, inc=1.2, dcr=0.5):
    alpha *= inc
    for i in range(1, 1001):
        counter += 1
        x1 = prox_g(x - alpha * grad_fx, alpha)
        if np.linalg.norm(x1 - x) < 10**-20 or alpha < 10**-20:
            return x, fx, grad_fx, alpha, counter
        fx1, grad_fx1 = oracle_f(x1)
        if fx1 <= fx + np.sum(grad_fx * (x1 - x))  + 1.0 / (2 * alpha) * np.linalg.norm(x1 - x)**2: #np.dot(grad_fx, x1 - x)
            x, fx, grad_fx = x1, fx1, grad_fx1
            break
        else:
            alpha *= dcr
    return x, fx, grad_fx, alpha, counter

def collect_history(history_dic, data_dic):
    for key in history_dic.keys():
        history_dic[key].append(data_dic[key])
    return history_dic

def AdProxGrad(oracle_f, g, prox_g, x0, maxit=1000, tol=1e-9, stop="res", lns_init=True, verbose=False, ver=2, track=["res", "obj", "grad", "steps","time"], fixed_step=1e-2,tuning_params =None, xopt = 0):
    def obj(fx, x):
        return fx + g(x)
    start = time.perf_counter()
    
    x_prev = x0
    theta = 1.0 / 3
    
    if lns_init:
        alpha_prev, x, grad_prev, grad_fx, f_prev, fx, lns_iter = linesearch_initial(oracle_f, g, prox_g, x0, fixed_step, veryLargeSize=0.9)
        if verbose:
            print(f"Linesearch found initial stepsize {alpha_prev} in {lns_iter} iterations")
    else:
        alpha_prev = fixed_step
        if verbose:
            print("No linesearch, initial stepsize is ", alpha_prev)
        f_prev, grad_prev = oracle_f(x_prev)
        x = prox_g(x_prev - alpha_prev * grad_prev, alpha_prev)
        fx, grad_fx = oracle_f(x)
    dict_values = {
        "res": [ np.linalg.norm(x - x_prev)], 
        "obj": [ obj(fx, x)],
        "grad": [np.linalg.norm(grad_fx)],
        "steps": [ alpha_prev],
        "time":[0],
        "mse":[mse(x, xopt)]
    }
    history_dic = {key: [] for key in track}
    history_dic = {k: v for k, v in dict_values.items() if k in track}
    for i in range(1, maxit):
        if ver == 1:
            L = np.linalg.norm(grad_fx - grad_prev) / np.linalg.norm(x - x_prev)
            alpha = min(np.sqrt(1 + theta) * alpha_prev, 1 / (np.sqrt(2) * L))
        elif ver == 2:
            L = np.linalg.norm(grad_fx - grad_prev) / np.linalg.norm(x - x_prev)
            alpha = min(np.sqrt(2 / 3 + theta) * alpha_prev, alpha_prev / np.sqrt(max(2 * alpha_prev**2 * L**2 - 1, 0)))
        elif ver == 3:
            alpha = fixed_step
        theta = alpha / alpha_prev
        x_prev, grad_prev, alpha_prev = x, grad_fx, alpha
        x = prox_g(x - alpha * grad_fx, alpha)
        end = time.perf_counter()

        residual = np.linalg.norm(x_prev - x) 
        fx, grad_fx = oracle_f(x)
        current_info = {"res": residual, "obj": obj(fx, x), "grad": np.linalg.norm(grad_fx), "steps": alpha,"time":end-start, "mse":mse(x, xopt)}
        history_dic = collect_history(history_dic, current_info)
        if current_info[stop] <= tol:
            if verbose:
                print(f"The AdPG algorithm reached required accuracy in {i+1} iterations")
            break
    if current_info[stop] > tol:
        if verbose:
            print(f"The AdPG algorithm terminated without reaching required accuracy")
    return x, history_dic

def NPG(oracle_f, g, prox_g, x0, maxit=1000, tol=1e-9, stop="res", lns_init=True, verbose=False, ver=2, track=["res", "obj", "grad", "steps","time"], fixed_step=1e-2, tuning_params = [0.9,5,0.5,0.3], xopt = 0):
    def obj(fx, x):
        return fx + g(x)
    def gamma_sequence(k):
        a = tuning_params[0]
        b = tuning_params[1]
        return (a * np.log(k+1)**b ) / (k+1)**1.1
    
    start = time.perf_counter()
    c0 = tuning_params[2]
    c1 = tuning_params[3]
    x_prev = x0

    if lns_init  :
        alpha_prev, x, grad_prev, grad_fx, f_prev, fx, lns_iter = linesearch_initial(oracle_f, g, prox_g, x0, fixed_step, veryLargeSize=10)
        alpha_prev_prev = alpha_prev
        if verbose:
            print(f"Linesearch found initial stepsize {alpha_prev} in {lns_iter} iterations")
    else:
        alpha_prev = alpha_prev_prev = fixed_step
        f_prev, grad_prev = oracle_f(x_prev)
        x = prox_g(x_prev - alpha_prev * grad_prev, alpha_prev)
        fx, grad_fx = oracle_f(x)
        if verbose:
            print("No linesearch, initial stepsize is ", alpha_prev)
    dict_values = {
        "res": [np.linalg.norm(x - x_prev)],
        "obj": [obj(fx, x)],
        "grad": [np.linalg.norm(grad_fx)],
        "steps": [alpha_prev],
        "time":[0],
        "mse":[mse(x, xopt)]
    }
    history_dic = {key: [] for key in track}
    history_dic = {k: v for k, v in dict_values.items() if k in track}

    for i in range(1, maxit):
        L =  np.linalg.norm(grad_fx - grad_prev) / np.linalg.norm(x - x_prev) 
        if L > c0 / alpha_prev :
            alpha = c1 / L
        else:
            gamma_prime = gamma_sequence(i-1)
            if alpha_prev / alpha_prev_prev < 1:
                gamma_prime  = np.min([gamma_prime, np.sqrt(1 + alpha_prev / alpha_prev_prev)-1])
            alpha = (1 + gamma_prime) * alpha_prev

        x_prev, grad_prev, alpha_prev,alpha_prev_prev = x, grad_fx, alpha, alpha_prev_prev
        x = prox_g(x - alpha * grad_fx, alpha)
        end = time.perf_counter()
        residual = np.linalg.norm(x_prev - x) 
        fx, grad_fx = oracle_f(x)
        current_info = {"res": residual, "obj": obj(fx, x), "grad": np.linalg.norm(grad_fx), "steps": alpha, "time":end-start, "mse":mse(x, xopt)}
                
        history_dic = collect_history(history_dic, current_info)
        if current_info[stop] <= tol:
            if verbose:
                print(f"The NPG{ver} algorithm reached required accuracy in {i+1} iterations")
            break
    if current_info[stop] > tol:
        if verbose:
            print(f"The NPG{ver} algorithm terminated without reaching required accuracy")    
    return x, history_dic

def NPG_quad(oracle_f, g, prox_g, x0, maxit=1000, tol=1e-9, stop="res", lns_init=True, verbose=False, ver=2, track=["res", "obj", "grad", "steps","time"], fixed_step=1e-2, tuning_params = [0.1,5.7,0.99,0.98], xopt = 0):
    def obj(fx, x):
        return fx + g(x)
    def gamma_sequence(k):
        a = tuning_params[0]
        b = tuning_params[1]
        return (a * np.log(k+1)**b ) / (k+1)**1.1
    
    start = time.perf_counter()
    c0 = tuning_params[2]
    c1 = tuning_params[3]
    x_prev = x0

    if lns_init  :
        alpha_prev, x, grad_prev, grad_fx, f_prev, fx, lns_iter = linesearch_initial(oracle_f, g, prox_g, x0, fixed_step, veryLargeSize=0.9)
        alpha_prev_prev = alpha_prev
        if verbose:
            print(f"Linesearch found initial stepsize {alpha_prev} in {lns_iter} iterations")
    else:
        alpha_prev = alpha_prev_prev = fixed_step
        f_prev, grad_prev = oracle_f(x_prev)
        x = prox_g(x_prev - alpha_prev * grad_prev, alpha_prev)
        fx, grad_fx = oracle_f(x)
        if verbose:
            print("No linesearch, initial stepsize is ", alpha_prev)
    dict_values = {
        "res": [np.linalg.norm(x - x_prev)],
        "obj": [obj(fx, x)],
        "grad": [np.linalg.norm(grad_fx)],
        "steps": [alpha_prev],
        "time":[0],
        "mse":[mse(x, xopt)]
    }
    history_dic = {key: [] for key in track}
    history_dic = {k: v for k, v in dict_values.items() if k in track}

    for i in range(1, maxit):
        L = np.sum((x - x_prev).T @ (grad_fx - grad_prev)) / np.linalg.norm(x - x_prev)**2 
        if L > c0/alpha_prev :
            alpha = c1 / L
        else:
            gamma_prime = gamma_sequence(i-1)
            if alpha_prev / alpha_prev_prev < 1:
                gamma_prime  = np.min([gamma_prime, np.sqrt(1 + alpha_prev / alpha_prev_prev)-1])
            alpha = (1 + gamma_prime) * alpha_prev

        x_prev, grad_prev, alpha_prev,alpha_prev_prev = x, grad_fx, alpha, alpha_prev_prev
        x = prox_g(x - alpha * grad_fx, alpha)
        end = time.perf_counter()
        residual = np.linalg.norm(x_prev - x)
        fx, grad_fx = oracle_f(x)
        current_info = {"res": residual, "obj": obj(fx, x), "grad": np.linalg.norm(grad_fx), "steps": alpha, "time":end-start, "mse":mse(x, xopt)}
                
        history_dic = collect_history(history_dic, current_info)
        if current_info[stop] <= tol:
            if verbose:
                print(f"The NPG-quad algorithm reached required accuracy in {i+1} iterations")
            break
    if current_info[stop] > tol:
        if verbose:
            print(f"The NPG-quad algorithm terminated without reaching required accuracy")    
    return x, history_dic

def ProxGrad(oracle_f, g, prox_g, x0, maxit=1000, tol=1e-9, stop="res", lns_init=True, verbose=False, ver=2, track=["res", "obj", "grad", "steps","time"], fixed_step=1e-2, tuning_params = [1.2,0.5], xopt = 0):
    def obj(fx, x):
        return fx + g(x)
    
    start = time.perf_counter()
    inc, dcr = tuning_params[0],tuning_params[1]
    total = 0
    x_prev = x0
    
    if lns_init:
        alpha_prev, x, grad_prev, grad_fx, f_prev, fx, lns_iter = linesearch_initial(oracle_f, g, prox_g, x0, fixed_step, veryLargeSize=0.9)
        if verbose:
            print(f"Linesearch found initial stepsize {alpha_prev} in {lns_iter} iterations")
    else:
        alpha_prev = fixed_step
        if verbose:
            print("No linesearch, initial stepsize is ", alpha_prev)
        f_prev, grad_prev = oracle_f(x_prev)
        x = prox_g(x_prev - alpha_prev * grad_prev, alpha_prev)
        fx, grad_fx = oracle_f(x)
    
    dict_values = {
        "res": [np.linalg.norm(x - x_prev)],
        "obj": [obj(fx, x)],
        "grad": [np.linalg.norm(grad_fx)],
        "steps": [alpha_prev],
        "time":[0],
        "mse":[mse(x, xopt)]
    }
    history_dic = {key: [] for key in track}
    history_dic = {k: v for k, v in dict_values.items() if k in track}
    i = 0
    for _ in range(1, maxit):
        i += 1

        x, fx, grad_fx, alpha, total = linesearch(
            oracle_f,
            prox_g,
            x,
            fx,
            grad_fx,
            alpha_prev,
            total,
            inc=inc,
            dcr=dcr,
        )
        end = time.perf_counter()
        residual = np.linalg.norm(x_prev - x)
        x_prev, alpha_prev = x, alpha
        current_info = {"res": residual, "obj": obj(fx, x), "grad": np.linalg.norm(grad_fx), "steps": alpha,"time":end-start, "mse":mse(x, xopt)}
        history_dic = collect_history(history_dic, current_info)
        if current_info[stop] <= tol:
            if verbose:
                print(f"The PG-LS{tuning_params} algorithm reached required accuracy in {i+1} iterations.")
            break
    if current_info[stop] > tol:
        if verbose:
            print(f"The PG-LS{tuning_params} algorithm terminated without reaching required accuracy")
    return x, history_dic