This is the instruction file for the code of paper "COMPOSITE OPTIMIZATION MODELS VIA PROXIMAL GRADIENT METHOD WITH NOVEL ENHANCED ADAPTIVE STEPSIZES" by Pham Thi Hoai and Nguyen Pham Duy Thai

### Introduction
- There're seven python files named by corresponding tested instances. 
- The "run_experiment.py" file is designed to run many problems, sizes, random datasets all at once.
- In the main function call the function correspond to the problem that you want to run for example : "experiment_lasso()", sizes and random seeds need to be specified inside each experiment function.

- Each *.py file is started with the control section, which need to be correctly set up:
    LOAD_DATA: if True, the code would find the dataset and use that data to run the algorithms, if False, then data would be created.
    SAVE_DATA: if True the data would be saved, noticed that sometimes data could be large and cost a significant amount of memory.
    LOAD_RESULTS: if True the results of already run and saved problem would be used to show, if False the algorithms would run and the show the results.
    SAVE_RESULTS: if True the rersults of already run problems would be saved, this one is highly recommended to be set True. 

- Initial stepsize is set to be using linesearch with problem lasso, min_length_curve, BCQP and NMF, and it's fixed at 0.001 for problems maximum_likelyhood, dual_max_entropy, BCFP.

- We use numpy.random.seed as the mechanism to create re-producible results, our datasets 1-10 correspond to random seed 1-10. Problem with same size and random seed should always produce identical results.

### Running the experiments
- To obtain Figure 1, run the file k_star_illustration.py
- To obtain Figure 2-8, uncomment the respective function call in the function main() and run the file run_experiment.py
- To obtain Figure 9, run the single files correspond to the problem (e.g. lasso.py, bcqp.py, ...)

### Experimental results
- The plots generated are saved in the plots folder
- The folder results contains the detailed results (in the format of the .npy files) of all the experiments.
- The file Results_NPG.xlxs contains the final results of all the experiments.

### Experimental details
For the figure 9 in the paper, the detailed sizes and random seeds for each problem are provided here:
- Lasso                       : m = 1024, n = 8192, seed = 1   
- Min length curve problem    : m = 100, n = 10000, seed = 1 
- Maximum likelyhood problem  : n = 30, lb = 0.1, ub = 1000, M = 50, seed = 1
- Dual max entropy problem    : m = 4000, n = 5000, seed = 1
- BCQP                        : n = 1000, r = 10, seed = 1
- BCFP                        : n = 5000, r = 10, seed = 1
- NMF problem                 : m = 3000, n = 3000, r = 30, seed = 1
    
