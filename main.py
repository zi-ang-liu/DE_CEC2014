from DE import DE 
import numpy as np
import opfunu

N_RUN = 5    # number of run
N_FUNC = 30   # number of function  
N_DIM = 10     # number of dimension
N_GEN = 1000    # number of generation
UB = 100    # upper bound
LB = -100   # lower bound
N_POP = 20 # number of population

results = np.zeros((N_FUNC, N_RUN))

for i in range(N_FUNC):
    # get function
    func_name = "F" + str(i+1) + "2014"
    funcs = opfunu.get_functions_by_classname(func_name)
    func = funcs[0](ndim=10)

    for j in range(N_RUN):
        # run DE
        optimizer = DE(problem=func, n_dim=N_DIM, ub=UB, lb=LB, n_gen=N_GEN, n_pop=N_POP)
        results[i,j] = optimizer.optimize() - func.f_global

print(np.mean(results,axis=1))