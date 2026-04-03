from scipy.optimize import minimize
from .core import objective_function
import numpy as np

def minimize_cvar(scenarios, beta=0.95):
    num_assets = scenarios.shape[1]
    init_params = np.append(np.ones(num_assets)/num_assets, [0.0])
    
    # 1. Budget Constraint: Sum of weights (everything but alpha) must be 1.0
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x[:-1]) - 1.0})
    
    # 2. Bounds: Each weight between 0 and 1; Alpha is unconstrained
    # (None, None) for the last element allows alpha to be any real number
    bounds = [(0, 1) for _ in range(num_assets)] + [(None, None)]
    
    def total_objective(params):
        weights = params[:-1]
        alpha = params[-1]
        return objective_function(alpha, weights, scenarios, beta)

    # Adding 'bounds' and 'constraints' to the call
    res = minimize(total_objective, init_params, method='SLSQP', 
                   bounds=bounds, constraints=cons)
    return res