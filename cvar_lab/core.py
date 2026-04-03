import numpy as np

def loss_function(weights, returns):
    """
    Equation 12: f(x, y) = -x^T * y
    """
    return (-1)*np.dot(weights,returns)


def objective_function(alpha, weights, scenarios, beta):
    q = len(scenarios)
    # Calculate loss for all scenarios at once (vector)
    losses = np.dot(scenarios, -weights) 
    
    # Calculate the 'excess' losses above alpha (vector)
    excess_losses = np.maximum(losses - alpha, 0)
    
    # CRITICAL: You must sum these to get a single scalar value
    # This represents the "Expectation" in Expected Shortfall
    return alpha + (1 / (q * (1 - beta))) * np.sum(excess_losses)