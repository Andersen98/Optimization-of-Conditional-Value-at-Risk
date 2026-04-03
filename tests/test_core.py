import numpy as np
import pytest
from cvar_lab.core import loss_function, objective_function

def test_loss_function_simple():
    """
    Test that loss_function correctly calculates -x^T * y.
    If weights = [0.5, 0.5] and returns = [0.1, -0.1], loss should be 0.
    """
    weights = np.array([0.5, 0.5])
    returns = np.array([0.1, -0.1])
    expected_loss = 0.0
    
    assert loss_function(weights, returns) == pytest.approx(expected_loss)

def test_loss_function_negative_return():
    """
    If the market crashes (returns = -0.2), the loss should be positive.
    """
    weights = np.array([1.0])
    returns = np.array([-0.2])
    # - (1.0 * -0.2) = 0.2
    assert loss_function(weights, returns) == pytest.approx(0.2)

def test_objective_function_convexity_point():
    """
    Verify a single point of Equation 9.
    If alpha is very high, the max(loss - alpha, 0) term should be 0.
    """
    scenarios = np.array([[0.01, 0.02], [0.01, 0.02]]) # 2 identical scenarios
    weights = np.array([0.5, 0.5])
    beta = 0.95
    alpha = 10.0 # Extremely high VaR
    
    # F_beta should just equal alpha because the sum part is 0
    val = objective_function(alpha, weights, scenarios, beta)
    assert val == pytest.approx(alpha)

def test_objective_function_calculation():
    """
    Manual check of F_beta.
    Scenario loss = 0.1, alpha = 0.05, beta = 0.0 (for simplicity)
    F = 0.05 + (1 / (1 * (1-0))) * max(0.1 - 0.05, 0) = 0.1
    """
    scenarios = np.array([[-0.1]]) # 10% loss
    weights = np.array([1.0])
    alpha = 0.05
    beta = 0.0
    
    expected = 0.1
    val = objective_function(alpha, weights, scenarios, beta)
    assert val == pytest.approx(expected)