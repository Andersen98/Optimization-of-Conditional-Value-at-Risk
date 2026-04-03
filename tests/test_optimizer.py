import numpy as np
import pytest
from cvar_lab.optimizer import minimize_cvar

def test_optimizer_constraints():
    """
    Verify the budget constraint: sum(weights) == 1.0.
    Regardless of the asset returns, the optimizer must allocate 100% of capital.
    """
    # 100 scenarios for 3 assets
    scenarios = np.random.normal(0.01, 0.05, (100, 3))
    result = minimize_cvar(scenarios, beta=0.95)
    
    # Extract weights (everything except the last parameter alpha)
    weights = result.x[:-1]
    assert np.sum(weights) == pytest.approx(1.0, rel=1e-5)

def test_optimizer_bounds():
    """
    Verify the 'no-short-selling' constraint: weights >= 0.
    """
    scenarios = np.random.normal(0.01, 0.05, (100, 4))
    result = minimize_cvar(scenarios, beta=0.95)
    
    weights = result.x[:-1]
    # Check that all weights are non-negative
    assert np.all(weights >= -1e-7)

def test_optimizer_convergence():
    """
    Verify that the solver reports a successful optimization.
    """
    scenarios = np.random.normal(0.01, 0.05, (500, 5))
    result = minimize_cvar(scenarios, beta=0.95)
    
    assert result.success is True
    assert result.message is not None

def test_optimizer_single_asset():
    """
    Edge Case: In a single-asset portfolio, the weight must be 1.0.
    """
    scenarios = np.random.normal(0.01, 0.05, (100, 1))
    result = minimize_cvar(scenarios, beta=0.95)
    
    weights = result.x[:-1]
    assert weights[0] == pytest.approx(1.0)

def test_optimizer_deterministic_input():
    """
    If one asset significantly outperforms others with zero volatility,
    the optimizer should (theoretically) tilt heavily toward it.
    """
    num_scenarios = 100
    # Asset 0 is 'perfect' (10% return every time), others are risky
    scenarios = np.zeros((num_scenarios, 3))
    scenarios[:, 0] = 0.10
    scenarios[:, 1] = np.random.normal(-0.05, 0.2, num_scenarios)
    scenarios[:, 2] = np.random.normal(-0.05, 0.2, num_scenarios)
    
    result = minimize_cvar(scenarios, beta=0.95)
    weights = result.x[:-1]
    
    # Asset 0 should have the highest weighting
    assert weights[0] > weights[1]
    assert weights[0] > weights[2]