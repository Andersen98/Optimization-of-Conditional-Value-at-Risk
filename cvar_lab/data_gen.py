import numpy as np
from scipy.stats import multivariate_t

def generate_scenarios(num_scenarios=1000, num_assets=5, df=3, seed=42):
    """
    Generates synthetic asset returns using a Multivariate Student-T Distribution.
    
    Parameters:
    - num_scenarios: Number of market states to simulate (rows).
    - num_assets: Number of instruments in the portfolio (columns).
    - df: Degrees of Freedom. Lower values (e.g., 3-5) create "fatter tails" 
          representing higher risk of extreme events.
    - seed: Random seed for reproducibility.
    
    Returns:
    - scenarios: A (num_scenarios, num_assets) numpy array of returns.
    """
    np.random.seed(seed)
    
    # 1. Define Mean Returns (e.g., slight positive drift for each asset)
    means = np.random.uniform(0.0005, 0.002, num_assets)
    
    # 2. Define a Correlation Matrix
    # We create a random positive semi-definite matrix to represent asset dependencies.
    random_matrix = np.random.randn(num_assets, num_assets)
    cov_matrix = np.dot(random_matrix, random_matrix.T) * 0.001
    
    # 3. Generate Scenarios
    # The multivariate_t distribution captures the 'coupling' between asset failures.
    scenarios = multivariate_t.rvs(loc=means, shape=cov_matrix, df=df, size=num_scenarios)
    
    return scenarios

if __name__ == "__main__":
    # Test generation
    data = generate_scenarios(5, 3)
    print("Sample Scenarios (First 5):")
    print(data)