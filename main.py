import os
import numpy as np
from cvar_lab.data_gen import generate_scenarios
from cvar_lab.optimizer import minimize_cvar
from cvar_lab.utils import plot_loss_distribution, plot_cvar_surface, plot_asset_allocation

# 1. Ensure the results directory exists for your plots
os.makedirs('results', exist_ok=True)

def run_reproduction():
    # --- Configuration ---
    beta = 0.95            # Confidence level (95%)
    num_assets = 6         # Number of assets in the portfolio
    num_scenarios = 5000   # Number of 'Market States'
    asset_names = [f"Asset_{i+1}" for i in range(num_assets)]
    
    # 2. Generate Realistic "Fat-Tail" Scenarios (Student-T Distribution)
    print(f"Generating {num_scenarios} scenarios for {num_assets} assets...")
    scenarios = generate_scenarios(num_scenarios=num_scenarios, num_assets=num_assets, df=3)
    
    # 3. Perform Simultaneous VaR/CVaR Optimization
    print(f"Minimizing CVaR at Beta={beta}...")
    result = minimize_cvar(scenarios, beta=beta)
    
    if result.success:
        # Extract results from the optimizer's parameter vector [weights..., alpha]
        weights = result.x[:-1]
        var_alpha = result.x[-1]
        cvar_val = result.fun
        
        print("\n--- Optimization Results ---")
        print(f"Optimal Value-at-Risk (alpha): {var_alpha:.4f}")
        print(f"Optimal Conditional VaR: {cvar_val:.4f}")
        for name, w in zip(asset_names, weights):
            print(f"{name}: {w*100:.2f}%")
            
        # 4. Observation 1: The Loss Distribution (The Risk Tail)
        # Calculate the specific losses for the optimal portfolio configuration
        portfolio_losses = np.dot(scenarios, -weights)
        
        plot_loss_distribution(
            losses=portfolio_losses, 
            var=var_alpha, 
            cvar=cvar_val, 
            beta=beta,
            save_path="results/loss_tail.png"
        )
        
        # 5. Observation 2: Objective Convexity (Theorem 1 Proof)
        # This confirms that the solver found the true global minimum
        plot_cvar_surface(weights, scenarios, beta)
        
        # 6. Observation 3: Final System Configuration
        plot_asset_allocation(asset_names, weights)
        
    else:
        print("Optimization failed to converge:", result.message)

if __name__ == "__main__":
    run_reproduction()