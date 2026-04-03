import matplotlib.pyplot as plt
import numpy as np
from .core import objective_function

def plot_loss_distribution(losses, var, cvar, beta, save_path=None):
    """
    Visualizes the 'Tail' of the portfolio loss distribution, 
    shading the area representing the Conditional Value-at-Risk.
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Create the main histogram
    n, bins, patches = plt.hist(losses, bins=60, alpha=0.7, color='skyblue', 
                                edgecolor='white', label='Loss Distribution')
    
    # 2. Shade the CVaR zone (all losses > VaR)
    # We loop through the patches and change the color of those to the right of VaR
    for i in range(len(patches)):
        if bins[i] > var:
            patches[i].set_facecolor('salmon')
            patches[i].set_label('CVaR Zone' if i == len(patches)-1 else "")

    # 3. Draw vertical lines for VaR and CVaR
    plt.axvline(var, color='red', linestyle='--', linewidth=2, 
                label=f'VaR ({beta*100}%): {var:.4f}')
    plt.axvline(cvar, color='darkred', linestyle='-', linewidth=2, 
                label=f'CVaR: {cvar:.4f}')
    
    # Labeling and Formatting
    plt.title(f"Portfolio Loss Distribution and Expected Shortfall ($\\beta$={beta})", fontsize=14)
    plt.xlabel("Loss Amount (Positive = Loss, Negative = Gain)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_cvar_surface(weights, scenarios, beta, alpha_range=None):
    """
    Theorem 1 Visualization: Demonstrates the perfect convexity of F_beta.
    The global minimum of this curve should align with the optimized VaR.
    """
    if alpha_range is None:
        # We focus on the high-loss tail (80th to 99.9th percentile)
        losses = np.dot(scenarios, -weights)
        alpha_range = np.linspace(np.percentile(losses, 70), np.percentile(losses, 99.5), 100)
    
    # 1. Calculate F_beta values for the fixed weights across the alpha range
    f_vals = [objective_function(a, weights, scenarios, beta) for a in alpha_range]
    
    plt.figure(figsize=(8, 5))
    
    # 2. Plot the convex surface
    plt.plot(alpha_range, f_vals, color='purple', linewidth=2.5, label=r'$F_{\beta}(x, \alpha)$')
    
    # 3. Identify the minimum for visual confirmation
    min_idx = np.argmin(f_vals)
    plt.scatter(alpha_range[min_idx], f_vals[min_idx], color='gold', s=100, 
                edgecolor='black', zorder=5, label='Theoretical Minimum (VaR/CVaR)')
    
    plt.title(r"Convexity of the CVaR Objective Function $F_{\beta}$", fontsize=14)
    plt.xlabel(r"Threshold Value ($\alpha$)", fontsize=12)
    plt.ylabel(r"Function Value", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/cvar_surface.png")
    plt.show()

def plot_asset_allocation(asset_names, weights):
    """
    Visualizes the final 'Ground State' of the portfolio weights.
    """
    plt.figure(figsize=(10, 5))
    
    # Color bars teal
    bars = plt.bar(asset_names, weights * 100, color='teal', alpha=0.8, edgecolor='black')
    
    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom')

    plt.title("Optimal Asset Allocation (CVaR Minimized)", fontsize=14)
    plt.ylabel("Weight Percentage (%)", fontsize=12)
    plt.xlabel("Assets", fontsize=12)
    plt.ylim(0, max(weights * 100) + 10) # Add some head room for labels
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("results/asset_allocation.png")
    plt.show()