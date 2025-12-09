"""
Visualization utilities for drone navigation results.
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_comparison(results, save_path="comparison_results.png"):
    """
    Plot algorithm comparison bar chart.
    
    Args:
        results: List of tuples (algo_name, mean_reward, std_reward)
        save_path: Path to save the plot
    """
    algos, means, stds = zip(*results)
    
    # Create color map
    colors = ['#FF6B6B' if name == 'A*' else '#4ECDC4' for name in algos]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(algos, means, yerr=stds, capsize=8, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title('Algorithm Comparison: Drone Navigation Performance', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Mean Evaluation Reward', fontsize=12, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', edgecolor='black', label='Classical'),
        Patch(facecolor='#4ECDC4', edgecolor='black', label='Deep RL')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.show()


def plot_trajectory(trajectory, obstacles, target, grid_limit=10, 
                   save_path="trajectory.png"):
    """
    Plot 3D trajectory with obstacles and target.
    
    Args:
        trajectory: List of [x, y, z] positions
        obstacles: List of obstacle positions
        target: Target position [x, y, z]
        grid_limit: Maximum coordinate value
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot obstacles
    for obs in obstacles:
        ax.scatter(*obs, c='red', s=300, marker='o', alpha=0.6)
    
    # Plot target
    ax.scatter(*target, c='green', s=400, marker='*', 
              edgecolors='darkgreen', linewidths=2, label='Target')
    
    # Plot trajectory
    traj = np.array(trajectory)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
           'b-', linewidth=2, label='Path', alpha=0.8)
    
    # Plot start and end
    ax.scatter(*traj[0], c='blue', s=200, marker='^', 
              edgecolors='darkblue', linewidths=2, label='Start')
    ax.scatter(*traj[-1], c='orange', s=200, marker='s', 
              edgecolors='darkorange', linewidths=2, label='End')
    
    ax.set_xlim(0, grid_limit)
    ax.set_ylim(0, grid_limit)
    ax.set_zlim(0, grid_limit)
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title('Drone Trajectory in 3D Space', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved to {save_path}")
    plt.show()