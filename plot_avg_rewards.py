import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.ndimage import gaussian_filter1d
import matplotlib.ticker as ticker

def load_reward_data(file_paths, num_episodes=150, actions_per_episode=50):
    """
    Load rewards data from multiple run files for the same algorithm.
    
    Args:
        file_paths: List of file paths to load data from
        num_episodes: Number of episodes to process
        actions_per_episode: Number of actions per episode
        
    Returns:
        List of reward data per run
    """
    all_runs = []
    
    for path in file_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Extract episode rewards based on the format in your code
                rewards_data = data.get('rewards', {})
                step_rewards = rewards_data.get('episode_rewards', [])
                
                # Limit to num_episodes * actions_per_episode
                step_rewards = step_rewards[:num_episodes * actions_per_episode]
                
                # Calculate average reward per episode
                run_rewards = [np.mean(step_rewards[i:i + actions_per_episode])
                              for i in range(0, len(step_rewards), actions_per_episode)]
                
                all_runs.append(run_rewards)
                print(f"Loaded {len(run_rewards)} episodes from {path}")
            except Exception as e:
                print(f"Error loading file {path}: {e}")
        else:
            print(f"Warning: File {path} not found")
    
    return all_runs

def plot_reward_curves(all_runs, algorithm_name="DQN", window_size=10, save_path=None):
    """
    Plot reward curves with min-max ranges for a single algorithm with multiple runs.
    
    Args:
        all_runs: List of reward data per run
        algorithm_name: Name of the algorithm
        window_size: Size of moving average window
        save_path: Path to save the figure (if None, display instead)
    """
    # Make sure we have data
    if not all_runs:
        print("No data available to plot")
        return
    
    # Standardize lengths
    min_length = min(len(run) for run in all_runs)
    all_runs = [run[:min_length] for run in all_runs]
    
    # Create episode range
    episodes = np.arange(1, min_length + 1)
    
    # Convert to numpy array for easier manipulation
    runs_array = np.array(all_runs)
    
    # Calculate mean, min, max across runs
    if len(all_runs) > 1:
        mean_rewards = np.mean(runs_array, axis=0)
        min_rewards = np.min(runs_array, axis=0)
        max_rewards = np.max(runs_array, axis=0)
    else:
        mean_rewards = runs_array[0]
        min_rewards = runs_array[0]
        max_rewards = runs_array[0]
    
    # Calculate moving average
    moving_avg_rewards = np.array([
        np.mean(mean_rewards[max(0, i - window_size + 1):i + 1])
        for i in range(len(mean_rewards))
    ])
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define colors
    main_color = 'blue'
    moving_avg_color = 'red'
    fill_color = 'lightblue'
    
    # Plot individual runs if there are multiple
    for i, run in enumerate(all_runs):
        plt.plot(episodes, run, color=main_color, alpha=0.3, linewidth=0.5, 
             label=f"Run {i+1}" if i == 0 else "")

    # Always plot range (even with one run, it will show variation)
    plt.fill_between(
    episodes,
    min_rewards,
    max_rewards,
    alpha=0.2,
    color=fill_color
)
    # Plot mean line
    plt.plot(episodes, mean_rewards, color=main_color, linewidth=1.5, 
             label=f'Average Reward ({len(all_runs)} runs)' if len(all_runs) > 1 else 'Reward')
    
    # Plot moving average
    plt.plot(episodes, moving_avg_rewards, color=moving_avg_color, linewidth=2, 
             label=f'Moving Average (window={window_size})')
    
    # Add labels and styling
    plt.title(f'{algorithm_name} Rewards over Episodes', fontsize=16, pad=20)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Format y-axis with fewer decimal places
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_comparison(all_algorithms_runs, algorithm_names, window_size=10, save_path=None):
    """
    Plot comparison of multiple algorithms with min-max ranges.
    
    Args:
        all_algorithms_runs: Dictionary mapping algorithm names to lists of reward data per run
        algorithm_names: List of algorithm names to include
        window_size: Size of moving average window
        save_path: Path to save the figure (if None, display instead)
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define colors for different algorithms (extend as needed)
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot each algorithm
    for i, algorithm in enumerate(algorithm_names):
        if algorithm in all_algorithms_runs and all_algorithms_runs[algorithm]:
            runs = all_algorithms_runs[algorithm]
            
            # Standardize lengths within this algorithm
            min_length = min(len(run) for run in runs)
            runs = [run[:min_length] for run in runs]
            
            # Create episode range
            episodes = np.arange(1, min_length + 1)
            
            # Convert to numpy array
            runs_array = np.array(runs)
            
            # Calculate mean, min, max across runs
            mean_rewards = np.mean(runs_array, axis=0)
            min_rewards = np.min(runs_array, axis=0)
            max_rewards = np.max(runs_array, axis=0)
            
            # Apply smoothing
            smoothed_mean = gaussian_filter1d(mean_rewards, sigma=window_size/3)
            smoothed_min = gaussian_filter1d(min_rewards, sigma=window_size/3)
            smoothed_max = gaussian_filter1d(max_rewards, sigma=window_size/3)
            
            # Get color
            color = colors[i % len(colors)]
            
            # Plot mean line
            plt.plot(episodes, smoothed_mean, color=color, linewidth=2, label=algorithm)
            
            # Plot min-max range
            plt.fill_between(
                episodes,
                smoothed_min,
                smoothed_max,
                alpha=0.2,
                color=color
            )
    
    # Add labels and styling
    plt.title('Algorithm Comparison: Rewards over Episodes', fontsize=16, pad=20)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

# Example usage with your data
if __name__ == "__main__":
    # Example 1: Single algorithm with multiple runs
    file_paths = [
        "/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/final_double_dqn_metrics-rate=6-2.json"
    ]
    
    # Load reward data for multiple runs
    dqn_runs = load_reward_data(file_paths, num_episodes=150, actions_per_episode=50)
    
    # Plot reward curves for a single algorithm with multiple runs
    plot_reward_curves(dqn_runs, algorithm_name="Double DQN (Rate=6)", window_size=10, 
                      save_path="double_dqn_rewards.png")