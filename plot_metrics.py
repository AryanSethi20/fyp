import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_metrics(metrics_file="training_metrics.json"):
    # Load metrics from JSON file
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Set default style parameters
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Create a figure with subplots
    fig = plt.figure()

    # 1. Training Rewards Plot
    ax1 = plt.subplot(2, 2, 1)
    episodes = range(1, len(metrics['rewards']['episode_rewards']) + 1)
    ax1.plot(episodes, metrics['rewards']['episode_rewards'], 'b-', label='Episode Reward', alpha=0.6)
    ax1.plot(episodes, metrics['rewards']['moving_avg_rewards'], 'r-', label='Moving Average', linewidth=2)
    ax1.set_title('Training Rewards Over Time', pad=20)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()

    # 2. Policy Distribution Pie Chart
    ax2 = plt.subplot(2, 2, 2)
    policy_counts = metrics['policy_distribution']
    labels = ['CU Policy', 'ZW Policy']
    sizes = [policy_counts['0'], policy_counts['1']]
    total = sum(sizes)
    percentages = [count / total * 100 for count in sizes]
    colors = ['#1f77b4', '#2ca02c']
    ax2.pie(percentages, labels=[f'{l}\n({p:.1f}%)' for l, p in zip(labels, percentages)],
            colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Policy Distribution', pad=20)

    # 3. AoI Performance Comparison
    ax3 = plt.subplot(2, 2, (3, 4))
    cu_aoi = metrics['aoi_performance']['cu_aoi']
    zw_aoi = metrics['aoi_performance']['zw_aoi']

    # Calculate moving averages for smoother lines
    window = 20
    cu_ma = np.convolve(cu_aoi, np.ones(window) / window, mode='valid') if cu_aoi else []
    zw_ma = np.convolve(zw_aoi, np.ones(window) / window, mode='valid') if zw_aoi else []

    if cu_ma.size > 0:
        ax3.plot(cu_ma, 'b-', label='CU Policy AoI', alpha=0.8)
    if zw_ma.size > 0:
        ax3.plot(zw_ma, 'g-', label='ZW Policy AoI', alpha=0.8)

    ax3.set_title('AoI Performance by Policy', pad=20)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Age of Information')
    ax3.legend()

    # Adjust layout and save
    plt.tight_layout(pad=3.0)

    # Create plots directory if it doesn't exist
    Path('plots').mkdir(exist_ok=True)

    # Save plots
    plt.savefig('plots/training_metrics.png', dpi=300, bbox_inches='tight')

    # Save individual plots
    # Rewards
    plt.figure(figsize=(8, 6))
    plt.plot(episodes, metrics['rewards']['episode_rewards'], 'b-', label='Episode Reward', alpha=0.6)
    plt.plot(episodes, metrics['rewards']['moving_avg_rewards'], 'r-', label='Moving Average', linewidth=2)
    plt.title('Training Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/rewards.png', dpi=300, bbox_inches='tight')

    # Policy Distribution
    plt.figure(figsize=(8, 8))
    plt.pie(percentages, labels=[f'{l}\n({p:.1f}%)' for l, p in zip(labels, percentages)],
            colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Policy Distribution')
    plt.savefig('plots/policy_distribution.png', dpi=300, bbox_inches='tight')

    # AoI Performance
    plt.figure(figsize=(10, 6))
    if cu_ma.size > 0:
        plt.plot(cu_ma, 'b-', label='CU Policy AoI', alpha=0.8)
    if zw_ma.size > 0:
        plt.plot(zw_ma, 'g-', label='ZW Policy AoI', alpha=0.8)
    plt.title('AoI Performance by Policy')
    plt.xlabel('Step')
    plt.ylabel('Age of Information')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/aoi_performance.png', dpi=300, bbox_inches='tight')

    # Print summary statistics
    print("\nTraining Summary:")
    print(f"Total Episodes: {len(metrics['rewards']['episode_rewards'])}")
    print(f"Final Average Reward: {metrics['rewards']['moving_avg_rewards'][-1]:.2f}")
    print("\nPolicy Distribution:")
    print(f"CU Policy: {percentages[0]:.1f}%")
    print(f"ZW Policy: {percentages[1]:.1f}%")
    if cu_aoi and zw_aoi:
        print(f"\nFinal AoI Performance:")
        print(f"CU Average AoI: {np.mean(cu_aoi[-100:]):.2f}")
        print(f"ZW Average AoI: {np.mean(zw_aoi[-100:]):.2f}")


if __name__ == "__main__":
    plot_training_metrics()