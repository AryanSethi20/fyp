import json
import matplotlib.pyplot as plt
import numpy as np


def plot_training_metrics(metrics_file="training_metrics.json"):
    # Parameters
    num_episodes = 150
    actions_per_episode = 100
    episodes = range(1, num_episodes + 1)

    # Initialize lists to store data from all runs
    all_runs_rewards = []
    all_runs_policies = []
    all_runs_mu = []
    all_runs_lambda = []
    all_runs_ratio = []
    all_runs_aoi = []

    # Load and process data from all runs
    for i in range(1, 2):
        try:
            with open(metrics_file + f"final_double_dqn_metrics.json", 'r') as f:
                metrics = json.load(f)

            # Extract and trim data to desired length
            rewards = metrics.get('rewards', {}).get('episode_rewards', [])[:actions_per_episode * num_episodes]
            parameters = metrics.get('parameters', {})

            policy_history = parameters.get('policy_history', [])[:actions_per_episode * num_episodes]
            mu_history = parameters.get('mu_history', [])[:actions_per_episode * num_episodes]
            lambda_history = parameters.get('lambda_history', [])[:actions_per_episode * num_episodes]
            ratio_history = parameters.get('ratio_history', [])[:actions_per_episode * num_episodes]
            aoi_history = parameters.get('aoi_history', [])[:actions_per_episode * num_episodes]

            # Calculate episode averages
            episode_rewards = []
            episode_policies = []
            episode_mu = []
            episode_lambda = []
            episode_ratio = []
            episode_aoi = []

            for j in range(0, len(rewards), actions_per_episode):
                episode_rewards.append(np.mean(rewards[j:j + actions_per_episode]))
                episode_policies.extend(policy_history[j:j + actions_per_episode])
                episode_mu.append(np.mean(mu_history[j:j + actions_per_episode]))
                episode_lambda.append(np.mean(lambda_history[j:j + actions_per_episode]))
                episode_ratio.append(np.mean(ratio_history[j:j + actions_per_episode]))
                episode_aoi.append(np.mean(aoi_history[j:j + actions_per_episode]))

            all_runs_rewards.append(episode_rewards)
            all_runs_policies.extend(episode_policies)
            all_runs_mu.append(episode_mu)
            all_runs_lambda.append(episode_lambda)
            all_runs_ratio.append(episode_ratio)
            all_runs_aoi.append(episode_aoi)

        except Exception as e:
            print(f"Error processing run {i}: {e}")

    # Convert to numpy arrays and calculate means across runs
    all_runs_rewards = np.array(all_runs_rewards)
    mean_rewards = np.mean(all_runs_rewards, axis=0)
    mean_mu = np.mean(all_runs_mu, axis=0)
    mean_lambda = np.mean(all_runs_lambda, axis=0)
    mean_ratio = np.mean(all_runs_ratio, axis=0)
    mean_aoi = np.mean(all_runs_aoi, axis=0)

    # Calculate moving averages
    window_size = 10
    moving_avg_rewards = [np.mean(mean_rewards[max(0, i - window_size + 1):i + 1])
                          for i in range(len(mean_rewards))]

    # Plot 1: Rewards vs Episodes
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, mean_rewards, 'b-', label='Average Reward per Episode', linewidth=1)
    plt.plot(episodes, moving_avg_rewards, 'r-', label='Moving Average (Window=10)', linewidth=2)
    plt.title('Average Rewards over Episode', fontsize=16, pad=20)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig('plots/rewards-vs-episode.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Policy Distribution
    plt.figure(figsize=(10, 10))
    cu_count = all_runs_policies.count(0)
    zw_count = all_runs_policies.count(1)
    total_count = cu_count + zw_count

    plt.pie([cu_count / total_count * 100, zw_count / total_count * 100],
            labels=['CU', 'ZW'],
            autopct='%1.1f%%',
            colors=['lightblue', 'lightgreen'])
    plt.title('Policy Distribution Across All Runs', fontsize=16, pad=20)
    plt.savefig('plots/policy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Service and Arrival Rates
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, mean_mu, 'g-', label='μ (Service Rate)', linewidth=2)
    plt.plot(episodes, mean_lambda, 'b-', label='λ (Arrival Rate)', linewidth=2)
    plt.title('Average Service and Arrival Rates', fontsize=16, pad=20)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Rate', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig('plots/rates.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: λ/μ Ratio
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, mean_ratio, 'r-', label='Average Ratio', linewidth=2)
    plt.axhline(y=0.2, color='g', linestyle='--', label='Min Ratio (0.2)')
    plt.axhline(y=0.9, color='r', linestyle='--', label='Max Ratio (0.9)')
    plt.title('Average λ/μ Ratio', fontsize=16, pad=20)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('λ/μ Ratio', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig('plots/ratio.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 5: AoI
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, mean_aoi, 'b-', label='Average AoI', linewidth=2)
    plt.title('Average Age of Information (AoI)', fontsize=16, pad=20)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('AoI', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig('plots/aoi.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print("\nTraining Summary Statistics:")
    print(f"Number of Runs: 4")
    print(f"Episodes per Run: {num_episodes}")
    print(f"Actions per Episode: {actions_per_episode}")
    print("\nRewards:")
    print(f"  Mean: {np.mean(mean_rewards):.2f} ± {np.std(mean_rewards):.2f}")
    print(f"  Range: [{np.min(mean_rewards):.2f}, {np.max(mean_rewards):.2f}]")
    print("\nService Rate (μ):")
    print(f"  Mean: {np.mean(mean_mu):.2f} ± {np.std(mean_mu):.2f}")
    print(f"  Range: [{np.min(mean_mu):.2f}, {np.max(mean_mu):.2f}]")
    print("\nArrival Rate (λ):")
    print(f"  Mean: {np.mean(mean_lambda):.2f} ± {np.std(mean_lambda):.2f}")
    print(f"  Range: [{np.min(mean_lambda):.2f}, {np.max(mean_lambda):.2f}]")
    print("\nλ/μ Ratio:")
    print(f"  Mean: {np.mean(mean_ratio):.2f} ± {np.std(mean_ratio):.2f}")
    print(f"  Range: [{np.min(mean_ratio):.2f}, {np.max(mean_ratio):.2f}]")
    print("\nAoI:")
    print(f"  Mean: {np.mean(mean_aoi):.2f} ± {np.std(mean_aoi):.2f}")
    print(f"  Range: [{np.min(mean_aoi):.2f}, {np.max(mean_aoi):.2f}]")
    print(f"\nPolicy Distribution:")
    print(f"  CU: {cu_count / total_count * 100:.1f}%")
    print(f"  ZW: {zw_count / total_count * 100:.1f}%")


if __name__ == "__main__":
    plot_training_metrics("./")