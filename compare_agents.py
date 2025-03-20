import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import logging
import threading
from publisher import EnhancedMQTTPublisher
from subscriber import EnhancedMQTTSubscriber
from environment import AoIEnvironment, DQNAgent, TrainingMetrics, train_rl_agent, print_training_summary, DoubleDQNAgent, train_double_dqn_agent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("comparison")

def compare_agents(num_episodes=150, num_runs=3):
    """
    Compare DQN and Double DQN performance over multiple runs.
    
    Args:
        num_episodes: Number of episodes to train each agent
        num_runs: Number of training runs to average over
    """
    # Storage for results
    dqn_rewards = []
    double_dqn_rewards = []
    
    for run in range(num_runs):
        logger.info(f"Starting comparison run {run+1}/{num_runs}")
        
        # Initialize environment components
        publisher = EnhancedMQTTPublisher()
        subscriber = EnhancedMQTTSubscriber()
        
        # Setup environment
        env = AoIEnvironment(
            publisher,
            subscriber,
            updates_per_window=10,
            num_windows=50
        )
        
        # Set up MQTT connection
        env.env.client.connect(env.env.broker, env.env.port)
        mqtt_thread = threading.Thread(target=env.env.client.loop_start)
        mqtt_thread.daemon = True
        mqtt_thread.start()
        
        # Get environment parameters
        state_size = env.get_state().shape[0]
        policies = [0, 1]  # CU or ZW
        
        try:
            # Train standard DQN
            logger.info(f"Run {run+1}/{num_runs}: Training standard DQN")
            dqn_agent = DQNAgent(state_size, policies)
            dqn_metrics = train_rl_agent(
                env=env,
                agent=dqn_agent,
                episodes=num_episodes,
                batch_size=32,
                window_size=50,
                run=run
            )
            print_training_summary(dqn_metrics)
            rewards = [np.mean(dqn_metrics.episode_rewards[i:i + 50])
                       for i in range(0, len(dqn_metrics.episode_rewards), 50)]
            dqn_rewards.append(rewards)
            
            # Reset environment
            env.reset()
            
            # Train Double DQN
            logger.info(f"Run {run+1}/{num_runs}: Training Double DQN")
            double_dqn_agent = DoubleDQNAgent(state_size, policies)
            double_dqn_metrics = train_double_dqn_agent(
                env=env,
                agent=double_dqn_agent,
                episodes=num_episodes,
                batch_size=32,
                window_size=50,
                run=run
            )
            print_training_summary(double_dqn_metrics)
            rewards = [np.mean(double_dqn_metrics.episode_rewards[i:i + 50])
                       for i in range(0, len(double_dqn_metrics.episode_rewards), 50)]
            double_dqn_rewards.append(rewards)
            
        finally:
            # Clean up MQTT
            env.env.client.loop_stop()
            env.env.client.disconnect()
    
    # Calculate averages across runs
    avg_dqn_rewards = np.mean(dqn_rewards, axis=0)
    avg_double_dqn_rewards = np.mean(double_dqn_rewards, axis=0)
    
    # Save comparison data
    comparison_data = {
        'dqn_rewards': avg_dqn_rewards.tolist(),
        'double_dqn_rewards': avg_double_dqn_rewards.tolist(),
        'episodes': list(range(1, num_episodes + 1))
    }
    
    with open('agent_comparison_1.json', 'w') as f:
        json.dump(comparison_data, f)
    
    # Plot comparison
    plot_comparison(comparison_data)
    
    return comparison_data

def plot_comparison(data):
    """
    Plot comparison between DQN and Double DQN performance.
    
    Args:
        data: Dictionary containing reward data for both algorithms
    """
    plt.figure(figsize=(12, 6))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(data['episodes'], data['dqn_rewards'], label='DQN')
    plt.plot(data['episodes'], data['double_dqn_rewards'], label='Double DQN')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Reward Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot difference (Double DQN - DQN)
    plt.subplot(1, 2, 2)
    diff = np.array(data['double_dqn_rewards']) - np.array(data['dqn_rewards'])
    plt.plot(data['episodes'], diff)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Reward Difference')
    plt.title('Double DQN Improvement over DQN')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dqn_vs_double_dqn_1.png')
    plt.show()

def load_and_visualize_results(filename='agent_comparison_1.json'):
    """
    Load previously saved comparison results and visualize them.
    
    Args:
        filename: Path to the JSON file with comparison data
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        plot_comparison(data)
        
    except FileNotFoundError:
        logger.error(f"Results file {filename} not found. Run comparison first.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare DQN and Double DQN')
    parser.add_argument('--run', action='store_true', help='Run new comparison')
    parser.add_argument('--episodes', type=int, default=150, help='Number of episodes')
    parser.add_argument('--runs', type=int, default=2, help='Number of runs to average')
    parser.add_argument('--visualize', action='store_true', help='Visualize saved results')
    
    args = parser.parse_args()
    
    if args.run:
        logger.info(f"Starting comparison with {args.episodes} episodes x {args.runs} runs")
        compare_agents(num_episodes=args.episodes, num_runs=args.runs)
    
    if args.visualize or not args.run:
        load_and_visualize_results()