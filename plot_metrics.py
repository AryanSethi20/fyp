import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_metrics(path: str):
    run1 = []
    run2 = []
    run3 = []
    run4 = []
    run5 = []
    run6 = []
    num_episodes = 150
    actions_per_episode = 100
    episodes = range(1, num_episodes + 1)

    with open(path + "/training_metrics-1.json") as f:
        data = json.load(f)
    rewards_data = data.get('rewards', {})
    step_rewards = rewards_data.get('episode_rewards', [])
    step_rewards = step_rewards[:num_episodes * actions_per_episode]
    run1 = [np.mean(step_rewards[i:i + actions_per_episode])
                       for i in range(0, len(step_rewards), actions_per_episode)]
    with open(path + "/training_metrics-2.json") as f:
        data = json.load(f)
    rewards_data = data.get('rewards', {})
    step_rewards = rewards_data.get('episode_rewards', [])
    run2 = [np.mean(step_rewards[i:i + actions_per_episode])
            for i in range(0, len(step_rewards), actions_per_episode)]
    with open(path + "/training_metrics-3.json") as f:
        data = json.load(f)
    rewards_data = data.get('rewards', {})
    step_rewards = rewards_data.get('episode_rewards', [])
    run3 = [np.mean(step_rewards[i:i + actions_per_episode])
            for i in range(0, len(step_rewards), actions_per_episode)]
    with open(path + "/training_metrics-4.json") as f:
        data = json.load(f)
    rewards_data = data.get('rewards', {})
    step_rewards = rewards_data.get('episode_rewards', [])
    run4 = [np.mean(step_rewards[i:i + actions_per_episode])
            for i in range(0, len(step_rewards), actions_per_episode)]

    with open(path + "/training_metrics-5.json") as f:
        data = json.load(f)
    rewards_data = data.get('rewards', {})
    step_rewards = rewards_data.get('episode_rewards', [])
    run5 = [np.mean(step_rewards[i:i + actions_per_episode])
            for i in range(0, len(step_rewards), actions_per_episode)]

    with open(path + "/training_metrics-6.json") as f:
        data = json.load(f)
    rewards_data = data.get('rewards', {})
    step_rewards = rewards_data.get('episode_rewards', [])
    run6 = [np.mean(step_rewards[i:i + actions_per_episode])
            for i in range(0, len(step_rewards), actions_per_episode)]

    # Plot 1
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, run1, 'b-', label='Average Reward - Run 1', linewidth=1)
    plt.plot(episodes, run2, 'r-', label='Average Reward - Run 2', linewidth=1)
    plt.plot(episodes, run3, 'g-', label='Average Reward - Run 3', linewidth=1)
    plt.plot(episodes, run4, 'y-', label='Average Reward - Run 4', linewidth=1)
    plt.plot(episodes, run5, 'c-', label='Average Reward - Run 5', linewidth=1)
    plt.plot(episodes, run6, 'k-', label='Average Reward - Run 6', linewidth=1)
    plt.title('Average Rewards over Episode', fontsize=16, pad=20)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig(path+'/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def average_metrics(path: str):
    runs = []
    num_episodes = 150
    actions_per_episode = 100
    episodes = range(1, num_episodes + 1)
    for i in range(1,7):
        filename = "training_metrics-" + str(i) + ".json"
        with open(path + filename) as f:
            data = json.load(f)
        step_rewards = data.get('rewards', {}).get('episode_rewards', [])
        step_rewards = step_rewards[:actions_per_episode*num_episodes]
        print(len(step_rewards))
        runs.append([np.mean(step_rewards[i:i + actions_per_episode])
                for i in range(0, len(step_rewards), actions_per_episode)])

    res = np.mean(runs, axis=0)
    window_size = 10
    moving_avg_rewards = [np.mean(res[max(0, i - window_size + 1):i + 1])
                          for i in range(len(res))]

    plt.figure(figsize=(12, 8))
    plt.plot(episodes, res, 'b-', label='Average Reward vs Episodes', linewidth=2)
    plt.plot(episodes, moving_avg_rewards, 'r-', label='Moving Average Reward vs Episodes (Window Size = 10)', linewidth=2)
    plt.title('Average Rewards over Episodes', fontsize=16, pad=20)
    # plt.ylim(-0.8, 0)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig(path+"/rewards-vs-episode-average-metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_training_metrics("./Runs/8/")
    average_metrics("./Runs/8/")