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

    # with open(path + "/training_metrics-3.json") as f:
    #     data = json.load(f)
    # rewards_data = data.get('rewards', {})
    # step_rewards = rewards_data.get('episode_rewards', [])
    # run3 = [np.mean(step_rewards[i:i + actions_per_episode])
    #         for i in range(0, len(step_rewards), actions_per_episode)]

    # with open(path + "/training_metrics-4.json") as f:
    #     data = json.load(f)
    # rewards_data = data.get('rewards', {})
    # step_rewards = rewards_data.get('episode_rewards', [])
    # run4 = [np.mean(step_rewards[i:i + actions_per_episode])
    #         for i in range(0, len(step_rewards), actions_per_episode)]
    #
    # with open(path + "/training_metrics-5.json") as f:
    #     data = json.load(f)
    # rewards_data = data.get('rewards', {})
    # step_rewards = rewards_data.get('episode_rewards', [])
    # run5 = [np.mean(step_rewards[i:i + actions_per_episode])
    #         for i in range(0, len(step_rewards), actions_per_episode)]
    #
    # with open(path + "/training_metrics-6.json") as f:
    #     data = json.load(f)
    # rewards_data = data.get('rewards', {})
    # step_rewards = rewards_data.get('episode_rewards', [])
    # run6 = [np.mean(step_rewards[i:i + actions_per_episode])
    #         for i in range(0, len(step_rewards), actions_per_episode)]

    # Plot 1
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, run1, 'b-', label='Average Reward - Run 1', linewidth=1)
    plt.plot(episodes, run2, 'r-', label='Average Reward - Run 2', linewidth=1)
    # plt.plot(episodes, run3, 'g-', label='Average Reward - Run 3', linewidth=1)
    # plt.plot(episodes, run4, 'y-', label='Average Reward - Run 4', linewidth=1)
    # plt.plot(episodes, run5, 'c-', label='Average Reward - Run 5', linewidth=1)
    # plt.plot(episodes, run6, 'k-', label='Average Reward - Run 6', linewidth=1)
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


def avg_varying_parameters():
    # Load the AOI values from four different files
    file_paths = ["/Users/aryansethi20/Downloads/fyp/Runs/9/inference_metrics-1stSize.json", "/Users/aryansethi20/Downloads/fyp/Runs/9/inference_metrics-2ndSize.json",
                  "/Users/aryansethi20/Downloads/fyp/Runs/9/inference_metrics-3rdSize.json", "/Users/aryansethi20/Downloads/fyp/Runs/9/inference_metrics-4thSize.json",
                  "/Users/aryansethi20/Downloads/fyp/Runs/9/inference_metrics-5thSize.json"]

    mean_paoi = []

    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract first 3000 AOI values
        aoi_values = data["aoi_values"][6:3006]

        # Compute mean PAoI
        mean_paoi.append(np.mean(aoi_values))

    packet_size = [83, 185, 283, 399, 590]
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(packet_size, mean_paoi, marker='o', markersize=4, linestyle='-')
    plt.xlim(min(packet_size) - 50, max(packet_size) + 50)  # Expand the x-axis range
    plt.ylim(min(mean_paoi) - 0.1, max(mean_paoi) + 0.1)  # Expand the y-axis range
    plt.xlabel("Packet Size")
    plt.ylabel("Mean PAoI")
    plt.title("Mean PAoI across Different Packet Sizes")
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig("/Users/aryansethi20/Downloads/fyp/Runs/9/mean-paoi-across-different-sizes.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # plot_training_metrics("./Runs/9")
    # average_metrics("./Runs/8/")
    avg_varying_parameters()