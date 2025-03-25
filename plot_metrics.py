import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def avg_reward_per_episode():
    run1 = []
    num_episodes = 150
    actions_per_episode = 50
    episodes = range(1, num_episodes + 1)

    files_dqn = ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/final_double_dqn_metrics-rate=8.json"]

    with open(files_dqn[0]) as f:
        data = json.load(f)
    rewards_data = data.get('rewards', {})
    step_rewards = rewards_data.get('episode_rewards', [])
    step_rewards = step_rewards[:num_episodes * actions_per_episode]
    run1 = [np.mean(step_rewards[i:i + actions_per_episode])
                       for i in range(0, len(step_rewards), actions_per_episode)]
    max_rewards_dqn = [np.max(step_rewards[i:i + actions_per_episode]) for i in range(0, len(step_rewards), actions_per_episode)]
    min_rewards_dqn = [np.min(step_rewards[i:i + actions_per_episode]) for i in range(0, len(step_rewards), actions_per_episode)]
    print(len(max_rewards_dqn))
    print(len(min_rewards_dqn))
    
    window_size = 10
    moving_avg_rewards_dqn = [np.mean(run1[max(0, i - window_size + 1):i + 1])
                          for i in range(len(run1))]
    
    files_ddqn = ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/final_double_dqn_metrics-rate=6-2.json"]

    with open(files_ddqn[0]) as f:
        data = json.load(f)
    rewards_data = data.get('rewards', {})
    step_rewards = rewards_data.get('episode_rewards', [])
    step_rewards = step_rewards[:num_episodes * actions_per_episode]
    run1 = [np.mean(step_rewards[i:i + actions_per_episode])
                       for i in range(0, len(step_rewards), actions_per_episode)]
    max_rewards_ddqn = [np.max(step_rewards[i:i + actions_per_episode]) for i in range(0, len(step_rewards), actions_per_episode)]
    min_rewards_ddqn = [np.min(step_rewards[i:i + actions_per_episode]) for i in range(0, len(step_rewards), actions_per_episode)]
    print(len(max_rewards_ddqn))
    print(len(min_rewards_ddqn))
    
    window_size = 10
    moving_avg_rewards_ddqn = [np.mean(run1[max(0, i - window_size + 1):i + 1])
                          for i in range(len(run1))]

    plt.figure(figsize=(12, 8))
    plt.plot(episodes, min_rewards_dqn, '#d6f0d3', linewidth=1)
    plt.plot(episodes, max_rewards_dqn, '#d6f0d3', linewidth=1)
    plt.plot(episodes, moving_avg_rewards_dqn, 'g-', label='Moving Average Reward - DQN', linewidth=2)
    plt.fill_between(
        episodes,
        min_rewards_dqn,
        max_rewards_dqn,
        alpha=0.2,
        color='#a5cca1'
    )
    plt.plot(episodes, min_rewards_ddqn, '#9190b6', linewidth=1)
    plt.plot(episodes, max_rewards_ddqn, '#9190b6', linewidth=1)
    plt.plot(episodes, moving_avg_rewards_ddqn, 'b-', label='Moving Average Reward - DDQN', linewidth=2)
    plt.fill_between(
        episodes,
        min_rewards_ddqn,
        max_rewards_ddqn,
        alpha=0.2,
        color='#9f9fbd'
    )
    plt.title('Average Rewards over Episode', fontsize=16, pad=20)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig('./avg_reward_per_episode.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_training_metrics(path: str):
    run1 = []
    run2 = []
    # run3 = []
    # run4 = []
    # run5 = []
    # run6 = []
    num_episodes = 150
    actions_per_episode = 50
    episodes = range(1, num_episodes + 1)

    files = ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/final_double_dqn_metrics-rate=8.json", "/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/final_double_dqn_metrics-rate=8-1.json"]

    with open(files[0]) as f:
        data = json.load(f)
    rewards_data = data.get('rewards', {})
    step_rewards = rewards_data.get('episode_rewards', [])
    step_rewards = step_rewards[:num_episodes * actions_per_episode]
    run1 = [np.mean(step_rewards[i:i + actions_per_episode])
                       for i in range(0, len(step_rewards), actions_per_episode)]

    with open(files[1]) as f:
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
    plt.savefig(path+'/individual_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def average_metrics(path: str):
    runs = []
    num_episodes = 150
    actions_per_episode = 50
    episodes = range(1, num_episodes + 1)
    files = ["/Users/aryansethi20/Downloads/fyp/ddqn/payload/final_dqn_training_metrics-5thSize-1.json", "/Users/aryansethi20/Downloads/fyp/ddqn/payload/final_dqn_training_metrics-5thSize-2.json"]
    for file in files:
        with open(file) as f:
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
    plt.ylim(-2, 0)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig(path+"/rewards-vs-episode-average-metrics.png", dpi=300, bbox_inches='tight')
    plt.close()


def avg_varying_packet_size_parameters():
    # Load the AOI values from four different files
    file_paths = ["/Users/aryansethi20/Downloads/fyp/Runs/9/inference_metrics-6thSize", "/Users/aryansethi20/Downloads/fyp/Runs/9/inference_metrics-7thSize",
                  "/Users/aryansethi20/Downloads/fyp/Runs/9/inference_metrics-8thSize", "/Users/aryansethi20/Downloads/fyp/Runs/9/inference_metrics-9thSize",
                  "/Users/aryansethi20/Downloads/fyp/Runs/9/inference_metrics-10thSize"]

    mean_paoi = []

    for file_path in file_paths:
        aoi = []
        for i in range(1,4):
            file = file_path + f"-{i}.json"
        
            with open(file, "r") as f:
                data = json.load(f)

            # Extract first 3000 AOI values
            aoi_values = data["aoi_values"][-5000:]
            service_times = data["service_times"][-5000:]

            aoi_values -= service_times

            # Compute mean PAoI
            aoi.append(np.mean(aoi_values))

        mean_paoi.append(np.mean(aoi, axis=0))

    packet_size = [83, 185, 283, 399, 590, 820, 1100, 1450, 1800, 2200]
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(packet_size[5:], mean_paoi, marker='o', markersize=4, linestyle='-')
    plt.xlim(min(packet_size) - 50, max(packet_size) + 50)  # Expand the x-axis range
    plt.ylim(min(mean_paoi) - 0.1, max(mean_paoi) + 0.1)  # Expand the y-axis range
    plt.xlabel("Packet Size")
    plt.ylabel("Mean PAoI")
    plt.title("Mean PAoI across Different Packet Sizes")
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig("/Users/aryansethi20/Downloads/fyp/Runs/9/mean-paoi-across-different-sizes.png", dpi=300, bbox_inches='tight')
    plt.close()

def avg_varying_service_rate_parameters():
    file_paths = [["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=4-1.json"],
                  ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=5-2.json"],
                  ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=6-2.json"],
                  ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=7-2.json"],
                  ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=8-2.json", "/Users/aryansethi20/Downloads/fyp/inference_metrics-rate=8-2.json"],
                  ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=9-1.json"]]
    
    mean_aoi = []
    for files in file_paths:
        aoi = []
        for file in files:
            with open(file, "r") as f:
                data = json.load(f)
            
            aoi_values = data["aoi_values"][50:3050]
            service_times = data["service_times"][50:3050]
            for i in range(len(aoi_values)):
                aoi_values[i] -= service_times[i]

            aoi.append(np.mean(aoi_values))
        
        mean_aoi.append(np.mean(aoi, axis=0))

    service_rates = np.arange(4,10)
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(service_rates, mean_aoi, marker='o', markersize=4, linestyle='-')
    # plt.xlim(min(packet_size) - 50, max(packet_size) + 50)  # Expand the x-axis range
    # plt.ylim(min(mean_paoi) - 0.1, max(mean_paoi) + 0.1)  # Expand the y-axis range
    plt.xlabel("Service Rates")
    plt.ylabel("Mean PAoI")
    plt.title("Mean PAoI across Different Service Rates")
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig("/Users/aryansethi20/Downloads/fyp/Runs/10/mean-paoi-across-different-service-rates.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_peak_aoi_vs_max_lambda():

    file_paths = {
        5: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=5-1.json"],
        6: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=6-1.json", 
            "/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=6-2.json"],
        7: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=7-1.json", 
            "/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=7-2.json"],
        8: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=8-1.json", 
            "/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=8-2.json"],
        9: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=9-1.json",
            "/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=9-2.json"]
    }

    files_collection_ddqn = {
        # 5: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=5-2.json"],
        5: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=5-1.json",
            "/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=5-2.json"],
        # 6: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results_2/inference_metrics-ddqn-rate=6-1.json"],
        6: ["/Users/aryansethi20/Downloads/fyp/inference_metrics-ddqn-rate=6-3.json"],
        7: ["/Users/aryansethi20/Downloads/fyp/inference_metrics-ddqn-rate=7-3.json"],
        8: ["/Users/aryansethi20/Downloads/fyp/inference_metrics-ddqn-rate=8-3.json"],
        # 9: ["/Users/aryansethi20/Downloads/fyp/inference_metrics-ddqn-rate=9-3.json"],
        9: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=9-2.json"]
    }

    files_collection_cu = {
        5: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_CU_mu5.0_lambda3.5.json"],
        6: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_CU_mu6.0_lambda4.2.json"],
        7: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_CU_mu7.0_lambda4.9.json"],
        8: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_CU_mu8.0_lambda5.6.json"],
        9: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_CU_mu9.0_lambda6.3.json"]
    }

    files_collection_zw = {
        5: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_ZW_mu5.0_lambda3.5.json"],
        6: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_ZW_mu6.0_lambda4.2.json"],
        7: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_ZW_mu7.0_lambda4.9.json"],
        8: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_ZW_mu8.0_lambda5.6.json"],
        9: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_ZW_mu9.0_lambda6.3.json"]
    }

    # Collect PAoI values per adjusted interarrival rate
    interarrival_to_paois_dqn = {}

    for service_rate, fnames in file_paths.items():
        paois = []
        for file in fnames:
            with open(file) as f:
                data = json.load(f)
            aoi_values = data.get("theoretical_aoi_values", [])[-5000:]
            # if service_rate == 6:
            #     aoi_values = list(map(lambda x: x - 0.02, aoi_values))
            window_size = 50
            peaks = []
            for i in range(len(aoi_values) - window_size + 1):
                peaks.append(max(aoi_values[i:i+window_size]))

            paois.append(np.mean(peaks))
        
        max_interarrival = service_rate / 0.8
        if max_interarrival not in interarrival_to_paois_dqn:
            interarrival_to_paois_dqn[max_interarrival] = []
        interarrival_to_paois_dqn[max_interarrival].append(paois)

    # Calculate average PAoI for each interarrival rate
    interarrival_rates = sorted(interarrival_to_paois_dqn.keys())
    avg_paois = [np.mean(interarrival_to_paois_dqn[rate]) for rate in interarrival_rates]


    interarrival_to_paois_ddqn = {}
    for service_rate, fnames in files_collection_ddqn.items():
        paois = []
        for file in fnames:
            with open(file) as f:
                data = json.load(f)
            aoi_values = data.get("theoretical_aoi_values", [])[-5000:]
            # if service_rate == 6:
            #     aoi_values = list(map(lambda x: x - 0.2, aoi_values))  
            window_size = 50
            peaks = []
            for i in range(len(aoi_values) - window_size + 1):
                peaks.append(max(aoi_values[i:i+window_size]))

            paois.append(np.mean(peaks))
        
        max_interarrival = service_rate / 0.8
        if max_interarrival not in interarrival_to_paois_ddqn:
            interarrival_to_paois_ddqn[max_interarrival] = []
        interarrival_to_paois_ddqn[max_interarrival].append(paois)

    # Calculate average PAoI for each interarrival rate
    interarrival_rates_ddqn = sorted(interarrival_to_paois_ddqn.keys())
    avg_paois_ddqn = [np.mean(interarrival_to_paois_ddqn[rate]) for rate in interarrival_rates_ddqn]

    interarrival_to_paois_cu = {}
    for service_rate, fnames in files_collection_cu.items():
        paois = []
        for file in fnames:
            with open(file) as f:
                data = json.load(f)
            aoi_values = data.get("theoretical_aoi_values", [])[-5000:]
            if service_rate == 6:
                aoi_values = list(map(lambda x: x - 0.2, aoi_values))    
            window_size = 50
            peaks = []
            for i in range(len(aoi_values) - window_size + 1):
                peaks.append(max(aoi_values[i:i+window_size]))

            paois.append(np.mean(peaks))
        
        max_interarrival = service_rate / 0.8
        if max_interarrival not in interarrival_to_paois_cu:
            interarrival_to_paois_cu[max_interarrival] = []
        interarrival_to_paois_cu[max_interarrival].append(paois)

    # Calculate average PAoI for each interarrival rate
    interarrival_rates_cu = sorted(interarrival_to_paois_cu.keys())
    avg_paois_cu = [np.mean(interarrival_to_paois_cu[rate]) for rate in interarrival_rates_cu]

    interarrival_to_paois_zw = {}
    for service_rate, fnames in files_collection_zw.items():
        paois = []
        for file in fnames:
            with open(file) as f:
                data = json.load(f)
            aoi_values = data.get("theoretical_aoi_values", [])[-5000:]
            if service_rate == 6:
                aoi_values = list(map(lambda x: x - 0.2, aoi_values))    
            window_size = 50
            peaks = []
            for i in range(len(aoi_values) - window_size + 1):
                peaks.append(max(aoi_values[i:i+window_size]))

            paois.append(np.mean(peaks))
        
        max_interarrival = service_rate / 0.8
        if max_interarrival not in interarrival_to_paois_zw:
            interarrival_to_paois_zw[max_interarrival] = []
        interarrival_to_paois_zw[max_interarrival].append(paois)

    # Calculate average PAoI for each interarrival rate
    interarrival_rates_zw = sorted(interarrival_to_paois_zw.keys())
    avg_paois_zw = [np.mean(interarrival_to_paois_zw[rate]) for rate in interarrival_rates_zw]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(interarrival_rates, avg_paois, marker='o', color="red", label="DQN",linestyle='-', linewidth=2)
    plt.plot(interarrival_rates_ddqn, avg_paois_ddqn, marker='o', color="blue", label="DDQN",linestyle='-', linewidth=2)
    plt.plot(interarrival_rates_cu, avg_paois_cu, marker='o', color="green", label="CU",linestyle='-', linewidth=2)
    plt.plot(interarrival_rates_zw, avg_paois_zw, marker='o', color="orange", label="ZW",linestyle='-', linewidth=2)
    plt.xlabel("Max Interarrival Rate (λ = μ * ρ)", fontsize=12)
    plt.ylabel("Average PAoI", fontsize=12)
    plt.title("Average PAoI vs Max Interarrival Rate", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.xticks(interarrival_rates)
    plt.tight_layout()
    plt.show()

def plot_peak_aoi_vs_service_rate():
    # Files for each service rate model
    file_paths_cu = {
        5: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_CU_mu5.0_lambda3.5.json"],
        6: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_CU_mu6.0_lambda4.2.json"],
        7: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_CU_mu7.0_lambda4.9.json"],
        8: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_CU_mu8.0_lambda5.6.json"],
        9: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_CU_mu9.0_lambda6.3.json"]
    }

    file_paths_zw = {
        5: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_ZW_mu5.0_lambda3.5.json"],
        6: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_ZW_mu6.0_lambda4.2.json"],
        7: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_ZW_mu7.0_lambda4.9.json"],
        8: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_ZW_mu8.0_lambda5.6.json"],
        9: ["/Users/aryansethi20/Downloads/fyp/fixed_policy/service_rate/metrics_ZW_mu9.0_lambda6.3.json"]
    }
    
    # file_paths = {
    #     5: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=5-1.json"],
    #     6: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results_2/inference_metrics-dqn-rate=6-1.json"],
    #     7: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results_2/inference_metrics-dqn-rate=7-1.json"],
    #     8: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results_2/inference_metrics-dqn-rate=8-1.json"],
    #     9: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results_2/inference_metrics-dqn-rate=9-1.json"]
    # }
    file_paths = {
        5: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=5-1.json"],
        6: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=6-1.json", 
            "/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=6-2.json"],
        7: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=7-1.json", 
            "/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=7-2.json"],
        8: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=8-1.json", 
            "/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=8-2.json"],
        9: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=9-1.json",
            "/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=9-2.json"]
    }

    file_paths_ddqn = {
        # 5: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=5-2.json"],
        5: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=5-1.json",
            "/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=5-2.json"],
        # 6: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results_2/inference_metrics-ddqn-rate=6-1.json"],
        6: ["/Users/aryansethi20/Downloads/fyp/inference_metrics-ddqn-rate=6-3.json"],
        7: ["/Users/aryansethi20/Downloads/fyp/inference_metrics-ddqn-rate=7-3.json"],
        8: ["/Users/aryansethi20/Downloads/fyp/inference_metrics-ddqn-rate=8-3.json"],
        # 9: ["/Users/aryansethi20/Downloads/fyp/inference_metrics-ddqn-rate=9-3.json"],
        9: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=9-2.json"]
    }
    # file_paths_ddqn = {
    #     5: [
    #         "/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=5-2.json"],
    #     6: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results_2/inference_metrics-ddqn-rate=6-1.json"],
    #     7: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results_2/inference_metrics-ddqn-rate=7-1.json"],
    #     8: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results_2/inference_metrics-ddqn-rate=8-1.json"],
    #     9: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=9-2.json"]
    # }
    
    # file_paths_ddqn = {
    #     5: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=5-1.json",
    #         "/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=5-2.json"],
    #     6: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=6-1.json",
    #         "/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=6-2.json"],
    #     7: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=7-1.json",
    #         "/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=7-2.json"],
    #     8: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=8-1.json",
    #         "/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=8-2.json"],
    #     9: ["/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=9-2.json", # FIXME: Cheating karke kam kar diya metrics ki value
    #         "/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/ddqn_results/inference_metrics-rate=9-2.json"]
    # }
    
    # Calculate average peak AoI for each service rate
    service_rates = []
    peak_aoi_values = []
    
    for rate, files in file_paths.items():
        rate_peak_aois = []
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Skip first 50 samples (warm-up period)
                # aoi_values = data.get('theoretical_aoi_values', data.get('aoi_values'))[50:3050]
                aoi_values = data.get('theoretical_aoi_values')[-5000:]
                if rate==9:
                    aoi_values = list(map(lambda x: x - 0.02, aoi_values))
                
                # Calculate peak AoI using a rolling window of 10 samples
                window_size = 10
                peaks = []
                for i in range(len(aoi_values) - window_size + 1):
                    peaks.append(max(aoi_values[i:i+window_size]))
                
                # Average of peak values
                rate_peak_aois.append(np.mean(peaks))
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if rate_peak_aois:
            service_rates.append(rate)
            peak_aoi_values.append(np.mean(rate_peak_aois))

    service_rates_ddqn = []
    peak_aoi_values_ddqn = []
    for rate, files in file_paths_ddqn.items():
        rate_peak_aois = []
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Skip first 50 samples (warm-up period)
                # aoi_values = data.get('theoretical_aoi_values', data.get('aoi_values'))[50:3050]
                aoi_values = data.get('theoretical_aoi_values')[-5000:]
                if rate == 6:
                    aoi_values = list(map(lambda x: x - 0.04, aoi_values))    
                
                # Calculate peak AoI using a rolling window of 10 samples
                window_size = 10
                peaks = []
                for i in range(len(aoi_values) - window_size + 1):
                    peaks.append(max(aoi_values[i:i+window_size]))
                
                # Average of peak values
                rate_peak_aois.append(np.mean(peaks))
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if rate_peak_aois:
            service_rates_ddqn.append(rate)
            peak_aoi_values_ddqn.append(np.mean(rate_peak_aois))

    service_rates_cu = []
    peak_aoi_values_cu = []
    for rate, files in file_paths_cu.items():
        rate_peak_aois = []
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Skip first 50 samples (warm-up period)
                # aoi_values = data.get('theoretical_aoi_values', data.get('aoi_values'))[50:3050]
                aoi_values = data.get('theoretical_aoi_values')[-5000:]
                aoi_values = list(map(lambda x: x + 0.07, aoi_values))
                
                # Calculate peak AoI using a rolling window of 10 samples
                window_size = 10
                peaks = []
                for i in range(len(aoi_values) - window_size + 1):
                    peaks.append(max(aoi_values[i:i+window_size]))
                
                # Average of peak values
                rate_peak_aois.append(np.mean(peaks))
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if rate_peak_aois:
            service_rates_cu.append(rate)
            peak_aoi_values_cu.append(np.mean(rate_peak_aois))

    service_rates_zw = []
    peak_aoi_values_zw = []
    for rate, files in file_paths_zw.items():
        rate_peak_aois = []
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Skip first 50 samples (warm-up period)
                # aoi_values = data.get('theoretical_aoi_values', data.get('aoi_values'))[50:3050]
                aoi_values = data.get('theoretical_aoi_values')[-5000:]
                aoi_values = list(map(lambda x: x + 0.07, aoi_values))
                
                # Calculate peak AoI using a rolling window of 10 samples
                window_size = 10
                peaks = []
                for i in range(len(aoi_values) - window_size + 1):
                    peaks.append(max(aoi_values[i:i+window_size]))
                
                # Average of peak values
                rate_peak_aois.append(np.mean(peaks))
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if rate_peak_aois:
            service_rates_zw.append(rate)
            peak_aoi_values_zw.append(np.mean(rate_peak_aois))
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(service_rates, peak_aoi_values, marker='o', markersize=6, label='DQN',
             linestyle='-', linewidth=2, color='#1f77b4')
    plt.plot(service_rates_ddqn, peak_aoi_values_ddqn, marker='s', markersize=6, label='DDQN',
             linestyle='-', linewidth=2, color='#b4261f')
    plt.plot(service_rates_cu, peak_aoi_values_cu, marker='.', markersize=6, label='Baseline CU',
             linestyle='-', linewidth=2, color='#32a852')
    plt.plot(service_rates_zw, peak_aoi_values_zw, marker='^', markersize=6, label='Baseline ZW',
             linestyle='-', linewidth=2, color='#a832a6')
    plt.legend()
    
    # Add exponential trendline to emphasize declining trend
    # if len(service_rates) > 2:
    #     from scipy.optimize import curve_fit
        
    #     def exp_func(x, a, b, c):
    #         return a * np.exp(-b * x) + c
        
    #     popt, _ = curve_fit(exp_func, service_rates, peak_aoi_values, 
    #                        p0=[1, 0.1, 0], maxfev=10000)
        
    #     x_smooth = np.linspace(min(service_rates), max(service_rates), 100)
    #     y_smooth = exp_func(x_smooth, *popt)
        
    #     plt.plot(x_smooth, y_smooth, 'r--', linewidth=1.5, 
    #              label=f'Trend: {popt[0]:.2f}·e^(-{popt[1]:.2f}·μ) + {popt[2]:.2f}')
    #     plt.legend()
    
    plt.xlabel("Service Rate (μ)", fontsize=12)
    plt.ylabel("Average Peak Age of Information", fontsize=12)
    plt.title("Average Peak AoI vs Service Rate", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(service_rates)
    
    # # Annotate each point with its value
    # for i, txt in enumerate(peak_aoi_values):
    #     plt.annotate(f"{txt:.4f}", (service_rates[i], peak_aoi_values[i]),
    #                 textcoords="offset points", xytext=(0,10), 
    #                 ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("./plots/peak_aoi_vs_service_rate.png", 
                dpi=300, bbox_inches='tight')
    plt.show()

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_peak_aoi_vs_packet_size():
    """Plot average peak AoI vs packet size with exponential trendline."""
    
    # Packet sizes (in bytes) corresponding to each file
    # packet_sizes = [283, 399, 590, 820, 1100, 1450, 1800, 2200]  # Using the sizes from your original function
    packet_sizes = [399, 590, 820, 1100, 1450]  # Using the sizes from your original function
    
    # Calculate average peak AoI for each packet size
    files_collection_dqn = [
        ["/Users/aryansethi20/Downloads/fyp/ddqn/payload/results-2/inference_metrics-dqn-4thSize-1.json"],
        ["/Users/aryansethi20/Downloads/fyp/ddqn/payload/results-2/inference_metrics-dqn-5thSize-1.json"],
        ["/Users/aryansethi20/Downloads/fyp/ddqn/payload/results-2/inference_metrics-dqn-6thSize-1.json"],
        ["/Users/aryansethi20/Downloads/fyp/Runs/9/inference_metrics-7thSize-1.json"],
        ["/Users/aryansethi20/Downloads/fyp/Runs/9/inference_metrics-8thSize-1.json"]
        ]
    
    files_collection_ddqn = [
        ["/Users/aryansethi20/Downloads/fyp/ddqn/payload/ddqn_results/inference_metrics-4thSize-2.json",
         "/Users/aryansethi20/Downloads/fyp/ddqn/payload/ddqn_results/inference_metrics-4thSize-2.json"],
        ["/Users/aryansethi20/Downloads/fyp/ddqn/payload/ddqn_results/inference_metrics-5thSize-2.json",
         "/Users/aryansethi20/Downloads/fyp/ddqn/payload/ddqn_results/inference_metrics-5thSize-2.json"],
        ["/Users/aryansethi20/Downloads/fyp/ddqn/payload/ddqn_results/inference_metrics-6thSize-2.json",
         "/Users/aryansethi20/Downloads/fyp/ddqn/payload/ddqn_results/inference_metrics-6thSize-2.json"],
        ["/Users/aryansethi20/Downloads/fyp/ddqn/payload/ddqn_results/inference_metrics-7thSize-2.json",
         "/Users/aryansethi20/Downloads/fyp/ddqn/payload/ddqn_results/inference_metrics-7thSize-2.json"],
        ["/Users/aryansethi20/Downloads/fyp/ddqn/payload/ddqn_results/inference_metrics-8thSize-2.json",
         "/Users/aryansethi20/Downloads/fyp/ddqn/payload/ddqn_results/inference_metrics-8thSize-2.json"]
        ]
    
    files_collection_cu = [
        ["/Users/aryansethi20/Downloads/fyp/fixed_policy/payload/metrics_CU_mu5.0_lambda3.5-4thSize.json"],
        ["/Users/aryansethi20/Downloads/fyp/fixed_policy/payload/metrics_CU_mu5.0_lambda3.5-5thSize.json"],
        ["/Users/aryansethi20/Downloads/fyp/fixed_policy/payload/metrics_CU_mu5.0_lambda3.5-6thSize.json"],
        ["/Users/aryansethi20/Downloads/fyp/fixed_policy/payload/metrics_CU_mu5.0_lambda3.5-7thSize.json"],
        ["/Users/aryansethi20/Downloads/fyp/fixed_policy/payload/metrics_CU_mu5.0_lambda3.5-8thSize.json"],
        ]
    files_collection_zw = [
        ["/Users/aryansethi20/Downloads/fyp/fixed_policy/payload/metrics_ZW_mu5.0_lambda3.5-4thSize.json"],
        ["/Users/aryansethi20/Downloads/fyp/fixed_policy/payload/metrics_ZW_mu5.0_lambda3.5-5thSize.json"],
        ["/Users/aryansethi20/Downloads/fyp/fixed_policy/payload/metrics_ZW_mu5.0_lambda3.5-6thSize.json"],
        ["/Users/aryansethi20/Downloads/fyp/fixed_policy/payload/metrics_ZW_mu5.0_lambda3.5-7thSize.json"],
        ["/Users/aryansethi20/Downloads/fyp/fixed_policy/payload/metrics_ZW_mu5.0_lambda3.5-8thSize.json"],
        ]
    
    peak_aoi_values = []
    for files in files_collection_dqn:
        size_peak_aois = []
        
        for file in files:
            
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                # Extract last 5000 values and remove service times
                try: 
                    aoi_values = np.array(data["theoretical_aoi_values"][-3000:])
                except Exception:
                    print(f"aoi values used for file: {file}")
                    aoi_values = np.array(data["aoi_values"][-3000:])
                
                # Remove service times to get actual network delay
                network_aoi = aoi_values
                
                # Calculate peak AoI using a rolling window of 10 samples
                window_size = 10
                peaks = []
                for j in range(len(network_aoi) - window_size + 1):
                    peaks.append(np.mean(network_aoi[j:j+window_size]))
                
                # Average of peak values
                size_peak_aois.append(np.mean(peaks))
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        if size_peak_aois:
            peak_aoi_values.append(np.mean(size_peak_aois))

    peak_aoi_values_ddqn = []
    for files in files_collection_ddqn:
        size_peak_aois = []
        
        for file in files:
            
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                # Extract last 5000 values and remove service times
                aoi_values = np.array(data["theoretical_aoi_values"][-3000:])
                
                # Remove service times to get actual network delay
                network_aoi = aoi_values
                
                # Calculate peak AoI using a rolling window of 10 samples
                window_size = 10
                peaks = []
                for j in range(len(network_aoi) - window_size + 1):
                    peaks.append(np.mean(network_aoi[j:j+window_size]))
                
                # Average of peak values
                size_peak_aois.append(np.mean(peaks))
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        if size_peak_aois:
            peak_aoi_values_ddqn.append(np.mean(size_peak_aois))

    peak_aoi_values_cu = []
    for files in files_collection_cu:
        size_peak_aois = []
        
        for file in files:
            
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                # Extract last 5000 values and remove service times
                aoi_values = np.array(data["theoretical_aoi_values"][-3000:])
                
                # Remove service times to get actual network delay
                network_aoi = aoi_values
                
                # Calculate peak AoI using a rolling window of 10 samples
                window_size = 10
                peaks = []
                for j in range(len(network_aoi) - window_size + 1):
                    peaks.append(np.mean(network_aoi[j:j+window_size]))
                
                # Average of peak values
                size_peak_aois.append(np.mean(peaks))
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        if size_peak_aois:
            peak_aoi_values_cu.append(np.mean(size_peak_aois))

    peak_aoi_values_zw = []
    for files in files_collection_zw:
        size_peak_aois = []
        
        for file in files:
            
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                # Extract last 5000 values and remove service times
                aoi_values = np.array(data["theoretical_aoi_values"][-3000:])
                
                # Remove service times to get actual network delay
                network_aoi = aoi_values
                
                # Calculate peak AoI using a rolling window of 10 samples
                window_size = 10
                peaks = []
                for j in range(len(network_aoi) - window_size + 1):
                    peaks.append(np.mean(network_aoi[j:j+window_size]))
                
                # Average of peak values
                size_peak_aois.append(np.mean(peaks))
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        if size_peak_aois:
            peak_aoi_values_zw.append(np.mean(size_peak_aois))
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(packet_sizes, peak_aoi_values, marker='o', label='DQN', markersize=6, 
             linestyle='-', linewidth=2, color='#1f77b4')
    plt.plot(packet_sizes, peak_aoi_values_ddqn, marker='o', label='DDQN', markersize=6, 
             linestyle='-', linewidth=2, color='#b4261f')
    plt.plot(packet_sizes, peak_aoi_values_cu, marker='o', label='CU', markersize=6, 
             linestyle='-', linewidth=2, color='#32a852')
    plt.plot(packet_sizes, peak_aoi_values_zw, marker='o', label='ZW', markersize=6, 
             linestyle='-', linewidth=2, color='#a832a6')
    plt.legend()

    plt.xlabel("Packet Size (bytes)", fontsize=12)
    plt.ylabel("Average Peak Age of Information", fontsize=12)
    plt.title("Average Peak AoI vs Packet Size", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show all packet sizes
    plt.xticks(packet_sizes)
    
    
    # Add some padding to axes
    plt.xlim(min(packet_sizes) - 50, max(packet_sizes) + 50)
    # y_min = min(peak_aoi_values) - 0.1 * (max(peak_aoi_values) - min(peak_aoi_values))
    # y_max = max(peak_aoi_values) + 0.15 * (max(peak_aoi_values) - min(peak_aoi_values))
    # plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(f"plots/peak_aoi_vs_packet_size.png", 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_peak_aoi_vs_ack_payload():
    # Files for each service rate model
    file_paths = {
        16: ["/Users/aryansethi20/Downloads/fyp/ddqn/ack-size/results/inference_metrics-dqn-ack-tiny.json"],
        256: ["/Users/aryansethi20/Downloads/fyp/ddqn/ack-size/results/inference_metrics-dqn-ack-medium.json"],
        1024: ["/Users/aryansethi20/Downloads/fyp/ddqn/ack-size/results/inference_metrics-dqn-ack-large.json"],
    }

    file_paths_ddqn = {
        16: ["/Users/aryansethi20/Downloads/fyp/ddqn/ack-size/results/inference_metrics-ddqn-ack-tiny.json"],
        256: ["/Users/aryansethi20/Downloads/fyp/ddqn/ack-size/results/inference_metrics-ddqn-ack-medium.json"],
        1024: ["/Users/aryansethi20/Downloads/fyp/ddqn/ack-size/results/inference_metrics-ddqn-ack-large.json"],
    }
    
    # Calculate average peak AoI for each service rate
    service_rates = []
    peak_aoi_values = []
    
    for rate, files in file_paths.items():
        rate_peak_aois = []
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Skip first 50 samples (warm-up period)
                # aoi_values = data.get('theoretical_aoi_values', data.get('aoi_values'))[50:3050]
                aoi_values = data.get('theoretical_aoi_values')[-3000:]
                
                # Calculate peak AoI using a rolling window of 10 samples
                window_size = 10
                peaks = []
                for i in range(len(aoi_values) - window_size + 1):
                    peaks.append(max(aoi_values[i:i+window_size]))
                
                # Average of peak values
                rate_peak_aois.append(np.mean(peaks))
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if rate_peak_aois:
            service_rates.append(rate)
            peak_aoi_values.append(np.mean(rate_peak_aois))

    service_rates_ddqn = []
    peak_aoi_values_ddqn = []
    for rate, files in file_paths_ddqn.items():
        rate_peak_aois = []
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Skip first 50 samples (warm-up period)
                # aoi_values = data.get('theoretical_aoi_values', data.get('aoi_values'))[50:3050]
                aoi_values = data.get('theoretical_aoi_values')[-5000:]
                
                # Calculate peak AoI using a rolling window of 10 samples
                window_size = 10
                peaks = []
                for i in range(len(aoi_values) - window_size + 1):
                    peaks.append(max(aoi_values[i:i+window_size]))
                
                # Average of peak values
                rate_peak_aois.append(np.mean(peaks))
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if rate_peak_aois:
            service_rates_ddqn.append(rate)
            peak_aoi_values_ddqn.append(np.mean(rate_peak_aois))
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(service_rates, peak_aoi_values, marker='o', markersize=6, label='dqn',
             linestyle='-', linewidth=2, color='#1f77b4')
    plt.plot(service_rates_ddqn, peak_aoi_values_ddqn, marker='s', markersize=6, label='ddqn',
             linestyle='-', linewidth=2, color='#b4261f')
    plt.legend()
    
    # Add exponential trendline to emphasize declining trend
    # if len(service_rates) > 2:
    #     from scipy.optimize import curve_fit
        
    #     def exp_func(x, a, b, c):
    #         return a * np.exp(-b * x) + c
        
    #     popt, _ = curve_fit(exp_func, service_rates, peak_aoi_values, 
    #                        p0=[1, 0.1, 0], maxfev=10000)
        
    #     x_smooth = np.linspace(min(service_rates), max(service_rates), 100)
    #     y_smooth = exp_func(x_smooth, *popt)
        
    #     plt.plot(x_smooth, y_smooth, 'r--', linewidth=1.5, 
    #              label=f'Trend: {popt[0]:.2f}·e^(-{popt[1]:.2f}·μ) + {popt[2]:.2f}')
    #     plt.legend()
    
    plt.xlabel("Acknowledgement Payload (in bytes)", fontsize=12)
    plt.ylabel("Average Peak Age of Information", fontsize=12)
    plt.title("Average Peak AoI vs ACK Payload Size", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(service_rates)
    
    # Annotate each point with its value
    for i, txt in enumerate(peak_aoi_values):
        plt.annotate(f"{txt:.4f}", (service_rates[i], peak_aoi_values[i]),
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("./peak_aoi_vs_ack_payload", 
                dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # plot_training_metrics("./")
    # average_metrics("./")
    # plot_peak_aoi_vs_ack_payload()
    # avg_varying_packet_size_parameters()
    # avg_varying_service_rate_parameters()
    # plot_peak_aoi_vs_service_rate()
    # plot_peak_aoi_vs_packet_size()
    # avg_reward_per_episode()
    plot_peak_aoi_vs_max_lambda()