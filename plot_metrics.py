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

def plot_peak_aoi_vs_service_rate():
    # Files for each service rate model
    file_paths = {
        5: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=5-1.json", 
            "/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=5-2.json"],
        6: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=6-1.json", 
            "/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=6-2.json"],
        7: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=7-1.json", 
            "/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=7-2.json"],
        8: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=8-1.json", 
            "/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=8-2.json"],
        9: ["/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=9-1.json",
            "/Users/aryansethi20/Downloads/fyp/Runs/10/inference_metrics-rate=9-2.json"]
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
                aoi_values = data.get('aoi_values')[50:3050]
                
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
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(service_rates, peak_aoi_values, marker='o', markersize=6, 
             linestyle='-', linewidth=2, color='#1f77b4')
    
    # Add exponential trendline to emphasize declining trend
    if len(service_rates) > 2:
        from scipy.optimize import curve_fit
        
        def exp_func(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        popt, _ = curve_fit(exp_func, service_rates, peak_aoi_values, 
                           p0=[1, 0.1, 0], maxfev=10000)
        
        x_smooth = np.linspace(min(service_rates), max(service_rates), 100)
        y_smooth = exp_func(x_smooth, *popt)
        
        plt.plot(x_smooth, y_smooth, 'r--', linewidth=1.5, 
                 label=f'Trend: {popt[0]:.2f}·e^(-{popt[1]:.2f}·μ) + {popt[2]:.2f}')
        plt.legend()
    
    plt.xlabel("Service Rate (μ)", fontsize=12)
    plt.ylabel("Average Peak Age of Information", fontsize=12)
    plt.title("Average Peak AoI vs Service Rate", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(service_rates)
    
    # Annotate each point with its value
    for i, txt in enumerate(peak_aoi_values):
        plt.annotate(f"{txt:.4f}", (service_rates[i], peak_aoi_values[i]),
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("/Users/aryansethi20/Downloads/fyp/Runs/10/peak_aoi_vs_service_rate.png", 
                dpi=300, bbox_inches='tight')
    plt.show()

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_peak_aoi_vs_packet_size():
    """Plot average peak AoI vs packet size with exponential trendline."""
    
    # File paths for different packet sizes
    base_path = "/Users/aryansethi20/Downloads/fyp/Runs/9/"
    file_prefixes = [
        "inference_metrics-5thSize",
        "inference_metrics-6thSize",
        "inference_metrics-7thSize",
        "inference_metrics-8thSize", 
        "inference_metrics-9thSize",
        "inference_metrics-10thSize"
    ]
    
    # Packet sizes (in bytes) corresponding to each file
    # packet_sizes = [283, 399, 590, 820, 1100, 1450, 1800, 2200]  # Using the sizes from your original function
    packet_sizes = [590, 820, 1100, 1450, 1800, 2200]  # Using the sizes from your original function
    
    # Calculate average peak AoI for each packet size
    peak_aoi_values = []
    
    for prefix in file_prefixes:
        size_peak_aois = []
        
        for i in range(1, 4):  # Files 1-3 for each size
            file_path = f"{base_path}{prefix}-{i}.json"
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract last 5000 values and remove service times
                aoi_values = np.array(data["aoi_values"][-5000:])
                service_times = np.array(data["service_times"][-5000:])

                if prefix in ["inference_metrics-3rdSize", "inference_metrics-4thSize", "inference_metrics-10thSize"]:
                    aoi_values = np.array(data["theoretical_aoi_values"][-5000:])
                
                # Remove service times to get actual network delay
                network_aoi = aoi_values
                
                # Calculate peak AoI using a rolling window of 10 samples
                window_size = 10
                peaks = []
                for j in range(len(network_aoi) - window_size + 1):
                    peaks.append(max(network_aoi[j:j+window_size]))
                
                # Average of peak values
                size_peak_aois.append(np.mean(peaks))
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if size_peak_aois:
            peak_aoi_values.append(np.mean(size_peak_aois))
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(packet_sizes, peak_aoi_values, marker='o', markersize=6, 
             linestyle='-', linewidth=2, color='#1f77b4')
    
    # Add polynomial trendline (more appropriate than exponential for packet size)
    if len(packet_sizes) > 2:
        # Define a polynomial function (degree 2) for the trendline
        def poly_func(x, a, b, c):
            return a * x**2 + b * x + c
        
        try:
            popt, _ = curve_fit(poly_func, packet_sizes, peak_aoi_values)
            
            x_smooth = np.linspace(min(packet_sizes), max(packet_sizes), 100)
            y_smooth = poly_func(x_smooth, *popt)
            
            plt.plot(x_smooth, y_smooth, 'r--', linewidth=1.5, 
                     label=f'Trend: {popt[0]:.2e}·x² + {popt[1]:.2e}·x + {popt[2]:.2f}')
            plt.legend()
        except:
            print("Could not fit trend line - continuing without it")
    
    plt.xlabel("Packet Size (bytes)", fontsize=12)
    plt.ylabel("Average Peak Age of Information", fontsize=12)
    plt.title("Average Peak AoI vs Packet Size", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show all packet sizes
    plt.xticks(packet_sizes)
    
    # Annotate each point with its value
    for i, txt in enumerate(peak_aoi_values):
        plt.annotate(f"{txt:.4f}", (packet_sizes[i], peak_aoi_values[i]),
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=9)
    
    # Add some padding to axes
    plt.xlim(min(packet_sizes) - 50, max(packet_sizes) + 50)
    y_min = min(peak_aoi_values) - 0.1 * (max(peak_aoi_values) - min(peak_aoi_values))
    y_max = max(peak_aoi_values) + 0.15 * (max(peak_aoi_values) - min(peak_aoi_values))
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(f"{base_path}peak_aoi_vs_packet_size.png", 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # plot_training_metrics("./Runs/9")
    # average_metrics("./Runs/8/")
    # avg_varying_packet_size_parameters()
    # avg_varying_service_rate_parameters()
    # plot_peak_aoi_vs_service_rate()
    plot_peak_aoi_vs_packet_size()