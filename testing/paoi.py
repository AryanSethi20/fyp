import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d
import os

def plot_paoi(file_path=None, save_path="peak_age_of_information.png"):
    """
    Plot Peak Age of Information from a metrics file or using synthetic data.
    
    Args:
        file_path: Path to a metrics file containing AoI values, or None to use synthetic data
        save_path: Path to save the figure
    """
    if file_path and os.path.exists(file_path):
        # Load actual data if available
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract AoI values and timestamps
        aoi_values = np.array(data.get('aoi_values', []))
        timestamps = np.array(data.get('timestamps', []))
        policies = data.get('policies', [])
        batch_ids = data.get('batch_ids', [])
        
        if len(timestamps) == 0:
            print("No timestamp data found, using synthetic timestamps")
            timestamps = np.arange(len(aoi_values))
            
        # Normalize timestamps to seconds from start
        if len(timestamps) > 0:
            timestamps = timestamps - timestamps[0]
        
        print(f"Loaded {len(aoi_values)} AoI values from {file_path}")
    else:
        print("Using synthetic PAoI data")
        # Generate synthetic data
        np.random.seed(42)
        num_points = 1000
        
        # Time values in seconds
        timestamps = np.arange(num_points) * 0.1
        
        # Base AoI function (sawtooth pattern typical of AoI)
        aoi_values = np.zeros(num_points)
        
        # Simulate system with different update patterns
        update_times = []
        current_time = 0
        
        while current_time < timestamps[-1]:
            # Random update interval (more realistic)
            interval = np.random.exponential(scale=2.0)  
            current_time += interval
            if current_time < timestamps[-1]:
                update_times.append(current_time)
        
        # Calculate AoI based on update times
        current_aoi = 0
        last_update = 0
        
        for i, t in enumerate(timestamps):
            # Check if there's an update at this time
            update_occurred = False
            for update_time in update_times:
                if last_update < update_time <= t:
                    current_aoi = t - update_time
                    last_update = update_time
                    update_occurred = True
            
            # If no update, AoI increases linearly
            if not update_occurred:
                current_aoi = t - last_update
            
            aoi_values[i] = current_aoi
        
        # Add some noise to make it more realistic
        aoi_values += np.random.normal(0, 0.05, num_points)
        aoi_values = np.maximum(0, aoi_values)  # AoI can't be negative
        
        # Create synthetic policies (alternating between CU and ZW)
        policies = ['CU' if i % 2 == 0 else 'ZW' for i in range(len(aoi_values))]
        
        # Create synthetic batch IDs (changing every 200 points)
        batch_ids = [i // 200 for i in range(len(aoi_values))]
    
    # Calculate Peak AoI
    # Method 1: Find local maxima
    peak_indices = []
    for i in range(1, len(aoi_values) - 1):
        if aoi_values[i] > aoi_values[i-1] and aoi_values[i] >= aoi_values[i+1]:
            peak_indices.append(i)
    
    # Keep only significant peaks (remove small fluctuations)
    avg_aoi = np.mean(aoi_values)
    significant_peak_indices = [i for i in peak_indices if aoi_values[i] > avg_aoi * 1.2]
    
    # If too many peaks, keep only the top N
    max_peaks_to_show = 8
    if len(significant_peak_indices) > max_peaks_to_show:
        # Sort by peak value and keep the highest ones
        significant_peak_indices = sorted(significant_peak_indices, 
                                         key=lambda i: aoi_values[i], 
                                         reverse=True)[:max_peaks_to_show]
        # Re-sort by time
        significant_peak_indices.sort()
    
    # Prepare for policy visualization
    policy_changes = []
    current_policy = policies[0] if policies else None
    for i, policy in enumerate(policies):
        if policy != current_policy:
            policy_changes.append(i)
            current_policy = policy
    
    # Smooth the AoI curve slightly for better visualization
    smoothed_aoi = gaussian_filter1d(aoi_values, sigma=2)
    
    # Compute average PAoI
    peak_values = [aoi_values[i] for i in peak_indices]
    avg_paoi = np.mean(peak_values) if peak_values else np.mean(aoi_values)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot AoI values
    plt.plot(timestamps, aoi_values, 'b-', alpha=0.6, linewidth=0.8, label='Age of Information')
    plt.plot(timestamps, smoothed_aoi, 'b-', linewidth=2, label='Smoothed AoI')
    
    # Highlight peak values
    if significant_peak_indices:
        plt.plot(timestamps[significant_peak_indices], 
                aoi_values[significant_peak_indices], 
                'ro', markersize=8, label='Peak AoI')
        
        # Add value annotations for significant peaks
        for idx in significant_peak_indices:
            plt.annotate(f'{aoi_values[idx]:.2f}', 
                        xy=(timestamps[idx], aoi_values[idx]),
                        xytext=(5, 10), textcoords='offset points',
                        fontsize=10)
    
    # Add horizontal line for average PAoI
    plt.axhline(y=avg_paoi, color='r', linestyle='--', alpha=0.7, 
               label=f'Avg. Peak AoI: {avg_paoi:.2f}')
    
    # Mark policy changes if we have policy data
    if policy_changes:
        for idx in policy_changes[:10]:  # Limit to first 10 changes to avoid clutter
            plt.axvline(x=timestamps[idx], color='g', linestyle='-', alpha=0.3)
            plt.annotate(f"→{policies[idx]}", 
                        xy=(timestamps[idx], max(aoi_values) * 0.9),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=9, rotation=90)
    
    # Add labels and styling
    plt.title('Peak Age of Information (PAoI) Analysis', fontsize=16, pad=20)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Age of Information', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Format y-axis with fewer decimal places
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # Add batch transition markers if we have more than one batch
    if len(set(batch_ids)) > 1:
        batch_transitions = []
        current_batch = batch_ids[0]
        for i, batch in enumerate(batch_ids):
            if batch != current_batch:
                batch_transitions.append(i)
                current_batch = batch
        
        for idx in batch_transitions[:5]:  # Limit to first 5 transitions
            plt.axvline(x=timestamps[idx], color='purple', linestyle='-', alpha=0.2, linewidth=2)
            plt.annotate(f"Batch {batch_ids[idx]}", 
                        xy=(timestamps[idx], max(aoi_values) * 0.8),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=9, rotation=90)
    
    # # Add explanatory text
    # plt.figtext(0.02, 0.02, 
    #            "Peak Age of Information (PAoI) measures the maximum staleness of information at the receiver.\n"
    #            f"Lower values indicate fresher information. Average PAoI: {avg_paoi:.2f}",
    #            fontsize=10, wrap=True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()

def plot_paoi_comparison(file_paths, labels, save_path="paoi_comparison.png"):
    """
    Compare PAoI across different system configurations.
    
    Args:
        file_paths: List of paths to metrics files
        labels: List of labels for each file
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    # Define colors
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
    
    avg_paoi_values = []
    
    for i, (file_path, label) in enumerate(zip(file_paths, labels)):
        if os.path.exists(file_path):
            # Load data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract AoI values
            aoi_values = np.array(data.get('aoi_values', []))
            
            if len(aoi_values) == 0:
                print(f"No AoI values found in {file_path}")
                continue
                
            # Calculate Peak AoI (local maxima)
            peak_indices = []
            for j in range(1, len(aoi_values) - 1):
                if aoi_values[j] > aoi_values[j-1] and aoi_values[j] >= aoi_values[j+1]:
                    peak_indices.append(j)
            
            # Calculate average PAoI
            peak_values = [aoi_values[j] for j in peak_indices]
            avg_paoi = np.mean(peak_values) if peak_values else np.mean(aoi_values)
            avg_paoi_values.append(avg_paoi)
            
            # Get color
            color = colors[i % len(colors)]
            
            # Plot PAoI distribution as histogram
            counts, bins = np.histogram(peak_values, bins=20)
            plt.hist(bins[:-1], bins, weights=counts/sum(counts), 
                    alpha=0.5, color=color, label=f"{label} (Avg PAoI: {avg_paoi:.2f})")
            
            print(f"Processed {file_path} - Avg PAoI: {avg_paoi:.2f}")
        else:
            print(f"File {file_path} not found")
    
    # Add labels and styling
    plt.title('PAoI Comparison Across System Configurations', fontsize=16, pad=20)
    plt.xlabel('Peak Age of Information (PAoI)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add system comparison
    if len(avg_paoi_values) > 1:
        best_idx = np.argmin(avg_paoi_values)
        plt.figtext(0.02, 0.02, 
                   f"System Comparison: '{labels[best_idx]}' achieves the lowest average PAoI ({avg_paoi_values[best_idx]:.2f}),\n"
                   "indicating the freshest information delivery.",
                   fontsize=10, wrap=True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()

# Example usage
if __name__ == "__main__":
    # Option 1: Generate synthetic data
    plot_paoi(file_path=None, save_path="synthetic_paoi.png")
    
    # Option 2: Use actual data from a metrics file
    # If you have actual data files, uncomment and modify this section
    """
    # For single PAoI visualization
    metrics_file = "inference_metrics.json"  # Path to your metrics file
    plot_paoi(file_path=metrics_file, save_path="actual_paoi.png")
    
    # For PAoI comparison across different configurations
    metrics_files = [
        "metrics_config1.json",
        "metrics_config2.json",
        "metrics_config3.json"
    ]
    labels = ["CU Policy", "ZW Policy", "Adaptive Policy"]
    plot_paoi_comparison(metrics_files, labels, save_path="paoi_policy_comparison.png")
    """