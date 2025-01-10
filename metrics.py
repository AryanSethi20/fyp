import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path


def plot_aoi_correlation(metrics_file):
    # Load data from file
    print(f"Loading data from {metrics_file}")
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Extract the values
    if 'detailed_metrics' in metrics:
        # Convert service times to rates (1/service_time)
        service_rates = [1 / t for t in metrics['detailed_metrics']['service_times']]
        aoi_values = metrics['detailed_metrics']['cu_aoi_values']
    else:
        raise ValueError("File format not recognized. Please ensure it contains detailed_metrics.")

    if len(service_rates) != len(aoi_values):
        raise ValueError(f"Length mismatch: {len(service_rates)} service rates vs {len(aoi_values)} AoI values")

    # Filter data to keep only values between 0 and 10 on the x-axis
    filtered_data = [(sr, aoi) for sr, aoi in zip(service_rates, aoi_values) if 0 <= sr <= 10]
    if not filtered_data:
        raise ValueError("No data points fall within the x-axis range of 0 to 10.")

    service_rates, aoi_values = zip(*filtered_data)

    # Calculate correlation coefficient
    correlation, p_value = stats.pearsonr(service_rates, aoi_values)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Create scatter plot
    plt.scatter(service_rates, aoi_values, alpha=0.5, color='blue', label='Data points')

    # Add correlation line
    z = np.polyfit(service_rates, aoi_values, 1)
    p = np.poly1d(z)
    x_range = np.linspace(min(service_rates), max(service_rates), 100)
    # plt.plot(x_range, p(x_range), "r--",
    #          label=f'Correlation line (r={correlation:.3f})',
    #          alpha=0.8)

    # Add labels and title
    plt.xlabel('Service Rate')
    plt.ylabel('AoI (s)')
    plt.title('AoI vs Service Rate Correlation')

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add correlation information in a text box
    # text = f'Correlation coefficient (r): {correlation:.3f}\n'
    # text += f'P-value: {p_value:.3e}\n'
    # text += f'Slope: {z[0]:.3f}\n'
    # text += f'Y-intercept: {z[1]:.3f}'


    # plt.text(0.05, 0.95, text,
    #          transform=plt.gca().transAxes,
    #          bbox=dict(facecolor='white', alpha=0.8),
    #          verticalalignment='top',
    #          fontsize=10)

    plt.legend()

    # Print statistics
    print("\nCorrelation Analysis:")
    print(f"Number of points: {len(service_rates)}")
    print(f"Service Rate range: {min(service_rates):.3f} to {max(service_rates):.3f} updates/s")
    print(f"AoI range: {min(aoi_values):.3f} to {max(aoi_values):.3f} s")
    print(f"Correlation coefficient (r): {correlation:.3f}")
    print(f"P-value: {p_value:.3e}")
    print(f"Linear regression: y = {z[0]:.3f}x + {z[1]:.3f}")

    # Save and show plot
    plt.savefig('aoi_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    try:
        # Look for metrics file in current directory
        metrics_files = list(Path('.').glob('*metrics*.json'))
        if not metrics_files:
            print("No metrics file found. Please ensure the file exists in the current directory.")
            exit(1)

        # Use the most recent metrics file
        latest_file = max(metrics_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest metrics file: {latest_file}")

        plot_aoi_correlation("subscriber_metrics_20250110_141043.json")

    except Exception as e:
        print(f"Error: {e}")
