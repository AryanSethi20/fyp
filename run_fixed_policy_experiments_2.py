import subprocess
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def run_experiment(policy, mu, lambda_rate):
    """Run a single experiment with the given parameters"""
    print(f"Starting experiment: policy={policy}, mu={mu}, lambda_rate={lambda_rate}")
    print("Press Enter to stop the experiment and save metrics...")
    
    cmd = [
        "python", "subscriber-fixed-policy_2.py",
        "--policy", policy,
        "--mu", str(mu),
        "--lambda_rate", str(lambda_rate)
    ]
    
    # Start the process
    process = subprocess.Popen(cmd)
    
    # Wait for user input to terminate the experiment
    input()
    
    print(f"Terminating experiment. Please wait while metrics are being saved...")
    
    # Send SIGINT (equivalent to Ctrl+C) instead of SIGTERM
    # This allows the process to run its finally block and save metrics
    import signal
    import os
    import time
    
    # On Windows, we need to use CTRL_C_EVENT
    try:
        os.kill(process.pid, signal.SIGINT)
        # Give time for the process to handle the signal and save metrics
        time.sleep(3)
        # Check if the process is still running
        if process.poll() is None:
            # If still running, force terminate
            process.terminate()
    except Exception as e:
        print(f"Error sending signal: {e}")
        process.terminate()
        time.sleep(3)
    
    # Wait for the process to finish
    process.wait()
    
    # Check if metrics file was created
    filename = f'metrics_{policy}_mu{mu}_lambda{lambda_rate}.json'
    if os.path.exists(filename):
        print(f"Experiment completed, metrics saved to {filename}")
        return filename
    else:
        print(f"Warning: Metrics file {filename} not found. Process may not have saved metrics.")
        return None

def load_metrics(filename):
    """Load metrics from a file"""
    with open(filename, 'r') as f:
        return json.load(f)

def run_all_experiments(experiments):
    """Run all experiments and return results"""
    results = []
    
    for exp in experiments:
        policy, mu, lambda_rate = exp
        metrics_file = run_experiment(policy, mu, lambda_rate)
        
        if metrics_file:
            metrics = load_metrics(metrics_file)
            results.append({
                'policy': policy,
                'mu': mu,
                'lambda_rate': lambda_rate,
                'metrics': metrics
            })
    
    return results

def analyze_results(results):
    """Analyze and compare results"""
    # Create a summary table
    print("\n===== RESULTS SUMMARY =====")
    print("Policy\tμ\tλ\tλ/μ\tAvg AoI\tMax AoI")
    print("------\t-\t-\t---\t-------\t-------")
    
    for result in results:
        policy = result['policy']
        mu = result['mu']
        lambda_rate = result['lambda_rate']
        ratio = lambda_rate / mu
        avg_aoi = result['metrics']['summary']['avg_aoi']
        max_aoi = result['metrics']['summary']['max_aoi']
        
        print(f"{policy}\t{mu:.1f}\t{lambda_rate:.1f}\t{ratio:.2f}\t{avg_aoi:.4f}\t{max_aoi:.4f}")
    
    # Plot comparison
    plot_comparison(results)

def plot_comparison(results):
    """Generate comparative plots"""
    # Group by policy
    cu_results = [r for r in results if r['policy'] == 'CU']
    zw_results = [r for r in results if r['policy'] == 'ZW']
    
    # Plot AoI vs utilization ratio (λ/μ)
    plt.figure(figsize=(10, 6))
    
    if cu_results:
        ratios = [r['lambda_rate']/r['mu'] for r in cu_results]
        avg_aois = [r['metrics']['summary']['avg_aoi'] for r in cu_results]
        plt.plot(ratios, avg_aois, 'bo-', label='CU Policy')
    
    if zw_results:
        ratios = [r['lambda_rate']/r['mu'] for r in zw_results]
        avg_aois = [r['metrics']['summary']['avg_aoi'] for r in zw_results]
        plt.plot(ratios, avg_aois, 'ro-', label='ZW Policy')
    
    plt.xlabel('Utilization Ratio (λ/μ)')
    plt.ylabel('Average AoI (seconds)')
    plt.title('Policy Performance: Average AoI vs Utilization Ratio')
    plt.grid(True)
    plt.legend()
    plt.savefig('policy_comparison.png')
    print("Saved comparison plot to policy_comparison.png")
    
def main():
    # Define experiments: (policy, mu, lambda_rate)
    experiments = [
        ('CU', 5.0, 2.75),
        ('ZW', 5.0, 2.75),
        ('CU', 6.0, 3.3),
        ('ZW', 6.0, 3.3),
        ('CU', 7.0, 3.85),
        ('ZW', 7.0, 3.85),
        ('CU', 8.0, 4.4),
        ('ZW', 8.0, 4.4),
        ('CU', 9.0, 4.95),
        ('ZW', 9.0, 4.95),
    ]
    
    print(f"Running {len(experiments)} experiments one by one")
    print("For each experiment, press Enter when you want to stop and collect metrics")
    results = run_all_experiments(experiments)
    analyze_results(results)

if __name__ == "__main__":
    main()