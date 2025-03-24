import json
import numpy as np
import matplotlib.pyplot as plt

def interarrival_time_constant():
    files_zw = ["/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_ZW_mu5.0_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_ZW_mu5.0_lambda1.5.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_ZW_mu5.0_lambda2.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_ZW_mu5.0_lambda2.5.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_ZW_mu5.0_lambda3.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_ZW_mu5.0_lambda3.5.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_ZW_mu5.0_lambda4.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_ZW_mu5.0_lambda4.5.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_ZW_mu5.0_lambda5.0.json"]
    files_cu = ["/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_CU_mu5.0_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_CU_mu5.0_lambda1.5.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_CU_mu5.0_lambda2.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_CU_mu5.0_lambda2.5.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_CU_mu5.0_lambda3.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_CU_mu5.0_lambda3.5.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_CU_mu5.0_lambda4.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_CU_mu5.0_lambda4.5.json",
                "/Users/aryansethi20/Downloads/fyp/testing/service_times/metrics_CU_mu5.0_lambda5.0.json"]

    zw_aoi_vals = []
    cu_aoi_vals = []
    for file in files_zw:
        aoi_values = []
        with open(file, 'r') as f:
            data = json.load(f)
            
        # Skip first 50 samples (warm-up period)
        # aoi_values = data.get('theoretical_aoi_values', data.get('aoi_values'))[50:3050]
        aoi_values = data.get('theoretical_aoi_values')[-200:]
        zw_aoi_vals.append(np.mean(aoi_values))

    for file in files_cu:
        aoi_values = []
        with open(file, 'r') as f:
            data = json.load(f)
            
        # Skip first 50 samples (warm-up period)
        # aoi_values = data.get('theoretical_aoi_values', data.get('aoi_values'))[50:3050]
        aoi_values = data.get('theoretical_aoi_values')[-200:]
        cu_aoi_vals.append(np.mean(aoi_values))

    interarrival_times = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(interarrival_times, zw_aoi_vals, marker='o', markersize=6, label='ZW Policy',
             linestyle='-', linewidth=2, color='red')
    plt.plot(interarrival_times, cu_aoi_vals, marker='o', markersize=6, label='CU Policy',
             linestyle='-', linewidth=2, color='blue')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Server Utilisation Ratios', fontsize=12)
    plt.ylabel('Age of Information (AoI)', fontsize=12)
    plt.title('AoI vs Server Utilisation Ratios', fontsize=14)

    # Set axis limits
    # plt.xlim(0.9, 3.1)
    # plt.ylim(0, max(cu_singapore) * 1.1)  # Set y limit with some margin

    # Add legend
    plt.legend(loc='upper right')

    # Add parameters text box
    parameters_text = (
        'Parameters:\n'
        '• Mean Service Rate: 5\n'
        '• Number of samples:\n'
        '  10000 (for each\n'
        '  server utilisation ratio)'
    )

    plt.figtext(0.75, 0.25, parameters_text, 
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))

    plt.tight_layout()
    plt.savefig('interarrival_rate_comparison.png', dpi=300, bbox_inches='tight')

def service_time_constant():
    files_zw = ["/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_ZW_mu1.0_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_ZW_mu1.5_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_ZW_mu2.0_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_ZW_mu2.5_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_ZW_mu3.0_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_ZW_mu3.5_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_ZW_mu4.0_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_ZW_mu4.5_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_ZW_mu5.0_lambda1.0.json"]
    files_cu = ["/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_CU_mu1.0_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_CU_mu1.5_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_CU_mu2.0_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_CU_mu2.5_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_CU_mu3.0_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_CU_mu3.5_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_CU_mu4.0_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_CU_mu4.5_lambda1.0.json",
                "/Users/aryansethi20/Downloads/fyp/testing/interarrival_times/metrics_CU_mu5.0_lambda1.0.json"]

    zw_aoi_vals = []
    cu_aoi_vals = []
    for file in files_zw:
        aoi_values = []
        with open(file, 'r') as f:
            data = json.load(f)
            
        # Skip first 50 samples (warm-up period)
        # aoi_values = data.get('theoretical_aoi_values', data.get('aoi_values'))[50:3050]
        aoi_values = data.get('theoretical_aoi_values')[-200:]
        zw_aoi_vals.append(np.mean(aoi_values))

    for file in files_cu:
        aoi_values = []
        with open(file, 'r') as f:
            data = json.load(f)
            
        # Skip first 50 samples (warm-up period)
        # aoi_values = data.get('theoretical_aoi_values', data.get('aoi_values'))[50:3050]
        aoi_values = data.get('theoretical_aoi_values')[-200:]
        cu_aoi_vals.append(np.mean(aoi_values))

    service_times = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(service_times, zw_aoi_vals, marker='o', markersize=6, label='ZW Policy',
             linestyle='-', linewidth=2, color='red')
    plt.plot(service_times, cu_aoi_vals, marker='o', markersize=6, label='CU Policy',
             linestyle='-', linewidth=2, color='blue')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Service Rate', fontsize=12)
    plt.ylabel('Age of Information (AoI)', fontsize=12)
    plt.title('AoI vs Service Rate', fontsize=14)

    # Set axis limits
    # plt.xlim(0.9, 3.1)
    # plt.ylim(0, max(cu_singapore) * 1.1)  # Set y limit with some margin

    # Add legend
    plt.legend(loc='upper right')

    # Add parameters text box
    parameters_text = (
        'Parameters:\n'
        '• Mean Interarrival Rate: 1\n'
        '• Number of samples:\n'
        '  10000 (for each\n'
        '  service rate)'
    )

    plt.figtext(0.75, 0.25, parameters_text, 
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))

    plt.tight_layout()
    plt.savefig('service_rate_comparison.png', dpi=300, bbox_inches='tight')

interarrival_time_constant()