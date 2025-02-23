import matplotlib.pyplot as plt
import numpy as np

# Load the metrics data
import json
with open('./inference_metrics.json', 'r') as f:
    metricsData = json.load(f)

# Process AoI Time Series data
aoiData = [{'index': i, 'aoi': v} for i, v in enumerate(metricsData['raw_data']['aoi_values'][:100])]

# Process Policy Distribution
policyData = [
    {'name': 'Continuous Update (CU)', 'value': metricsData['summary_stats']['policy_distribution']['CU']},
    {'name': 'Zero Wait (ZW)', 'value': metricsData['summary_stats']['policy_distribution']['ZW']}
]

# Process Service Rate vs Arrival Rate data
ratesData = [
    {'metric': 'Service Rate (μ)', 'mean': metricsData['summary_stats']['service_rate']['mean'], 'std': metricsData['summary_stats']['service_rate']['std']},
    {'metric': 'Arrival Rate (λ)', 'mean': metricsData['summary_stats']['arrival_rate']['mean'], 'std': metricsData['summary_stats']['arrival_rate']['std']}
]

# Plot the graphs
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# AoI Time Series
ax = axes[0, 0]
ax.plot([d['aoi'] for d in aoiData])
ax.set_xlabel('Sample Index')
ax.set_ylabel('AoI (seconds)')
ax.set_title('Age of Information Over Time')

# Policy Distribution
ax = axes[0, 1]
labels = [d['name'] for d in policyData]
values = [d['value'] for d in policyData]
ax.pie(values, labels=labels, autopct='%1.1f%%')
ax.set_title('Policy Distribution')

# Service Rate vs Arrival Rate
ax = axes[1, 0]
bar_width = 0.4
x = np.arange(len(ratesData))
ax.bar(x - bar_width/2, [d['mean'] for d in ratesData], bar_width, label='Mean Rate')
ax.bar(x + bar_width/2, [d['std'] for d in ratesData], bar_width, label='Standard Deviation')
ax.set_xticks(x)
ax.set_xticklabels([d['metric'] for d in ratesData])
ax.set_ylabel('Rate')
ax.set_title('Service Rate vs Arrival Rate')
ax.legend()

# Performance Summary
ax = axes[1, 1]
ax.text(0.5, 0.8, f"Average AoI: {metricsData['summary_stats']['aoi']['mean']:.2f} s", ha='center', va='center', transform=ax.transAxes)
ax.text(0.5, 0.6, f"λ/μ Ratio: {metricsData['summary_stats']['ratio']['mean']:.2f}", ha='center', va='center', transform=ax.transAxes)
ax.text(0.5, 0.4, f"Service Rate: {metricsData['summary_stats']['service_rate']['mean']:.2f} req/s", ha='center', va='center', transform=ax.transAxes)
ax.text(0.5, 0.2, f"Duration: {metricsData['summary_stats']['duration']:.1f} s", ha='center', va='center', transform=ax.transAxes)
ax.axis('off')
ax.set_title('Performance Summary')

plt.tight_layout()
plt.show()