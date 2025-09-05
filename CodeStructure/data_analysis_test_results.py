import pandas as pd
import os
import ast
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import sys
import shutil
from datetime import datetime

# Set up argument parsing
parser = argparse.ArgumentParser(description='Analyze test results from a CSV file.')
parser.add_argument('--file', '-f', type=str, help='Path to the test_results.csv file')
parser.add_argument('--output-dir', '-o', type=str, help='Directory to save plots', default='plots')
parser.add_argument('--clean', '-c', action='store_true', help='Clean previous plots directory')
args = parser.parse_args()

# Set aesthetics for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Determine the file path: first try user-specified path, then common locations
if args.file:
    file_path = args.file
else:
    # Look in these locations in order
    possible_locations = [
        os.path.join(script_dir, 'test_results.csv'),
        os.path.join(script_dir, 'datasets', 'results', 'test_results.csv'),
        *[os.path.join(script_dir, 'datasets', 'results', d, 'test_results.csv') 
          for d in sorted([d for d in os.listdir(os.path.join(script_dir, 'datasets', 'results')) 
                          if os.path.isdir(os.path.join(script_dir, 'datasets', 'results', d))], 
                         reverse=True) 
          if os.path.exists(os.path.join(script_dir, 'datasets', 'results', d, 'test_results.csv'))]
    ]
    
    file_path = None
    for loc in possible_locations:
        if os.path.exists(loc):
            file_path = loc
            print(f"Found results file at: {loc}")
            break
    
    if file_path is None:
        print("Error: Could not find test_results.csv in any of the expected locations.")
        print("\nPossible solutions:")
        print("1. Run test_LSEnsemble.py first to generate the results file")
        print("2. Specify the file path using the --file option:")
        print("   python data_analysis_test_results.py --file /path/to/your/test_results.csv")
        print("\nCommon locations to check:")
        for loc in possible_locations:
            print(f"- {loc}")
        sys.exit(1)

# Try to read the CSV file
try:
    data = pd.read_csv(file_path)
    print(f"Successfully loaded data from {file_path} with {len(data)} records")
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    sys.exit(1)

# Get the directory containing the results file
results_dir = os.path.dirname(os.path.abspath(file_path))
plots_dir = os.path.join(results_dir, "plots")

# Always clean previous plots directory
if os.path.exists(plots_dir):
    print(f"Cleaning previous plots directory: {plots_dir}")
    shutil.rmtree(plots_dir)

# Create fresh plots directory 
os.makedirs(plots_dir, exist_ok=True)
print(f"Plots will be saved to: {plots_dir}")

# Function to safely convert string representations of dictionaries to actual dictionaries
def parse_dict_string(dict_str):
    try:
        # Using ast.literal_eval is safer than eval()
        return ast.literal_eval(dict_str)
    except (ValueError, SyntaxError):
        return {}

# Extract numeric parameters from the 'params' dictionary strings
params_dicts = data['params'].apply(parse_dict_string)

# Create new columns for numeric parameters we're interested in
key_params = ['alpha', 'beta', 'Q_RB_S', 'Q_RB_C', 'num_experts', 'hidden_size', 
              'drop_out', 'n_batch', 'n_epoch', 'input_size']

for param in key_params:
    data[param] = params_dicts.apply(lambda x: x.get(param, None))

# Add label switching percentages from the dataset (these are actual label switching metrics, not parameters)
data['label_switching_pct_pos'] = data['pct_pos_switched'].apply(lambda x: float(x.strip('%')) / 100 if isinstance(x, str) else x)
data['label_switching_pct_neg'] = data['pct_neg_switched'].apply(lambda x: float(x.strip('%')) / 100 if isinstance(x, str) else x)
data['label_switching_ratio'] = data['label_switching_pct_pos'] / data['label_switching_pct_neg'].replace(0, np.nan)

# Function to safely save plots and track progress
def save_plot(fig, filename):
    try:
        full_path = os.path.join(plots_dir, filename)
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {full_path}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")

# 1. Alpha vs Beta Performance Analysis
print("Creating Alpha vs Beta heatmap...")
plt.figure(figsize=(12, 8))
pivot_table = pd.pivot_table(data, values='metric', index='alpha', columns='beta', aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
plt.title('Performance Metric by Alpha and Beta Values', fontsize=16)
plt.xlabel('Beta Value', fontsize=14)
plt.ylabel('Alpha Value', fontsize=14)
plt.tight_layout()
save_plot(plt.gcf(), 'alpha_beta_heatmap.png')
plt.close()

# 2. Q_RB_S vs Q_RB_C Performance Analysis
print("Creating Q_RB_S vs Q_RB_C heatmap...")
plt.figure(figsize=(12, 8))
pivot_table = pd.pivot_table(data, values='metric', index='Q_RB_S', columns='Q_RB_C', aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5)
plt.title('Performance Metric by Q_RB_S and Q_RB_C Values', fontsize=16)
plt.xlabel('Q_RB_C (Classification Cost)', fontsize=14)
plt.ylabel('Q_RB_S (Rebalancing Factor)', fontsize=14)
plt.tight_layout()
save_plot(plt.gcf(), 'qrbs_qrbc_heatmap.png')
plt.close()

# 3. Alpha vs Q_RB_S Performance Analysis
print("Creating Alpha vs Q_RB_S heatmap...")
plt.figure(figsize=(12, 8))
pivot_table = pd.pivot_table(data, values='metric', index='alpha', columns='Q_RB_S', aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="coolwarm", linewidths=0.5)
plt.title('Performance Metric by Alpha and Q_RB_S Values', fontsize=16)
plt.xlabel('Q_RB_S (Rebalancing Factor)', fontsize=14)
plt.ylabel('Alpha Value', fontsize=14)
plt.tight_layout()
save_plot(plt.gcf(), 'alpha_qrbs_heatmap.png')
plt.close()

# 4. Beta vs Q_RB_C Performance Analysis
print("Creating Beta vs Q_RB_C heatmap...")
plt.figure(figsize=(12, 8))
pivot_table = pd.pivot_table(data, values='metric', index='beta', columns='Q_RB_C', aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="plasma", linewidths=0.5)
plt.title('Performance Metric by Beta and Q_RB_C Values', fontsize=16)
plt.xlabel('Q_RB_C (Classification Cost)', fontsize=14)
plt.ylabel('Beta Value', fontsize=14)
plt.tight_layout()
save_plot(plt.gcf(), 'beta_qrbc_heatmap.png')
plt.close()

# 5. 3D Visualization of Q_RB_S, Q_RB_C and Performance
print("Creating 3D visualization of Q_RB_S, Q_RB_C and Performance...")
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    data['Q_RB_S'], 
    data['Q_RB_C'], 
    data['metric'],
    c=data['metric'], 
    s=100,
    alpha=0.7,
    cmap='viridis'
)
ax.set_xlabel('Q_RB_S (SMOTE Rebalancing Factor)', fontsize=12)
ax.set_ylabel('Q_RB_C (Classification Cost Factor)', fontsize=12)
ax.set_zlabel('Performance Metric', fontsize=12)
plt.colorbar(scatter, label='Performance Metric')
plt.title('Interaction Between SMOTE (Q_RB_S) and Cost Factor (Q_RB_C)', fontsize=16)
plt.tight_layout()
save_plot(plt.gcf(), 'qrbs_qrbc_3d.png')
plt.close()

# 6. Parameter interactions through pair plots
print("Creating parameter interaction plots (this may take a while)...")
# Select relevant parameters for the pair plot
pair_params = ['alpha', 'beta', 'Q_RB_S', 'Q_RB_C', 'metric', 'time_taken']
pair_data = data[pair_params].copy()

# Create pair plot
g = sns.pairplot(
    pair_data, 
    hue='metric',  
    palette='viridis',
    diag_kind='kde',
    plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
    height=2.5
)
g.fig.suptitle('Parameter Interactions and Their Effect on Performance', fontsize=16, y=1.02)
plt.tight_layout()
save_plot(g.fig, 'parameter_interactions.png')
plt.close()

# 7. Effect of Q_RB_S (SMOTE) on actual label switching percentages
print("Creating Q_RB_S effect on label switching plots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Q_RB_S effect on positive label switching
sns.boxplot(x='Q_RB_S', y='label_switching_pct_pos', data=data, ax=axes[0])
axes[0].set_title('Effect of SMOTE (Q_RB_S) on Positive Label Switching %', fontsize=14)
axes[0].set_xlabel('Q_RB_S (SMOTE Rebalancing Factor)', fontsize=12)
axes[0].set_ylabel('% Positive Labels Switched', fontsize=12)

# Q_RB_S effect on negative label switching
sns.boxplot(x='Q_RB_S', y='label_switching_pct_neg', data=data, ax=axes[1])
axes[1].set_title('Effect of SMOTE (Q_RB_S) on Negative Label Switching %', fontsize=14)
axes[1].set_xlabel('Q_RB_S (SMOTE Rebalancing Factor)', fontsize=12)
axes[1].set_ylabel('% Negative Labels Switched', fontsize=12)

plt.tight_layout()
save_plot(fig, 'qrbs_label_switching_effect.png')
plt.close()

# 8. Performance vs Time by Q_RB_S and Q_RB_C
print("Creating performance vs time plot...")
plt.figure(figsize=(12, 9))
scatter = plt.scatter(
    data['time_taken'],
    data['metric'],
    c=data['Q_RB_S'], 
    s=data['Q_RB_C']*50,
    alpha=0.7,
    cmap='viridis',
    edgecolors='k'
)

# Add annotations for selected points
for i, row in data.iterrows():
    if i % 3 == 0:  # Annotate every 3rd point to avoid crowding
        plt.annotate(
            f'SMOTE={row["Q_RB_S"]},Cost={row["Q_RB_C"]}',
            (row['time_taken'], row['metric']),
            fontsize=8,
            xytext=(5, 5),
            textcoords='offset points'
        )

# Add colorbar and legend
cbar = plt.colorbar(scatter, label='Q_RB_S (SMOTE Factor)')
plt.scatter([], [], s=50, edgecolors='k', facecolors='none', label='Q_RB_C=1')
plt.scatter([], [], s=100, edgecolors='k', facecolors='none', label='Q_RB_C=2')
plt.scatter([], [], s=250, edgecolors='k', facecolors='none', label='Q_RB_C=5')
plt.legend(title='Cost Factor Values', loc='upper right')

plt.title('Performance vs Time Trade-off by SMOTE and Cost Factors', fontsize=16)
plt.xlabel('Execution Time (seconds)', fontsize=14)
plt.ylabel('Performance Metric', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_plot(plt.gcf(), 'performance_time_by_rebalancing.png')
plt.close()

# 9. Cost efficiency analysis - Metric improvement per time increase
print("Creating cost efficiency analysis plots...")

# Calculate normalized values for better comparison
param_metrics = {
    'alpha': {'param': data['alpha'], 'name': 'Alpha', 'color': 'blue'},
    'beta': {'param': data['beta'], 'name': 'Beta', 'color': 'green'},
    'Q_RB_S': {'param': data['Q_RB_S'], 'name': 'SMOTE Factor', 'color': 'red'},
    'Q_RB_C': {'param': data['Q_RB_C'], 'name': 'Cost Factor', 'color': 'purple'}
}

# Figure for time increase by parameter value
plt.figure(figsize=(14, 10))

# Create 4 subplots, one for each parameter
for i, (param_key, param_data) in enumerate(param_metrics.items(), 1):
    plt.subplot(2, 2, i)
    
    # Get unique values and calculate mean time for each value
    unique_values = sorted(data[param_key].unique())
    times = [data[data[param_key] == val]['time_taken'].mean() for val in unique_values]
    
    # Calculate relative increase from baseline
    if len(times) > 0 and times[0] > 0:
        relative_increase = [(t / times[0]) for t in times]
        
        # Plot
        plt.plot(unique_values, relative_increase, marker='o', color=param_data['color'], 
                 linewidth=2, markersize=8)
        plt.grid(True, alpha=0.3)
        plt.title(f'Time Increase by {param_data["name"]} Value', fontsize=14)
        plt.xlabel(param_data["name"], fontsize=12)
        plt.ylabel('Relative Time (baseline=1)', fontsize=12)
        
        # Add text showing the slope (time cost)
        if len(unique_values) > 1:
            # Simple linear regression to get slope
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(unique_values, relative_increase)
            plt.text(0.05, 0.95, f'Time cost: {slope:.2f}x per unit increase', 
                     transform=plt.gca().transAxes, fontsize=10, 
                     verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

plt.tight_layout()
save_plot(plt.gcf(), 'time_cost_by_parameter.png')
plt.close()

# 10. Performance efficiency - Metric improvement per parameter unit
plt.figure(figsize=(14, 10))

for i, (param_key, param_data) in enumerate(param_metrics.items(), 1):
    plt.subplot(2, 2, i)
    
    # Get unique values and calculate mean metric for each value
    unique_values = sorted(data[param_key].unique())
    metrics = [data[data[param_key] == val]['metric'].mean() for val in unique_values]
    
    # Calculate relative improvement from baseline
    if len(metrics) > 0 and metrics[0] > 0:
        relative_improvement = [(m / metrics[0]) for m in metrics]
        
        # Plot
        plt.plot(unique_values, relative_improvement, marker='o', color=param_data['color'], 
                 linewidth=2, markersize=8)
        plt.grid(True, alpha=0.3)
        plt.title(f'Metric Improvement by {param_data["name"]}', fontsize=14)
        plt.xlabel(param_data["name"], fontsize=12)
        plt.ylabel('Relative Metric (baseline=1)', fontsize=12)
        
        # Add text showing the slope (performance gain)
        if len(unique_values) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(unique_values, relative_improvement)
            plt.text(0.05, 0.95, f'Performance gain: {slope:.3f}x per unit increase', 
                     transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

plt.tight_layout()
save_plot(plt.gcf(), 'performance_gain_by_parameter.png')
plt.close()

# 11. Efficiency ratio - Metric improvement / Time increase
plt.figure(figsize=(14, 10))

for i, (param_key, param_data) in enumerate(param_metrics.items(), 1):
    plt.subplot(2, 2, i)
    
    # Get unique values
    unique_values = sorted(data[param_key].unique())
    
    if len(unique_values) > 1:
        # Calculate average metrics and times for each parameter value
        metrics = [data[data[param_key] == val]['metric'].mean() for val in unique_values]
        times = [data[data[param_key] == val]['time_taken'].mean() for val in unique_values]
        
        # Calculate efficiency ratios (metric/time)
        efficiency_ratios = [m/t for m, t in zip(metrics, times)]
        
        # Normalize to the baseline for easy comparison
        if efficiency_ratios[0] > 0:
            norm_efficiency = [r / efficiency_ratios[0] for r in efficiency_ratios]
            
            # Plot
            plt.plot(unique_values, norm_efficiency, marker='o', color=param_data['color'], 
                     linewidth=2, markersize=8)
            plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)  # Baseline reference
            plt.grid(True, alpha=0.3)
            plt.title(f'Efficiency Ratio for {param_data["name"]}', fontsize=14)
            plt.xlabel(param_data["name"], fontsize=12)
            plt.ylabel('Normalized Efficiency (metric/time)', fontsize=12)
            
            # Find optimal value
            best_idx = norm_efficiency.index(max(norm_efficiency))
            plt.scatter([unique_values[best_idx]], [norm_efficiency[best_idx]], 
                         s=150, facecolors='none', edgecolors='black')
            plt.text(unique_values[best_idx], norm_efficiency[best_idx], 
                     f'Optimal: {unique_values[best_idx]}', 
                     fontsize=10, ha='center', va='bottom')

plt.tight_layout()
save_plot(plt.gcf(), 'efficiency_ratio_by_parameter.png')
plt.close()

# 12. Comprehensive performance vs cost analysis
plt.figure(figsize=(12, 10))

# Calculate performance-time ratio for each configuration
data['efficiency'] = data['metric'] / data['time_taken']

# Normalize efficiency for better visualization
min_eff = data['efficiency'].min()
max_eff = data['efficiency'].max()
data['norm_efficiency'] = (data['efficiency'] - min_eff) / (max_eff - min_eff)

# Create a scatter plot with alpha, beta controlling position and Q_RB_S, Q_RB_C encoded as size/color
scatter = plt.scatter(
    data['alpha'], 
    data['beta'],
    c=data['Q_RB_S'],
    s=data['norm_efficiency'] * 400 + 50,  # Size based on efficiency
    alpha=0.7,
    cmap='viridis',
    edgecolors='k'
)

# Add efficiency contour lines
from scipy.interpolate import griddata
# Define grid
x_grid = np.linspace(data['alpha'].min(), data['alpha'].max(), 100)
y_grid = np.linspace(data['beta'].min(), data['beta'].max(), 100)
X, Y = np.meshgrid(x_grid, y_grid)

# Grid the data
Z = griddata((data['alpha'], data['beta']), data['norm_efficiency'], (X, Y), method='cubic')

# Add contour lines
contour = plt.contour(X, Y, Z, 6, colors='k', alpha=0.5, linestyles='dashed')
plt.clabel(contour, inline=True, fontsize=8)

# Add annotations for top configurations
top_n = 3
for idx in data.nlargest(top_n, 'norm_efficiency').index:
    row = data.loc[idx]
    plt.annotate(
        f'α={row["alpha"]},β={row["beta"]},S={row["Q_RB_S"]},C={row["Q_RB_C"]}',
        (row['alpha'], row['beta']),
        fontsize=9,
        xytext=(10, 10),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='black', lw=1),
        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
    )

plt.colorbar(scatter, label='SMOTE Factor (Q_RB_S)')
plt.title('Configuration Efficiency Map\n(Size = Efficiency, Color = SMOTE Factor)', fontsize=16)
plt.xlabel('Alpha', fontsize=14)
plt.ylabel('Beta', fontsize=14)
plt.grid(True, alpha=0.2)
plt.tight_layout()
save_plot(plt.gcf(), 'configuration_efficiency_map.png')
plt.close()

# 13. Parameter impact visualization
print("Creating parameter impact visualization...")

# Calculate parameter impact on both metric and time
param_impacts = {}
for param in ['alpha', 'beta', 'Q_RB_S', 'Q_RB_C']:
    # Get unique values for this parameter
    unique_vals = sorted(data[param].unique())
    if len(unique_vals) <= 1:
        continue
        
    # Calculate average metric and time for each value
    metrics = []
    times = []
    for val in unique_vals:
        subset = data[data[param] == val]
        metrics.append(subset['metric'].mean())
        times.append(subset['time_taken'].mean())
    
    # Calculate slopes (impact)
    metric_slope = (metrics[-1] - metrics[0]) / (unique_vals[-1] - unique_vals[0])
    time_slope = (times[-1] - times[0]) / (unique_vals[-1] - unique_vals[0])
    
    # Store normalized impact (to make them comparable)
    param_impacts[param] = {
        'metric_impact': metric_slope / metrics[0] if metrics[0] > 0 else 0,
        'time_impact': time_slope / times[0] if times[0] > 0 else 0,
        'efficiency': metric_slope / time_slope if time_slope > 0 else float('inf')
    }

# Create a bar chart of parameter impacts
plt.figure(figsize=(14, 8))

# Set up bars
params = list(param_impacts.keys())
metric_impacts = [param_impacts[p]['metric_impact'] for p in params]
time_impacts = [param_impacts[p]['time_impact'] for p in params]
efficiency = [param_impacts[p]['efficiency'] for p in params]

# Plot metric impact
ax1 = plt.subplot(1, 2, 1)
bars1 = ax1.bar(params, metric_impacts, color=['blue', 'green', 'red', 'purple'])
ax1.set_title('Impact on Performance Metric', fontsize=14)
ax1.set_ylabel('Normalized Metric Impact', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.3f}', ha='center', fontsize=9)

# Plot time impact
ax2 = plt.subplot(1, 2, 2)
bars2 = ax2.bar(params, time_impacts, color=['blue', 'green', 'red', 'purple'])
ax2.set_title('Impact on Execution Time', fontsize=14)
ax2.set_ylabel('Normalized Time Impact', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.3f}', ha='center', fontsize=9)

plt.tight_layout()
save_plot(plt.gcf(), 'parameter_impact.png')
plt.close()

# Add the new visualizations to the report summary
results_file = os.path.join(plots_dir, 'parameter_analysis_results.txt')
with open(results_file, 'a') as f:
    f.write("\n\n5. PARAMETER COST EFFICIENCY ANALYSIS:\n")
    f.write("   Time cost per parameter increase:\n")
    for param, impact in param_impacts.items():
        f.write(f"   - {param}: {impact['time_impact']:.3f}x relative increase in time\n")
    
    f.write("\n   Performance gain per parameter increase:\n")
    for param, impact in param_impacts.items():
        f.write(f"   - {param}: {impact['metric_impact']:.3f}x relative increase in metric\n")
    
    f.write("\n   Efficiency ratio (performance gain / time cost):\n")
    for param, impact in param_impacts.items():
        if impact['efficiency'] == float('inf'):
            f.write(f"   - {param}: ∞ (performance improves with no significant time increase)\n")
        else:
            f.write(f"   - {param}: {impact['efficiency']:.3f}\n")
    
    f.write("\n\nADDITIONAL VISUALIZATIONS:\n")
    f.write("9. time_cost_by_parameter.png - Shows how execution time increases with each parameter\n")
    f.write("10. performance_gain_by_parameter.png - Shows performance improvement by parameter\n")
    f.write("11. efficiency_ratio_by_parameter.png - Efficiency ratio (metric/time) for each parameter\n")
    f.write("12. configuration_efficiency_map.png - Map showing efficiency of different configurations\n")
    f.write("13. parameter_impact.png - Visualizes relative impact of each parameter on metrics and time\n")

print(f"\nAdditional cost analysis plots have been created and saved to {plots_dir}")