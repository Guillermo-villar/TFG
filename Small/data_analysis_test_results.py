import pandas as pd
import os
import ast
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set aesthetics for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'test_results.csv')
data = pd.read_csv(file_path)

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
numeric_params = ['alpha', 'beta', 'Q_RB_S', 'Q_RB_C', 'num_experts', 'hidden_size', 
                 'drop_out', 'n_batch', 'n_epoch', 'input_size']

for param in numeric_params:
    data[param] = params_dicts.apply(lambda x: x.get(param, None))

# Select only numeric columns for correlation analysis, excluding 'accuracy'
numeric_columns = numeric_params + ['metric', 'time_taken']
numeric_data = data[numeric_columns].dropna()

# Analyze the most relevant parameters affecting the 'metric' column
metric_correlation = numeric_data.corr()['metric'].sort_values(ascending=False)
most_relevant_metric_params = metric_correlation.drop('metric').head()

# Analyze the most relevant parameters affecting the 'time_taken' column
time_correlation = numeric_data.corr()['time_taken'].sort_values(ascending=False)
most_relevant_time_params = time_correlation.drop('time_taken').head()

# Find the best performing configuration
best_performance_idx = data['metric'].idxmax()
best_performance = data.loc[best_performance_idx]

# Find average time for each parameter value
param_time_analysis = {}
for param in numeric_params:
    if param in numeric_data.columns:
        param_time_analysis[param] = numeric_data.groupby(param)['time_taken'].mean()

# Create a plots directory if it doesn't exist
plots_dir = os.path.join(script_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# 1. Create a focused correlation heatmap with only the most relevant parameters
key_params = ['metric', 'time_taken', 'n_epoch', 'hidden_size', 'drop_out', 'n_batch']
plt.figure(figsize=(10, 8))
focused_corr = numeric_data[key_params].corr()
sns.heatmap(focused_corr, annot=True, fmt=".2f", cmap="coolwarm", 
            linewidths=0.5, cbar_kws={"shrink": .8})
plt.title("Correlation Between Key Model Parameters", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'key_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Bar plot of parameters affecting performance metric
plt.figure(figsize=(10, 6))
most_relevant_metric_params.plot(kind='bar', color='skyblue')
plt.title('Parameters Most Affecting Performance Metric', fontsize=16)
plt.ylabel('Correlation Coefficient', fontsize=14)
plt.xlabel('Parameter', fontsize=14)
plt.xticks(rotation=45)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'metric_correlation_bars.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Create a powerful scatter plot showing metric vs time trade-off with multiple dimensions
plt.figure(figsize=(12, 9))
scatter = plt.scatter(numeric_data['time_taken'], numeric_data['metric'], 
                     c=numeric_data['n_epoch'], cmap='viridis', 
                     s=numeric_data['hidden_size']*5, alpha=0.7, edgecolors='k')
plt.colorbar(scatter, label='Number of Epochs')
plt.title('Multi-dimensional Analysis: Performance vs Time Trade-off', fontsize=16)
plt.xlabel('Execution Time (seconds)', fontsize=14)
plt.ylabel('Performance Metric (Balanced Accuracy)', fontsize=14)
plt.grid(True, alpha=0.3)

# Add batch size annotations to selected points
for i, (idx, row) in enumerate(numeric_data.iterrows()):
    if i % 5 == 0:  # Label only some points to avoid crowding
        plt.annotate(f'batch={int(row["n_batch"])}', 
                    (row['time_taken'], row['metric']),
                    fontsize=9, 
                    xytext=(5, 5), 
                    textcoords='offset points')

# Add a best performance marker
best_idx = numeric_data['metric'].idxmax()
best_point = numeric_data.loc[best_idx]
plt.scatter(best_point['time_taken'], best_point['metric'], 
           s=200, facecolors='none', edgecolors='red', linewidth=2)
plt.annotate('Best Performance', 
            (best_point['time_taken'], best_point['metric']),
            fontsize=12, 
            xytext=(10, 10), 
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color='red'))

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'performance_time_tradeoff.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Line plot showing how epoch count affects time and metric
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot time vs epochs
color = 'tab:blue'
ax1.set_xlabel('Number of Epochs', fontsize=14)
ax1.set_ylabel('Execution Time (s)', color=color, fontsize=14)
time_by_epoch = numeric_data.groupby('n_epoch')['time_taken'].mean()
ax1.plot(time_by_epoch.index, time_by_epoch.values, color=color, marker='o', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

# Plot metric vs epochs on secondary y-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Performance Metric', color=color, fontsize=14)
metric_by_epoch = numeric_data.groupby('n_epoch')['metric'].mean()
ax2.plot(metric_by_epoch.index, metric_by_epoch.values, color=color, marker='s', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Epoch Count Impact on Performance and Time', fontsize=16)
fig.tight_layout()
plt.savefig(os.path.join(plots_dir, 'epoch_impact.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Confusion Matrix visualization for best model
try:
    best_cm = ast.literal_eval(best_performance['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    # Calculate and display metrics on the plot
    tn, fp, fn, tp = best_cm[0][0], best_cm[0][1], best_cm[1][0], best_cm[1][1]
    sensitivity = tp/(tp+fn) if (tp+fn) > 0 else 0
    specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
    balanced_acc = (sensitivity + specificity)/2
    
    plt.title(f'Confusion Matrix for Best Model\nBalanced Accuracy: {balanced_acc:.4f}', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Add text annotations for key metrics
    plt.figtext(0.15, 0.02, f"Sensitivity: {sensitivity:.4f}", fontsize=12)
    plt.figtext(0.55, 0.02, f"Specificity: {specificity:.4f}", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'best_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
except (ValueError, SyntaxError, KeyError):
    print("Couldn't create confusion matrix visualization")

# Output the results to console
print("\nMOST IMPORTANT VISUALIZATION RESULTS")
print("====================================")
print("\n1. KEY CORRELATION HEATMAP:")
print("   Shows how key parameters correlate with performance metric and execution time")
print("   File: key_correlation_heatmap.png")

print("\n2. PARAMETERS AFFECTING PERFORMANCE:")
print("   Bar chart showing the most significant parameters influencing model performance")
print("   File: metric_correlation_bars.png")
print("   Most relevant parameters:")
print(most_relevant_metric_params)

print("\n3. MULTI-DIMENSIONAL TRADE-OFF ANALYSIS:")
print("   Complex visualization showing the relationship between:")
print("   - Execution time (x-axis)")
print("   - Performance metric (y-axis)")
print("   - Number of epochs (color)")
print("   - Hidden layer size (point size)")
print("   - Batch size (text annotations)")
print("   File: performance_time_tradeoff.png")

print("\n4. EPOCH IMPACT ANALYSIS:")
print("   Dual-axis plot showing how the number of epochs affects both:")
print("   - Execution time (left y-axis)")
print("   - Performance metric (right y-axis)")
print("   File: epoch_impact.png")

print("\n5. BEST MODEL CONFUSION MATRIX:")
print("   Detailed view of the best model's classification performance")
print("   File: best_confusion_matrix.png")
print("   Best performance configuration:")
for param in ['n_epoch', 'hidden_size', 'drop_out', 'n_batch']:
    if param in best_performance:
        print(f"   - {param}: {best_performance[param]}")
print(f"   - metric: {best_performance['metric']}")
print(f"   - time_taken: {best_performance['time_taken']}")

print(f"\nAll visualization plots have been saved to {plots_dir}")

# Save the key analysis results to a file
with open('analysis_results.txt', 'w') as f:
    f.write("KEY DATA ANALYSIS VISUALIZATIONS\n")
    f.write("===============================\n\n")
    
    f.write("1. KEY CORRELATION HEATMAP\n")
    f.write("   Shows how key parameters correlate with performance metric and execution time\n")
    f.write("   File: key_correlation_heatmap.png\n\n")
    
    f.write("2. PARAMETERS AFFECTING PERFORMANCE\n")
    f.write("   Bar chart showing the most significant parameters influencing model performance\n")
    f.write("   File: metric_correlation_bars.png\n")
    f.write("   Most relevant parameters:\n")
    f.write(most_relevant_metric_params.to_string() + "\n\n")
    
    f.write("3. MULTI-DIMENSIONAL TRADE-OFF ANALYSIS\n")
    f.write("   Complex visualization showing the relationship between:\n")
    f.write("   - Execution time (x-axis)\n")
    f.write("   - Performance metric (y-axis)\n")
    f.write("   - Number of epochs (color)\n")
    f.write("   - Hidden layer size (point size)\n")
    f.write("   - Batch size (text annotations)\n")
    f.write("   File: performance_time_tradeoff.png\n\n")
    
    f.write("4. EPOCH IMPACT ANALYSIS\n")
    f.write("   Dual-axis plot showing how the number of epochs affects both:\n")
    f.write("   - Execution time (left y-axis)\n")
    f.write("   - Performance metric (right y-axis)\n")
    f.write("   File: epoch_impact.png\n\n")
    
    f.write("5. BEST MODEL CONFUSION MATRIX\n")
    f.write("   Detailed view of the best model's classification performance\n")
    f.write("   File: best_confusion_matrix.png\n")
    f.write("   Best performance configuration:\n")
    for param in ['n_epoch', 'hidden_size', 'drop_out', 'n_batch']:
        if param in best_performance:
            f.write(f"   - {param}: {best_performance[param]}\n")
    f.write(f"   - metric: {best_performance['metric']}\n")
    f.write(f"   - time_taken: {best_performance['time_taken']}\n\n")
    
    f.write(f"All visualization plots have been saved to {plots_dir}")