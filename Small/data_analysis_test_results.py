import pandas as pd
import os
import ast
import json

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

# Select only numeric columns for correlation analysis
numeric_columns = numeric_params + ['metric', 'accuracy', 'time_taken']
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

# Output the results
print("Most relevant parameters affecting 'metric':")
print(most_relevant_metric_params)
print("\nMost relevant parameters affecting 'time_taken':")
print(most_relevant_time_params)
print("\nBest performing configuration:")
print(best_performance)
print("\nAverage time for different parameter values:")
for param, times in param_time_analysis.items():
    print(f"\n{param}:")
    print(times)

# Save the analysis results to a file
with open('analysis_results.txt', 'w') as f:
    f.write("Most relevant parameters affecting 'metric':\n")
    f.write(most_relevant_metric_params.to_string())
    f.write("\n\nMost relevant parameters affecting 'time_taken':\n")
    f.write(most_relevant_time_params.to_string())
    f.write("\n\nBest performing configuration:\n")
    f.write(best_performance.to_string())
    f.write("\n\nAverage time for different parameter values:\n")
    for param, times in param_time_analysis.items():
        f.write(f"\n{param}:\n")
        f.write(times.to_string())