import numpy as np
import matplotlib.pyplot as plt

# Labels for the x-axis
models = ['GB', 'XGBoost', 'CatBoost', 'Stacking']

# Metrics for each model
metrics = ['R\u00b2', 'RMSE', 'MAE']

# Corresponding values for each label
R2_values = np.array([0.8872, 0.7823, 0.9044, 0.9377])
RMSE_values = np.array([5.2487, 7.2951, 4.8327, 3.8992])
MAE_values = np.array([3.7918, 5.8839, 3.5770, 3.0800])

values = [R2_values, RMSE_values, MAE_values]

# Create a bar chart
x = np.arange(len(models))  # The label locations
width = 0.2  # The width of the bars

fig, ax1 = plt.subplots()

# Create a separate y-axis for R² values
ax2 = ax1.twinx()

# Plot R² values on the first y-axis
bar1 = ax1.bar(x - width, values[0], width, label=metrics[0], color='mediumseagreen')
ax1.set_ylabel(metrics[0], fontdict={'size': 20, 'weight': 'normal', 'family': 'Calibri'})

# Plot RMSE and MAE values on the second y-axis
bar2 = ax2.bar(x, values[1], width, label=metrics[1], color='khaki')
bar3 = ax2.bar(x + width, values[2], width, label=metrics[2], color='cornflowerblue')
ax2.set_ylabel(metrics[1] + ', ' + metrics[2], fontdict={'size': 20, 'weight': 'normal', 'family': 'Calibri'})

# Label the x-axis
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontdict={'size': 22, 'weight': 'normal', 'family': 'Calibri'})

# Set font properties for y-ticks
ax1.tick_params(axis='y', labelsize=22)
ax2.tick_params(axis='y', labelsize=22)

# Set font properties for x-ticks
ax1.tick_params(axis='x', labelsize=22)

# Combine the legends
bars = [bar1, bar2, bar3]
labels = [metrics[0], metrics[1], metrics[2]]

# Add a legend for the combined bars
ax1.legend(bars, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, prop={'size': 24, 'weight': 'normal', 'family': 'Calibri'})

# Display the chart
plt.show()
