import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# These data are taken from the validation dataset of row 1, 3 and 19
x_data = [
    np.array([0.34, 0.39, 0.44, 0.49, 0.54]),  # Varied by 0.05
    np.array([0.1, 0.15, 0.2, 0.25, 0.3]),  # Varied by 0.05
    np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # Varied by 0.05
]
y_data = [
    np.array([35.56, 35.17, 37.88, 32.42, 34.34]),
    np.array([41.79, 44.31, 43.80, 44.34, 45.48]),
    np.array([28.58, 31.02, 31.21, 31.64, 32.35])
]
labels = ['Sample row 1', 'Sample row 2', 'Sample row 3']

# Custom font
font = {'family': 'Calibri', 'weight': 'normal', 'size': 20}

plt.figure(figsize=(10, 10))

plt.xlabel("Precursor concentration (M)", fontdict=font)
plt.ylabel("Nanoparticle size (nm)", fontdict=font)

# Plotting lines with labels in a loop
for x, y, label in zip(x_data, y_data, labels):
    plt.plot(x, y, label=label, linewidth=2.5)

# Adding legend
plt.legend(prop={'size': 22, 'weight': 'normal', 'family': 'Calibri'})

# Grid settings
plt.minorticks_on()
plt.xticks(fontfamily='Calibri', fontsize=22)
plt.yticks(fontfamily='Calibri', fontsize=22)
plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')

plt.show()
