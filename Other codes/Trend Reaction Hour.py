import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# These data are taken from the validation dataset of row 1, 3 and 19
x_data = [
    np.array([1, 2, 3, 4, 5]),  # Varied by 1
    np.array([2, 3, 4, 5, 6]),  # Varied by 1
    np.array([2, 3, 4, 5, 6])  # Varied by 1
]
y_data = [
    np.array([35.56, 34.35, 30.54, 30.80, 30.73]),
    np.array([41.79, 38.07, 38.10, 38.10, 38.03]),
    np.array([28.58, 27.29, 27.35, 28.15, 28.29])
]
labels = ['Sample row 1', 'Sample row 2', 'Sample row 3']

# Custom font
font = {'family': 'Calibri', 'weight': 'normal', 'size': 20}

plt.figure(figsize=(10, 10))

plt.xlabel("Reaction hour (hr)", fontdict=font)
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
