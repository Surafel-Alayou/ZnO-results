import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# These data are taken from the validation dataset of row 1, 3 and 19
x_data = [
    np.array([500, 400, 300, 200, 100]),  # Varied by 100
    np.array([150, 250, 350, 450, 550]),  # Varied by 100
    np.array([600, 500, 400, 300, 200])  # Varied by 100
]
y_data = [
    np.array([42.8, 35.56, 40.53, 41.30, 44.77]),
    np.array([41.79, 40.97, 39.86, 48.41, 49.33]),
    np.array([37.70, 41.70, 28.58, 33.46, 34.68])
]
labels = ['Sample row 1', 'Sample row 2', 'Sample row 3']

# Custom font
font = {'family': 'Calibri', 'weight': 'normal', 'size': 20}

plt.figure(figsize=(10, 10))

plt.xlabel("Calcination temperature (°C)", fontdict=font)
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
