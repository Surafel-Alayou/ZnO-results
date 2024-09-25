import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# These data are taken from the validation dataset of row 1, 3 and 19
x_data = [
    np.array([2.67, 2.87, 3.07, 3.27, 3.47]),  # Varied by 0.2
    np.array([2.7, 2.9, 3.1, 3.3, 3.5]),  # Varied by 0.2
    np.array([2.7, 2.9, 3.1, 3.3, 3.5])  # Varied by 0.2
]
y_data = [
    np.array([35.56, 31.97, 25.71, 29.52, 28.98]),
    np.array([41.79, 37.53, 34.99, 34.86, 30.46]),
    np.array([35.44, 31.27, 28.58, 23.65, 21.43])
]
labels = ['Sample row 1', 'Sample row 2', 'Sample row 3']

# Custom font
font = {'family': 'Calibri', 'weight': 'normal', 'size': 20}

plt.figure(figsize=(10, 10))

plt.xlabel("Energy band gap (eV)", fontdict=font)
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
