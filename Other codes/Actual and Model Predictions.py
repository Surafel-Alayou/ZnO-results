import pandas as pd
import matplotlib.pyplot as plt

# Data
y1 = [35, 25, 39.4, 44.7, 51.23, 51.86, 30, 40, 26, 27, 21.7, 16.9, 50, 40, 50, 27.81, 49.44, 40, 25, 40]
y2 = [35.56, 33.03, 41.79, 40.97, 39.85, 48.41, 43.94, 42.63, 30.99, 26.6, 26.8, 26.4, 46.02, 42.29, 44.73, 31.47, 42.69, 37.25, 28.58, 39.99]

# Create DataFrame
data = {
    'Actual': y1,
    'Model': y2,
}

df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 10))

for column in df.columns:
    plt.plot(df.index + 1, df[column], marker='o', linewidth=1.5, alpha=0.9, label=column)

# Custom font
font = {'family': 'Calibri', 'weight': 'normal', 'size': 20}

plt.xlabel("Index", fontdict=font)
plt.ylabel("Nanoparticle size (nm)", fontdict=font)
plt.legend(prop={'size': 22, 'weight': 'normal', 'family': 'Calibri'})
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.xticks(fontfamily='Calibri', fontsize=22)
plt.yticks(fontfamily='Calibri', fontsize=22)
plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
plt.show()
