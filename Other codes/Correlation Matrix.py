import pandas as pd
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

# Setting SEED for reproducibility
SEED = 42

# Importing dataset
ZnO_data_set = pd.read_csv('ZnO_dataset.csv')

# Create a separate LabelEncoder object for each categorical column
le_synthesis_method = LabelEncoder()
le_precursor = LabelEncoder()

# Fit the label encoder and transform each categorical column individually
ZnO_data_set["Synthesis method"] = le_synthesis_method.fit_transform(ZnO_data_set["Synthesis method"])
ZnO_data_set["Precursor"] = le_precursor.fit_transform(ZnO_data_set["Precursor"])

# Handling missing values by replacing them with the mean of each feature
X = ZnO_data_set.iloc[:, :-1]
X.fillna(X.mean(), inplace=True)  # Replace missing values with the mean of each feature

y = ZnO_data_set.iloc[:, -1]
y.fillna(y.mean(), inplace=True)  # Replace missing values with the mean of the target variable

# Create a DataFrame from the imputed data
averaged_df = pd.DataFrame(X, columns=ZnO_data_set.columns[:-1])
averaged_df["Target"] = y

# Create a correlation matrix
corr = averaged_df.corr()
my_mask = np.triu(np.ones_like(corr,dtype=bool))

# Generate a heatmap
heatmap = sns.heatmap(corr, cmap="Blues", vmin=-1, vmax=1, mask=my_mask, annot=True, fmt=".4f")

# Set font properties for ticks on both axes
heatmap.set_xticklabels(heatmap.get_xticklabels(), ha='right', fontsize=14, fontname='Calibri')
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14, fontname='Calibri')

# Set font properties for numbers in the cells
for text in heatmap.texts:
    text.set_fontname('Calibri')
    text.set_fontsize(14)

# Set font properties for color bar labels
cbar = heatmap.collections[0].colorbar
cbar.set_ticklabels(cbar.ax.get_yticklabels(), fontsize=14, fontname='Calibri')

plt.show()