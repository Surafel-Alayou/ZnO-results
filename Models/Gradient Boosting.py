import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  

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
X.fillna(X.mean(), inplace=True) 

y = ZnO_data_set.iloc[:, -1]
y.fillna(y.mean(), inplace=True)  

# Splitting dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state = SEED)

# Instantiate Gradient Boosting Regressor
gbr = GradientBoostingRegressor(
                                loss='squared_error', # ‘squared_error’, ‘absolute_error’, ‘huber’, ‘quantile’
                                learning_rate=0.32, # [0.01, 1] step of 0.01
                                n_estimators=101,  # [100, 400] step of 1
                                max_depth = 8, # [1, 15] step of 1
                                random_state = SEED,
                                max_features = 1, # [1, 10] step of 1
                                subsample=0.5, # (0.0, 1.0] step of 0.01
                                )
 
# Fit to training set
gbr.fit(train_X, train_y)

# Predict on test set
pred_y = gbr.predict(test_X)

font = {'family': 'Calibri',
        'weight': 'normal',
        'size': 20,
        }

# Scatter plot of test_y against test predictions
plt.scatter(test_y, pred_y, s=150, edgecolor='#000000', color='#00a053', linewidths=0.6, label='data')
m, b = np.polyfit(test_y, pred_y, 1)
plt.plot([0, 100], [0, 100], color='#febd15', linewidth=1.0, label='fit')
r2_test = r2_score(test_y, pred_y)
plt.title(f'GB: R\u00b2 = {r2_test:.4f}', fontfamily='Calibri', fontsize=24)
plt.xlabel('Actual (nm)', fontdict=font)
plt.ylabel('Predicted (nm)', fontdict=font)
plt.xticks(fontfamily='Calibri', fontsize=22)
plt.yticks(fontfamily='Calibri', fontsize=22)
plt.legend(prop={'size': 24, 'weight': 'normal','family': 'Calibri'})
plt.show()

# Model evaluation 
print(f"r_square_for_the_test_dataset: {r2_score(test_y, pred_y):.4f}") 
print(f"mean_absolute_error : {mean_absolute_error(test_y, pred_y):.4f}")
print(f"root_mean_squared_error : {mean_squared_error(test_y, pred_y, squared=False):.4f}")

