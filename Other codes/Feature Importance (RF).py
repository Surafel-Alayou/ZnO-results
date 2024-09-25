import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
                                                    test_size=0.25, 
                                                    random_state=SEED)

# Instantiate Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=122, # [100, 400] step of 1
    max_features=1, # [1, 15] step of 1
    min_samples_split=2, # [2, 10] step of 1
    min_samples_leaf=1, # [1, 10] step of 1
    random_state=SEED
)

# Fit to training set
rf.fit(train_X, train_y)

# Predict on test set
pred_y = rf.predict(test_X)

# Compute feature importance
feature_importance = rf.feature_importances_

# Get feature names
feature_names = ZnO_data_set.columns[:-1]

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

font = {'family': 'Calibri',
        'weight': 'normal',
        'size': 20,
        }

# Plot feature importance
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xticks(fontfamily='Calibri', fontsize=22)
plt.yticks(fontfamily='Calibri', fontsize=22)
plt.xlabel('Importance', fontdict=font)
plt.gca().invert_yaxis()
plt.show()

# Model evaluation 
print(f"r_square_for_the_test_dataset: {r2_score(test_y, pred_y):.4f}") 
print(f"mean_absolute_error : {mean_absolute_error(test_y, pred_y):.4f}")
print(f"root_mean_squared_error : {mean_squared_error(test_y, pred_y, squared=False):.4f}")
