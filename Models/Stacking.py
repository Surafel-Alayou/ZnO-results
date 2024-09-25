import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
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


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

cat_model = CatBoostRegressor(
                            iterations=101, # [100, 400] step of 1
                            learning_rate=0.96, # [0.01, 1] step of 0.01
                            depth=12, # [1, 15] step of 1
                            l2_leaf_reg=1, # [0, 5] step of 1
                            random_strength=1.4, # [0, 5] step of 0.1
                            border_count=256, # [128, 512] step of 128
                            random_seed=SEED
                            )

xgb_model = XGBRegressor(
                            learning_rate=0.63, # [0.01, 1] step of 0.01
                            n_estimators=100, # [100, 400] step of 1
                            max_depth=3, # [1, 15] step of 1
                            random_state=SEED, 
                            subsample=1.0,  # [0, 1] step of 0.1
                            colsample_bytree = 0.6, # [0, 1] step of 0.1
                            gamma=0.59 # [0, 1] step of 0.01
                            )

gb_model = GradientBoostingRegressor(
                                loss='squared_error', # ‘squared_error’, ‘absolute_error’, ‘huber’, ‘quantile’
                                learning_rate=0.32, # [0.01, 1] step of 0.01
                                n_estimators=101,  # [100, 400] step of 1
                                max_depth = 8, # [1, 15] step of 1
                                random_state = SEED,
                                max_features = 1, # [1, 10] step of 1
                                subsample=0.5, # (0.0, 1.0] step of 0.01
                                )

# Fit base models on the training data
cat_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Generate predictions from base models
cat_preds = cat_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)
gb_preds = gb_model.predict(X_test)

# Create a new dataset with base model predictions as features
stacking_X = pd.DataFrame({'cat_pred': cat_preds, 'xgb_pred': xgb_preds, 'gb_pred': gb_preds})

# Meta-model
meta_model = LinearRegression()

# Fit the meta-model on the base model predictions
meta_model.fit(stacking_X, y_test)

# Generate final predictions using the meta-model
stacking_preds = meta_model.predict(stacking_X)

font = {'family': 'Calibri',
        'weight': 'normal',
        'size': 20,
        }

# Scatter plot of test_y against predicted values
plt.scatter(y_test, stacking_preds, s=150, edgecolor='#000000', color='#00a053', linewidths=0.6, label='data')
plt.plot([0, 100], [0, 100], color='#febd15', linewidth=1.0, label='fit')
plt.xlabel('Actual (nm)', fontdict=font)
plt.ylabel('Predicted (nm)', fontdict=font)
plt.xticks(fontfamily='Calibri', fontsize=22)
plt.yticks(fontfamily='Calibri', fontsize=22)
r2_stack = r2_score(y_test, stacking_preds)
plt.title(f'Stacking: R\u00b2 = {r2_stack:.4f}', fontfamily='Calibri', fontsize=20)
plt.legend(prop={'size': 24, 'weight': 'normal','family': 'Calibri'})
plt.show()

# Model evaluation 
print(f"r_square_for_the_model: {r2_score(y_test, stacking_preds):.4f}") 
print(f"mean_absolute_error : {mean_absolute_error(y_test, stacking_preds):.4f}")
print(f"root_mean_squared_error : {mean_squared_error(y_test, stacking_preds, squared=False):.4f}")
