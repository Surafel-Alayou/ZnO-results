# zno_models_python

This repository contains machine learning models developed to predict the sizes of ZnO nanoparticles based on nine features. These models include CatBoost, Gradient Boosting, Extreme Gradient Boosting, and their stacking ensemble. The features used for prediction are:

- Energy Band Gap (eV)
- Reaction Temperature (°C)
- Calcination Temperature (°C)
- Reaction Duration (hr)
- Calcination Duration (hr)
- Synthesis Method
- Precursor
- pH
- Precursor Concentration (M)

## Features

- **Machine Learning Models**: CatBoost, Gradient Boosting, Extreme Gradient Boosting, Stacking.
- **Hyperparameter Tuning**: Each model's hyperparameters have been meticulously tuned to achieve the best performance.
- **Performance Evaluation**: The models have been evaluated using several metrics, with stacking showing the best performance with an R² score of 0.9377.

## Contents

- **Models**: Implementations of CatBoost, Gradient Boosting, Extreme Gradient Boosting, and Stacking.
- **Dataset**: The dataset used for training and evaluation, containing the aforementioned features and ZnO nanoparticle sizes.
- **Code**: Scripts for training, hyperparameter tuning, and evaluating the models.

## Results

- **Stacking**: Best model with an R² score of 0.9377.
- **XGB**: Lowest performing model with an R² score of 0.7823.

## Dataset

The dataset (ZnO_dataset.csv) includes all necessary features to train and evaluate the models.

## Contributing

Feel free to submit issues or pull requests if you have any improvements or suggestions.

You can access the code and dataset here and start experimenting with the models to predict ZnO nanoparticle sizes.
