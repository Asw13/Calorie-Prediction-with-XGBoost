# Calorie Prediction with XGBoost

This repository contains code for predicting calories burned using an **XGBoost** model on the 2025 Kaggle Playground Series dataset. The project implements feature selection, hyperparameter tuning, and test set predictions, achieving a **Kaggle test RMSLE of 0.06013**.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Feature Selection](#feature-selection)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Training and Prediction](#model-training-and-prediction)
- [Results](#results)
- [Contributing](#contributing)

## Introduction
This project predicts calories burned during physical activity using an XGBoost regression model. The synthetic dataset from the 2025 Kaggle Playground Series includes features like `Sex`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, and `Body_Temp`. Feature selection and hyperparameter tuning optimized performance, addressing underutilized features (`Age`, `Body_Temp`). The final model achieved a validation RMSLE of **0.0616**, R² of **0.9966**, and a Kaggle test RMSLE of **0.06013**.

**Motivation**: Develop a high-accuracy regression model for calorie prediction using advanced feature engineering and tuning.

**Problem Solved**: Predict continuous `Calories` with minimal error, outperforming baseline models (e.g., Linear Regression RMSLE: 0.1399).

## Dataset
- **`train.csv`**: 750,000 entries with features and target `Calories`.
- **`test.csv`**: Test set for predictions (no `Calories`).
- **Columns**: `id`, `Sex`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`, `Calories` (train only).

**Source**: 2025 Kaggle Playground Series.

**Preprocessing**:
- Outlier capping: `Calories` <800, `Heart_Rate` (40–200), `Body_Temp` (36–42).
- Encoding: `Sex` (LabelEncoder), `Age_Category` (one-hot encoded as `Age_Young`, `Age_Middle`, `Age_Old`).
- Scaling: Numerical features with `RobustScaler` (quantile_range: 10–90), fitted on training data.

## Features
The model uses 16 features selected from 24:
- **Original**: `Sex`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`.
- **Engineered**:
  - `Duration_Heart_Rate`: Duration × Heart_Rate
  - `Weight_Duration`: Weight × Duration
  - `Age_Duration`: Age × Duration
  - `Body_Temp_Duration`: Body_Temp × Duration
  - `Body_Temp_Heart_Rate`: Body_Temp × Heart_Rate
  - `Age_Heart_Rate`: Age × Heart_Rate
  - `Body_Temp_Heart_Rate_Duration`: Body_Temp × Heart_Rate × Duration
- **One-Hot Encoded**: `Age_Young`, `Age_Middle`, `Age_Old`.

**Selected Features**:
- `Sex`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`, `Age_Middle`, `Age_Old`, `Age_Young`, `Duration_Heart_Rate`, `Weight_Duration`, `Age_Duration`, `Body_Temp_Duration`, `Body_Temp_Heart_Rate`, `Age_Heart_Rate`, `Body_Temp_Heart_Rate_Duration`

## Installation
### Prerequisites
- Python 3.8+
- pip
- Git


## Usage
Run the script to train the model and generate test predictions:

```bash
python xgboost_calorie_prediction.py
```

**Output**:
- `submission_xgboost.csv`: `id` and predicted `Calories` for the test set.
- Console: Validation RMSLE, R², feature importance, `Age_Category` statistics.

**Example**:
```python
import pandas as pd
import xgboost as xgb

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Train model
model = xgb.XGBRegressor(learning_rate=0.05, max_depth=7, n_estimators=300)
model.fit(X_train[best_features], y_train)

# Predict
test_preds = model.predict(X_test[best_features])
submission = pd.DataFrame({'id': test_df['id'], 'Calories': test_preds})
submission.to_csv('submission_xgboost.csv', index=False)
```

## Feature Selection
- **Method**: Greedy sequential elimination outperformed random subset sampling (1,000 subsets).
- **Process**:
  1. Start with 24 features.
  2. Drop one feature at a time, evaluate RMSE/R² on 80–20 validation split.
- **Results**:
  - **Best Subset**: 16 features (listed above)
  - **RMSE**: 0.0638
  - **R²**: 0.9964

## Hyperparameter Tuning
GridSearchCV optimized XGBoost:
- **Parameters**: `learning_rate`, `max_depth`, `n_estimators`.
- **Best Parameters**: `{'learning_rate': 0.05, 'max_depth': 7, 'n_estimators': 300}`
- **Validation**:
  - RMSLE: 0.0616
  - R²: 0.9966

## Model Training and Prediction
- **Model**: XGBoost (`learning_rate=0.05`, `max_depth=7`, `n_estimators=300`).
- **Training**: On full `train.csv` with 16 features.
- **Prediction**: On `test.csv`, negative `Calories` clipped to 0.
- **Kaggle Score**: Test RMSLE: **0.06013**

## Results
- **Submission**: `submission_xgboost.csv` achieved a Kaggle test RMSLE of **0.06013**.
- **Insights**:
  - Older individuals (`Age_Category` = Old) burn more calories, have faster calorie-burning rates, and lower heart rates.
  - Engineered features (`Age_Heart_Rate`, `Body_Temp_Heart_Rate_Duration`) enhance `Age` and `Body_Temp`.
- **Statistics** (example):
  ```
  Average Calories by Age Category:
  Young: 150.2
  Middle: 180.5
  Old: 200.7

  Average Calorie-Burning Rate (kcal/min):
  Young: 5.1
  Middle: 6.0
  Old: 6.8

  Average Heart Rate:
  Young: 100.5
  Middle: 98.2
  Old: 95.7
  ```

## Contributing
1. Fork the repository.
2. Create a feature branch: `git checkout -b my-feature`.
3. Commit: `git commit -m 'Add feature'`.
4. Push: `git push origin my-feature`.
5. Open a pull request.



