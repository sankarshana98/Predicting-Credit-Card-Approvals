# Credit Card Approval Predictor

This project automates the credit card approval process using supervised learning techniques. The goal is to classify credit card applications into approved or rejected categories based on the provided dataset. The project includes data preprocessing, model training, hyperparameter tuning, and evaluation to achieve the best performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)


## Project Overview

Credit card approval is a critical task for banks, where numerous applications are received daily. Analyzing these applications manually is time-consuming, prone to errors, and expensive. This project leverages machine learning to automate the approval process, ensuring accuracy and efficiency.

The project involves:
- Preprocessing the dataset to handle missing values and encode categorical variables.
- Standardizing features for better model performance.
- Training multiple models including Logistic Regression, Random Forest, and Support Vector Machine (SVM).
- Performing Grid Search for hyperparameter tuning to select the best model.
- Evaluating the final model on a test set, aiming for an accuracy score of at least 0.75.

## Dataset

The dataset consists of various features related to credit card applications, including categorical and numerical data. Each row represents a credit card application, and the target variable indicates whether the application was approved or rejected. Here is an example of the data structure:

| Column 0 | Column 1 | Column 2 | Column 3 | Column 4 | Column 5 | Column 6 | Column 7 | Column 8 | Column 9 | Column 10 | Column 11 | Column 12 | Column 13 |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|-----------|-----------|-----------|
| b        | 30.83    | 0        | u        | g        | w        | v        | 1.25     | t        | t        | 1         | g         | 0         | +         |
| a        | 58.67    | 4.46     | u        | g        | q        | h        | 3.04     | t        | t        | 6         | g         | 560       | +         |
| a        | 24.5     | 0.5      | u        | g        | q        | h        | 1.5      | t        | f        | 0         | g         | 824       | +         |

- **Features:** The columns include both categorical (e.g., `u`, `g`, `b`) and numerical (e.g., `30.83`, `4.46`) data points.
- **Target:** The last column (Column 13) is the target variable, indicating approval (`+`) or rejection (`-`).

## Data Preprocessing

Data preprocessing steps include:
- **Handling Missing Values:** If there are any missing values, they are replaced with the mean (for numerical features) or the most frequent value (for categorical features).
- **Encoding Categorical Variables:** Categorical features are converted into numerical values using `LabelEncoder`.
- **Feature Scaling:** The numerical features are standardized to have zero mean and unit variance using `StandardScaler`.


## Model Training

Three different models were trained:
- **Logistic Regression:** A simple linear model used as a baseline.
- **Random Forest:** An ensemble model to handle non-linear relationships.
- **Support Vector Machine (SVM):** A model that works well with higher-dimensional data.

## Hyperparameter Tuning

GridSearchCV was used to find the best hyperparameters for each model:
- **Logistic Regression:** Tuning the regularization parameter `C`.
- **Random Forest:** Tuning the number of estimators and maximum depth.
- **SVM:** Tuning the regularization parameter `C` and kernel type.

## Evaluation

After training and tuning multiple models, the best model selected was the **RandomForestClassifier** with 50 estimators.

- **Best Cross-Validated Accuracy Score:** 0.871
- **Test Set Accuracy Score:** 0.848

The model performed well on the test set, achieving an accuracy score of 0.848, which exceeds the target accuracy of 0.75. This indicates that the model is effective in predicting credit card approvals based on the features provided.

