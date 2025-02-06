# Customer Churn Prediction

## Overview
This project aims to predict customer churn using a dataset containing various customer attributes. The analysis includes data preprocessing, exploratory data analysis (EDA), feature engineering, and model training using Logistic Regression with hyperparameter tuning.

## Installation
To run this project, ensure you have the following dependencies installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## Dataset
The dataset consists of 7,043 customer records with 21 features, including categorical and numerical attributes.

## Data Preprocessing
- **Handling Missing Values:**
  - Converted `TotalCharges` to numeric and filled missing values with the mode.
- **Feature Encoding:**
  - Used `LabelEncoder` for binary categorical features.
  - Applied `OrdinalEncoder` to ordered categorical features.
  - One-hot encoded multi-category features.
- **Balancing the Dataset:**
  - Used **SMOTE** to handle class imbalance.
- **Feature Scaling:**
  - Applied `StandardScaler` to `tenure`, `MonthlyCharges`, and `TotalCharges`.

## Exploratory Data Analysis (EDA)
Several visualizations were created to understand churn distribution:
- **Churn distribution**
- **Gender vs. Churn**
- **Tenure and Monthly Charges distribution by Churn**
- **Payment Method vs. Churn**

## Model Training & Tuning
- Used **Logistic Regression** with `GridSearchCV` to tune hyperparameters:
  ```python
  param_grid = {
      'C': [0.01, 0.1, 1, 10, 50, 100, 200],
      'penalty': ['l2', 'elasticnet'],
      'class_weight': ['balanced', None],
      'max_iter': [200, 500, 1000, 2000],
      'solver': ['saga', 'liblinear']
  }
  ```

## Model Evaluation
Best hyperparameters:
```
C: 200
class_weight: None
max_iter: 200
penalty: l2
solver: liblinear
```

### Classification Report (Test Set)
```
              precision    recall  f1-score   support

           0       0.87      0.84      0.86      1036
           1       0.60      0.65      0.62       373

    accuracy                           0.79      1409
   macro avg       0.73      0.75      0.74      1409
weighted avg       0.80      0.79      0.79      1409
```

## Conclusion
- Logistic Regression achieved **79% accuracy** on the test set.
- Feature engineering and SMOTE significantly improved class balance.
- Further improvement can be achieved using ensemble models or deep learning approaches.

## Further Improvements
- Experiment with **Random Forest, LightGBM, or Neural Networks**.
- Perform **feature selection** to enhance model interpretability.
- Deploy the model using **Flask or FastAPI** for real-world application.

