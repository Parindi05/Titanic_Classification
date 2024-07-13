# Titanic Classification Project

## Introduction

This project aims to predict the survival of passengers on the Titanic based on various features such as socio-economic status, age, and gender. The analysis was performed using Logistic Regression and Random Forest classifiers.

## Dataset

The dataset used for this project is the Titanic dataset, which contains information about passengers on the Titanic. The dataset includes the following features:

- PassengerId
- Survived (target variable)
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

## Data Preprocessing

The following preprocessing steps were applied to the dataset:

1. **Handling Missing Values:**
   - Filled missing values in the 'Age' column with the median.
   - Created a new feature 'Cabin_ind' to indicate whether a cabin was assigned and dropped the original 'Cabin' column.
   - Filled missing values in the 'Embarked' column with the most frequent value.

2. **Feature Engineering:**
   - Converted categorical variables 'Sex' and 'Embarked' to numerical values.
   - Created dummy variables for 'Pclass'.
   - Dropped irrelevant columns: 'Name', 'Ticket', and 'PassengerId'.

3. **Feature Scaling:**
   - Standardized the feature set using `StandardScaler`.

## Model Training

Two machine learning models were trained and evaluated:

1. **Logistic Regression:**
   - Best Hyperparameters: `{'C': 0.01, 'max_iter': 100, 'solver': 'saga'}`

2. **Random Forest:**
   - Best Hyperparameters: `{'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}`

## Model Evaluation

The models were evaluated using Accuracy, Precision, Recall, and F1-Score metrics.

### Logistic Regression

- **Accuracy:** 0.804
- **Precision:** 0.783
- **Recall:** 0.730
- **F1-Score:** 0.755

### Random Forest

- **Accuracy:** 0.799
- **Precision:** 0.764
- **Recall:** 0.743
- **F1-Score:** 0.753

### Best Hyperparameters and Cross-Validation Scores

- **Logistic Regression Cross-Validation Score:** 0.808
- **Random Forest Cross-Validation Score:** 0.823

### Test Set Evaluation for Random Forest

- **Accuracy:** 0.838
- **Precision:** 0.846
- **Recall:** 0.743
- **F1-Score:** 0.791

## Feature Importance

The feature importance for the Random Forest model was analyzed to understand the contribution of each feature to the prediction. The following plot shows the importance of each feature:

![feature_importance_plot](https://github.com/user-attachments/assets/2c5d9ec9-1ec6-4ea5-aa7c-d6cd519ae255)


## Conclusion

The Random Forest model performed slightly better than the Logistic Regression model in terms of accuracy and F1-score on the test set. The most important features contributing to the prediction were analyzed using feature importance from the Random Forest model.

## Files

- `Titanic Classification.ipynb`: Jupyter notebook containing the entire code for data preprocessing, model training, evaluation, and feature importance analysis.
- `train.csv`: Training dataset.
- `test.csv`: Test dataset.
- `README.md`: This README file.

## How to Run

1. Clone the repository.
2. Ensure you have the necessary dependencies installed.
3. Run the `titanic_classification.ipynb` notebook to reproduce the results.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Author
Parindi Soysa
