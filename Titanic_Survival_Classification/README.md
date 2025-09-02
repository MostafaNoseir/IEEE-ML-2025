# Titanic Survival Prediction

## Project Objective

This project aims to predict whether a passenger survived the Titanic disaster using structured historical data. The goal is to build and evaluate machine learning models, specifically focusing on tree-based methods (Decision Tree, Random Forest, and XGBoost), and compare their performance after data preprocessing, feature engineering, and hyperparameter tuning.

## Dataset

*   **Source:** Kaggle Titanic Dataset
*   **Features:** Includes passenger information such as PassengerId, Pclass, Name, Sex, Age, SibSp (siblings/spouses aboard), Parch (parents/children aboard), Ticket, Fare, Cabin, and Embarked.
*   **Target:** `Survived` (0 = No, 1 = Yes)

## Methodology

1.  **Data Loading and Overview:** Loaded the dataset and performed initial exploration to understand its structure, summary statistics, and identify missing values and duplicates.
    *   Initial data shape: (891, 12)
    *   Missing values identified in: `Age` (177), `Cabin` (687), `Embarked` (2)
    *   Duplicate rows found: 111
2.  **Data Preprocessing:**
    *   Dropped non-essential columns (`PassengerId`, `Name`, `Ticket`, `Cabin`).
    *   Handled missing values in `Embarked` by filling with the mode ('S').
    *   Handled missing values in `Age` using KNN Imputation.
    *   Handled outliers in numerical features (`Age`, `SibSp`, `Parch`, `Fare`) by capping them using the IQR method.
    *   Removed duplicate rows.
3.  **Feature Engineering:** Categorical features (`Sex`, `Embarked`) were prepared for encoding.
4.  **Exploratory Data Analysis (EDA):** Visualized the distribution of survival and its relationship with key categorical and numerical features.
    *   Data is unbalanced, with more non-survivors (549) than survivors (342).
    *   Observations on survival by Sex, Pclass, and Embarked were noted.
5.  **Data Splitting and Scaling:** Split the data into training, validation, and test sets (80/10/10 split) and scaled the numerical features using StandardScaler. Addressed class imbalance using SMOTE on the training data.
6.  **Modeling:** Trained several classification models:
    *   Logistic Regression
    *   Decision Tree Classifier
    *   Random Forest Classifier (Bagging)
    *   XGBoost Classifier (Boosting)
7.  **Hyperparameter Tuning:** Performed manual, step-by-step hyperparameter tuning for the XGBoost and Random Forest models using the validation set to improve performance and reduce overfitting. Early stopping was incorporated for XGBoost tuning.
    *   **XGBoost Tuning:** Tuned `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`. Best validation ROC-AUC of 0.8928 achieved with `max_depth=5`, `learning_rate=0.01`, `subsample=0.7`, `colsample_bytree=0.9`, and best iteration 527.
    *   **Random Forest Tuning:** Tuned `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`. Best validation ROC-AUC of 0.8968 achieved with `n_estimators=300`, `max_depth=7`, `min_samples_split=2`, and `min_samples_leaf=2`.
8.  **Model Evaluation:** Evaluated the performance of the best-tuned XGBoost and Random Forest models on the unseen test set using metrics such as Classification Report (Precision, Recall, F1-score) and ROC-AUC.

## Results

The final evaluation on the test set provided the following performance metrics:

**Best Tuned XGBoost Model Evaluation on Test Set:**
