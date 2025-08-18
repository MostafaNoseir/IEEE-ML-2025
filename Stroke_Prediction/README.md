# 🏥 Stroke Prediction Analysis

## Dataset Overview

The dataset contains **multiple records** of individuals with the following features:

| Column Name        | Description |
|--------------------|-------------|
| `id`               | Unique identifier for each patient |
| `gender`           | Gender of the patient: `Male` or `Female` |
| `age`              | Age of the patient |
| `hypertension`     | Binary: `1` if the patient has hypertension, else `0` |
| `heart_disease`    | Binary: `1` if the patient has a heart disease, else `0` |
| `ever_married`     | Whether the patient has ever been married: `Yes` or `No` |
| `work_type`        | Type of employment: `Private`, `Self-employed`, `Govt_job`, `children`, or `Never_worked` |
| `Residence_type`   | Type of residence: `Urban` or `Rural` |
| `avg_glucose_level`| Average glucose level in blood |
| `bmi`              | Body Mass Index |
| `smoking_status`   | Smoking status: `formerly smoked`, `never smoked`, `smokes`, or `Unknown` |
| `stroke`           | Binary: `1` if the patient had a stroke, else `0`

### Target Variable

- `stroke`: The main label to predict. The goal is to identify patterns that lead to a stroke based on the other features.

---

## 🔍 Workflow Summary

### 1. Data Exploration
- Loaded and inspected dataset with 5110 entries and 12 features
- Performed summary statistics
- check all unique values at categorical columns
- Identified class imbalance (only 4.87% stroke cases)

### 2. Data Preprocessing
#### Data Cleaning
- Dropped irrelevant `id` column
- Removed single invalid record with `gender = 'Other'`
- Converted `age` from float to integer
- Replaced `smoking_status = 'Unknown'` with mode ('never smoked')
- Handled missing values in `bmi` (201 nulls) using KNN Imputation
- Handled Outliers in `bmi` by remove them and in `avg_glucose_level` by clip outliers with upper

### 3. Feature Engineering
- Split data into training (80%) and test sets (20%) with stratification.
- Perform label encoding for binary categorical features ('gender','ever_married','Residence_type')
- Performed one-hot encoding for multi categorical features ('work type','smoking status').
- Standardized numerical features using StandardScaler.
- Applied Randomforest feature importence, VarianceThreshold and L1-based feature selection to reduce irrelevant features.
- Apply SMOTE for over samping minority class to solve imbalanced data.

### 4. Modeling
- Train **Hard Margin** SVM (LinearSVC with very large C)
- Train **Soft Margin** SVM (LinearSVC with hyperparameter tuning for C using GridSearchCV -> `C`: [0.01, 0.1, 1, 10, 100])
- Train **RBF Kernel** SVM(SVC with gamma tuning via GridSearchCV -> `gamma`: [0.01, 0.1, 1, 'scale'])
- Train **Polynomial Kernel** SVM (SVC with degree tuning via GridSearchCV -> `degree`: [2, 3, 4])
- Best Parameters:
    - `C`: 1 for Soft Margin
    - `gamma`: 1 for RBF Kernel
    - `degree`: 3 for Polynomial Kernel
- Calculated metrics like F1-score for the test set
- Applying PCA (Principal Component Analysis) before plotting decision boundaries.

### 5. Results Summary

| Model                | F1-score |
|----------------------|----------|
| F1-score             | 0.272    |
| Soft Margin          | 0.272    |
| RBF Kernel           | 0.201    |
| RPolynomialBF Kernel | 0.237    |

## 🧠 Key Insights
- All Evaluations of all models in this data is very **bad** because **imbalanced** data
