# 💎 Diamond Price Prediction

## 📝 Overview
This project presents a comprehensive regression analysis to predict **diamond prices** using various linear models including **Linear Regression**, **Ridge**, **Lasso**, and **ElasticNet**. The notebook follows a structured machine learning pipeline from exploratory data analysis to model evaluation.

---

## 📁 Dataset
- **Source**: Diamonds dataset (53,940 records)
- **Features**:
  - `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`
- **Target**: `price` (in US dollars)

---

## 🔍 Workflow Summary

### 1. Data Exploration
- Loaded the dataset and viewed basic statistics (`shape`, `info()`, `describe()`)
- Identified categorical (`cut`, `color`, `clarity`) and numerical columns

### 2. Data Cleaning & Preprocessing
- ✅ Dropped unnecessary column: `Unnamed: 0`
- ✅ Removed duplicates
- ✅ Confirmed no missing values
- ✅ Visualized outliers with boxplots
- ✅ Removed outliers using IQR method
- ✅ Checked correlation with heatmap
- ✅ Dropped `depth` due to low correlation with target

### 3. Feature Engineering
- Applied **one-hot encoding** to categorical features (`cut`, `color`, `clarity`)
- Split data into **features (X)** and **target (y)**

### 4. Data Splitting & Scaling
- Split data into **training (80%)** and **testing (20%)** sets
- Scaled numeric features (`carat`, `table`, `x`, `y`, `z`) using `MinMaxScaler`

---

## 🤖 Model Training & Evaluation

### 🔹 Linear Regression
- Trained on scaled data
- Achieved:
  - **R² Score**: 0.93
  - **RMSE**: ~705
  - **Adjusted R²**: 0.93
- Visualized actual vs predicted prices with a scatter plot

### 🔹 Ridge Regression
- Regularization parameter: `alpha=0.001`
- Achieved similar performance as basic linear regression

### 🔹 Lasso Regression
- `alpha=0.001`, `max_iter=10000`
- Performance nearly identical to Ridge

### 🔹 ElasticNet Regression
- `alpha=0.001`, `l1_ratio=0.8`
- Slightly lower performance:
  - **R² Score**: 0.92
  - **RMSE**: ~716

---

## 📊 Results Comparison

| Model            | R² Score | RMSE |
|------------------|----------|------|
| Linear Regression| 0.93     | 705  |
| Ridge            | 0.93     | 705  |
| Lasso            | 0.93     | 705  |
| ElasticNet       | 0.92     | 716  |

---

## ✅ Conclusion
- The **basic Linear Regression model** outperformed regularized models on this dataset.
- **Carat** had the highest coefficient, indicating it's the most influential feature.
- **Regularization** did not significantly improve model performance, possibly due to proper preprocessing and low multicollinearity.

---

## 🧠 Key Takeaways
- Removing outliers and proper encoding significantly improves model accuracy.
- Always validate if regularization is necessary — sometimes simpler models perform best.
