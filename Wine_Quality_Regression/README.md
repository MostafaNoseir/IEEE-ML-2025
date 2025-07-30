📊 Logistic Regression on Food Items Dataset
---
📝 Overview:
This notebook presents an end-to-end machine learning pipeline for multiclass classification using Logistic Regression. It compares Multinomial Logistic Regression with the One-vs-Rest strategy, explores regularization, and performs hyperparameter tuning to improve model performance.

📁 Dataset:
- Source: Food Items Dataset
- Features: Includes nutritional information like calories, protein, fat, etc.
- Target: Multiclass labels indicating food categories.
---

1. Data Exploration
- Initial loading and inspection of the dataset.
- Summary statistics, null value checking.
- Visualizations using Seaborn and Matplotlib for feature distribution.
---

3. Data Preprocessing
Ordinal encoding for Target.
Standardization using StandardScaler.
Train-test split (typically 80/20).
Optional: Handling data imbalance using techniques like SMOTE.
---

4. Model Training
Trained Multinomial Logistic Regression using multi_class='multinomial'.
Trained One-vs-Rest Logistic Regression using multi_class='ovr'.
---

5. Evaluation Metrics
Used:
Accuracy
Precision, Recall, F1-score (per class)
Macro & weighted averages
Confusion Matrix
ROC-AUC curve (one-vs-rest setting using One-vs-Rest wrapper and predict_proba)
---

6. Visualization
Confusion matrix plot for each model.
ROC-AUC curves plotted for each class in multiclass setting.
---

6. Hyperparameter Tuning
Used GridSearchCV with 5-fold cross-validation.
Parameters tuned:
C (Inverse of regularization strength)
penalty (l1, l2)
solver (saga, lbfgs)
---

Best model:
{'C': 10, 'penalty': 'l1', 'solver': 'saga'}

Results Summary:
Model	               | Accuracy |
One-vs-Rest Logistic |	 0.81   |
Multinomial Logistic |   0.82   |
Tuned Multinomial    |	 0.83   |

Multinomial Logistic Regression showed slightly better generalization.
Regularization (L1 + saga) helped improve generalization, especially for minority class (class 2).
ROC-AUC curves confirmed model confidence in predictions across classes.
-------------------------

Key Insights:
- Regularization improves generalization and handles feature sparsity effectively.
- One-vs-Rest performs competitively, but Multinomial handles class probabilities more smoothly.
- Hyperparameter tuning significantly boosts the performance of logistic regression.
