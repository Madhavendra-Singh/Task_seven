# Task 7: Support Vector Machines (SVM) - Breast Cancer Classification

## Objective
The goal of this task is to implement and evaluate Support Vector Machine (SVM) models for binary classification using a breast cancer dataset. Both linear and non-linear (RBF kernel) SVMs are explored along with hyperparameter tuning and model evaluation techniques.

---

## Dataset
- **Source:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Total Samples:** 569
- **Features:** 30 numeric features describing cell nuclei in breast mass
- **Target Classes:**
  - **0** → Benign (B)
  - **1** → Malignant (M)

---

## Files
- `Task7_Intern.ipynb`: Main notebook with full code and results.
- `breast-cancer.csv`: CSV dataset file used in the notebook.

---

## Steps Performed

### 1. Load and Prepare Data
- Read the dataset using `pandas`
- Dropped unnecessary columns (`id`, `Unnamed: 32`)
- Encoded `diagnosis` column (M → 1, B → 0)
- Standardized the features using `StandardScaler`
- Split the dataset into 80% training and 20% testing using `train_test_split`

### 2. Train SVM Models

#### Linear SVM
- Trained with `kernel='linear'`, `C=1`
- Evaluation:

```
Accuracy: 0.9649

Confusion Matrix:
[[72  0]
 [ 4 38]]

Classification Report:
              precision    recall  f1-score   support
         0       0.95      1.00      0.97        72
         1       1.00      0.90      0.95        42
  accuracy                           0.96       114
 macro avg       0.97      0.95      0.96       114
weighted avg     0.97      0.96      0.96       114
```

#### RBF SVM with Hyperparameter Tuning
- Grid search performed using `GridSearchCV` with parameters:
  - `C`: [0.1, 1, 10, 100]
  - `gamma`: ['scale', 0.001, 0.01, 0.1, 1]
  - `kernel`: ['rbf']
- Best parameters found:
```
Best Parameters: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
Best Cross-Validation Score: 97.58%
```

- Test set evaluation:

```
Accuracy: 0.9561

Confusion Matrix:
[[70  2]
 [ 3 39]]

Classification Report:
              precision    recall  f1-score   support
         0       0.96      0.97      0.97        72
         1       0.95      0.93      0.94        42
  accuracy                           0.96       114
 macro avg       0.96      0.95      0.95       114
weighted avg     0.96      0.96      0.96       114
```

---

### 3. Cross-Validation Performance
- Evaluated the final tuned model using 5-fold cross-validation:

```
Cross-Validation Accuracy Scores:
[0.9649, 0.9561, 0.9649, 0.9825, 0.9734]
Mean CV Accuracy: 96.84%
```

---

### 4. Visualization
- Visualized the SVM decision boundary using only 2 features: `radius_mean` and `texture_mean`
- Used `matplotlib` and `seaborn` to plot a 2D decision region

---

## How to Run
1. Clone or download the repository
2. Make sure the required libraries are installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
3. Run the notebook `Task7_Intern.ipynb` in Jupyter Notebook or any IDE that supports `.ipynb` files

---

## Conclusion
- SVM with both linear and RBF kernels performed well on this binary classification task.
- Hyperparameter tuning further improved the model accuracy.
- The final model achieved a **test accuracy of ~95.6%** and **cross-validation accuracy of ~96.8%**.
