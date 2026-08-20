# diabetes_prediction.py — End-to-end beginner ML project

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# 2. Quick exploration
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values (as zeros in medical features):")
# In this dataset, zeros in certain columns mean missing
medical_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print((df[medical_cols] == 0).sum())

# 3. Preprocess: replace zeros with NaN, then impute with median
df[medical_cols] = df[medical_cols].replace(0, np.nan)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 4. Split data (stratify to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Build a pipeline for each model
def build_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

models = {
    'Logistic Regression': build_pipeline(LogisticRegression(max_iter=1000)),
    'Random Forest': build_pipeline(RandomForestClassifier(random_state=42))
}

# 6. Train and evaluate each model
for name, pipe in models.items():
    print(f"\n=== {name} ===")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Cross-validation on the best model (Random Forest)
print("\n=== Cross-validation (Random Forest) ===")
rf_pipe = models['Random Forest']
cv_scores = cross_val_score(rf_pipe, X_train, y_train, cv=5, scoring='accuracy')
print("CV Accuracy: {:.2f} (+/- {:.2f})".format(cv_scores.mean(), cv_scores.std()))

# 8. Hyperparameter tuning (simple grid search)
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 5, 10]
}
grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print("\nBest parameters:", grid.best_params_)
print("Best CV accuracy: {:.2f}".format(grid.best_score_))

# 9. Final evaluation on test set with tuned model
best_model = grid.best_estimator_
y_pred_final = best_model.predict(X_test)
print("\n=== Final Test Performance (Tuned Random Forest) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_final))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_final))
print("Classification Report:\n", classification_report(y_test, y_pred_final))