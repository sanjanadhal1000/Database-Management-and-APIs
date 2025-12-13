# Gradient Boosting & XGBoost

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target

X = df.drop('target', axis=1)
y = df['target']

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train RF model
rf.fit(X_train, y_train)

# Predict RF model
rf_pred = rf.predict(X_test)

# RF Metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

# XGBoost Model
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

xgb_model_pred = xgb_model.predict(X_test)

xgb_model_accuracy = accuracy_score(y_test, xgb_model_pred)
xgb_model_f1 = f1_score(y_test, xgb_model_pred)

# Confusion Matrix XGBoost Model
xgb_model_cm = confusion_matrix(y_test, xgb_model_pred)
plt.figure(figsize=(8,6))
sns.heatmap(xgb_model_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - XGBoost Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_xgb.png')

# Feature Importance XGBoost Model

# RF vs XGBoost Metrics
print('\n---------Random Forest---------\n')
print('\nAccuracy:', round(rf_accuracy,2))
print('\nF1-Score:', round(rf_f1,2))
print('\nClassification Report:', classification_report(y_test,rf_pred))

print('\n---------------------------------------\n')

print('\n--------------XGBoost----------------\n')
print('\nAccuracy:', round(xgb_model_accuracy,2))
print('\nF1-Score:', round(xgb_model_f1,2))
print('\nClassification Report:', classification_report(y_test,xgb_model_pred))

# ------------------------------------
# Feature Importance – Random Forest
# ------------------------------------
rf_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=rf_importance[:10], y=rf_importance.index[:10])
plt.title("Top 10 Feature Importances – Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance_random_forest.png")

# -----------------------------
# Feature Importance – XGBoost
# -----------------------------
xgb_importance = pd.Series(
    xgb_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_importance[:10], y=xgb_importance.index[:10])
plt.title("Top 10 Feature Importances – XGBoost")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance_xgboost.png")

# Save Comparison Results
with open('rf_vs_xgb_metrics.txt','w') as f:
    f.write("Random Forest\n")
    f.write(f"Accuracy: {rf_accuracy:.2f}\n")
    f.write(f"F1 Score: {rf_f1:.2f}\n\n")

    f.write("XGBoost\n")
    f.write(f"Accuracy: {xgb_model_accuracy:.2f}\n")
    f.write(f"F1 Score: {xgb_model_f1:.2f}\n")

print("\nComparison Results saved to 'rf_vs_xgb_metrics.txt'")