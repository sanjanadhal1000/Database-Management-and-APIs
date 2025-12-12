from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import joblib                        # Saves and Loads trained ML models
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names) # data.data = input features (numbers)
df['target'] = data.target                               # data.target = 0/1 labels

X = df.drop('target', axis=1)                            # All columns except target are features
y = df['target']

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize Random Forest
rf = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42
)

# Train Model
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:",round(accuracy,2))
print("\nF1-Score:",round(f1,2))
print("\nClassification Report:",classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest (Breast Cancer)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png")

# Save Model
joblib.dump(rf, "random_forest_model.pkl")
print("\nModel saved as random_forest_model.pkl")