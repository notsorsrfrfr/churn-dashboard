import os, pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Load data
df = pd.read_csv("C:/Users/SANIDHYA RAJ/Desktop/ML Projects/Customer Churn Prediction/data/Churn_Modelling.csv")
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

# Feature engineering
df["BalanceSalaryRatio"]  = df["Balance"] / (df["EstimatedSalary"] + 1)
df["TenureByAge"]         = df["Tenure"] / df["Age"]
df["CreditScoreByAge"]    = df["CreditScore"] / df["Age"]
df["ZeroBalance"]         = (df["Balance"] == 0).astype(int)
df["ProductsPerTenure"]   = df["NumOfProducts"] / (df["Tenure"] + 1)

CATEGORICAL = ["Geography", "Gender"]
NUMERICAL   = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "BalanceSalaryRatio", "TenureByAge", "CreditScoreByAge",
    "ZeroBalance", "ProductsPerTenure"
]

X = df[CATEGORICAL + NUMERICAL]
y = df["Exited"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERICAL),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL),
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train models, pick best by AUC
models = {
    "Random Forest":      RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting":  GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

best_auc, best_model, best_name = 0, None, ""
for name, clf in models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    auc  = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
    print(f"{name}: CV AUC = {auc:.4f}")
    if auc > best_auc:
        best_auc, best_model, best_name = auc, pipe, name

print(f"\nBest: {best_name} (AUC={best_auc:.4f})")
best_model.fit(X_train, y_train)

y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

# Save model + metadata
os.makedirs("models", exist_ok=True)
pickle.dump(best_model, open("models/churn_model.pkl", "wb"))
pickle.dump({"categorical": CATEGORICAL, "numerical": NUMERICAL},
            open("models/metadata.pkl", "wb"))
print("Model saved to models/")
