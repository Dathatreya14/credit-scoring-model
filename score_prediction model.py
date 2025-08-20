import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

# 1) Load
app = pd.read_csv("/content/application_record.csv")
if "ID" in app.columns:
    app = app.drop_duplicates(subset=["ID"], keep="last")
else:
    raise ValueError("application_record.csv must contain an 'ID' column.")

# 2) Features
numeric_features = ["CNT_CHILDREN", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"]
categorical_features = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

X = app[numeric_features + categorical_features].copy()

# 3) Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 4) Isolation Forest pipeline
iso = Pipeline(steps=[
    ("prep", preprocessor),
    ("iforest", IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    ))
])

# 5) Fit on all data
iso.fit(X)

# 6) Scores: higher = more anomalous (riskier)
raw_score = -iso.named_steps["iforest"].decision_function(iso.named_steps["prep"].transform(X))
risk_score = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-12)

# Attach back
out = app[["ID"]].copy()
out["risk_score"] = risk_score

print(out.head())

# ------------------------------
# Interactive user input
# ------------------------------
print("\nEnter customer details for credit risk scoring:")

user_data = {}
user_data["CODE_GENDER"] = input("Gender (M/F): ").strip().upper()
user_data["FLAG_OWN_CAR"] = input("Own Car (Y/N): ").strip().upper()
user_data["CNT_CHILDREN"] = int(input("Number of Children: "))
user_data["AMT_INCOME_TOTAL"] = float(input("Total Income: "))
user_data["NAME_INCOME_TYPE"] = input("Income Type (e.g., Working, Pensioner, Student): ").strip()
user_data["NAME_FAMILY_STATUS"] = input("Family Status (e.g., Married, Single): ").strip()
user_data["NAME_HOUSING_TYPE"] = input("Housing Type (e.g., House / apartment, Rented): ").strip()
user_data["OCCUPATION_TYPE"] = input("Occupation Type (e.g., Laborers, Managers, NaN if unknown): ").strip()
user_data["CNT_FAM_MEMBERS"] = int(input("Number of Family Members: "))

new_customer = pd.DataFrame([user_data])

# Predict risk score
new_risk = -iso.named_steps["iforest"].decision_function(
    iso.named_steps["prep"].transform(new_customer)
)
new_risk = (new_risk - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-12)

print("\nPredicted Unsupervised Credit Risk Score (0 = low risk, 1 = high risk):", float(new_risk[0]))
