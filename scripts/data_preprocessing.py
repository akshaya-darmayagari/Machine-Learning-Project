import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
STAGED_PATH = "data/staged/processed_data.csv"
SCALER_PATH = "models/scaler.pkl"

os.makedirs("data/staged", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv(RAW_PATH)

# Drop unnecessary columns
df.drop(["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], axis=1, inplace=True)

# Encode target
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Encode categorical columns
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Separate features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Scale features only
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled["Attrition"] = y

X_scaled.to_csv(STAGED_PATH, index=False)
joblib.dump(scaler, SCALER_PATH)

print("✅ Preprocessing completed.")