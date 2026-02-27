import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

DATA_PATH = "data/staged/processed_data.csv"
MODEL_DIR = "models/"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=300),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    print(f"{name} ROC-AUC: {roc:.4f}")
    results.append((name, roc))
    joblib.dump(model, MODEL_DIR + f"{name}.pkl")

results.sort(key=lambda x: x[1], reverse=True)

print("\n📊 Model Ranking:")
for r in results:
    print(r[0], ":", r[1])

print("✅ All models saved.")