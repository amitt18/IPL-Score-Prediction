from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "model"
PLOTS = ROOT / "eda_plots"

print("Loading IPL data...")
deliveries = pd.read_csv(DATA / "deliveries.csv")
matches = pd.read_csv(DATA / "matches.csv")
df = deliveries.merge(matches[["ID", "Venue", "Season"]], on="ID", how="left")
df["Venue"] = df["Venue"].fillna("Unknown").astype(str)
df = df.sort_values(["ID", "innings", "overs", "ballnumber"]).reset_index(drop=True)

# Features describe the delivery whose outcome is being predicted.
# These features are available in the app before that ball is bowled.
df["score_before_ball"] = (
    df.groupby(["ID", "innings"])["total_run"].cumsum() - df["total_run"]
)
df["wickets_before_ball"] = (
    df.groupby(["ID", "innings"])["isWicketDelivery"].cumsum() - df["isWicketDelivery"]
)
df["runs_last_6_balls"] = (
    df.groupby(["ID", "innings"])["total_run"]
      .transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum())
      .fillna(0)
)
df["wickets_last_6_balls"] = (
    df.groupby(["ID", "innings"])["isWicketDelivery"]
      .transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum())
      .fillna(0)
)
df["balls_elapsed"] = df["overs"] * 6 + df["ballnumber"] - 1
df["run_rate"] = np.where(
    df["balls_elapsed"] > 0,
    df["score_before_ball"] * 6 / df["balls_elapsed"],
    0.0
)

feature_cols = [
    "innings", "overs", "ballnumber", "balls_elapsed",
    "score_before_ball", "wickets_before_ball",
    "runs_last_6_balls", "wickets_last_6_balls", "run_rate",
    "batter", "bowler", "BattingTeam", "Venue"
]
categorical = ["batter", "bowler", "BattingTeam", "Venue"]
numeric = [c for c in feature_cols if c not in categorical]

X = df[feature_cols].copy()
y = df["total_run"].astype(int)

# Split by match ID so balls from one match don't appear in both train and test.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=df["ID"]))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

preprocess = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numeric)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(
        n_estimators=160, min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    ))
])

print(f"Rows: {len(df):,} | Train: {len(train_idx):,} | Test: {len(test_idx):,}")
print("Training model. This can take several minutes...")
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"\nMatch-held-out accuracy: {accuracy_score(y_test, pred):.4f}")
print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))
print("\nRun counts in data:")
print(y.value_counts().sort_index())

OUT.mkdir(exist_ok=True)
joblib.dump(model, OUT / "next_ball_model.joblib")
joblib.dump(feature_cols, OUT / "feature_cols.joblib")
joblib.dump(categorical, OUT / "categorical_cols.joblib")
PLOTS.mkdir(exist_ok=True)
plt.figure(figsize=(9, 4))
y.value_counts().sort_index().plot(kind="bar")
plt.title("Historical runs conceded/scored per delivery (total_run)")
plt.xlabel("Runs on delivery")
plt.ylabel("Number of deliveries")
plt.tight_layout()
plt.savefig(PLOTS / "run_distribution.png")
plt.close()
print(f"\nSaved model to: {OUT / 'next_ball_model.joblib'}")
print("Training complete. Now run: python -m streamlit run app.py")
