import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocess import create_features, preprocess_openings
from src.pipeline import build_pipeline
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "chess_model" / "data"
MODEL_DIR = ROOT / "models"

df = pd.read_parquet(DATA_DIR / "train-00000-of-00064.parquet")
df = create_features(df)
df = preprocess_openings(df)

numeric_features = ['rating_diff', 'WhiteElo']
categorical_features = ['opening_group', 'increment', 'game_length']
X = df[numeric_features + categorical_features]
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

pipeline = build_pipeline(numeric_features, categorical_features)
pipeline.fit(X_train, y_train)

print("Train Accuracy:", accuracy_score(y_train, pipeline.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, pipeline.predict(X_test)))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
with open(MODEL_DIR / "model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
