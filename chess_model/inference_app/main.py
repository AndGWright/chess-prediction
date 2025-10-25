from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pickle
import pandas as pd
import re
from contextlib import asynccontextmanager

MODEL_PATH = Path("models") / "model.pkl"

_model = None

# Lifespan handler 
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    print("Model loaded successfully.")
    yield

app = FastAPI(
    title="Chess Win Probability",
    version="0.1",
    lifespan=lifespan,
)

class MatchInput(BaseModel):
    WhiteElo: int = Field(..., description="White player's Elo rating")
    BlackElo: int = Field(..., description="Black player's Elo rating")
    TimeControl: str = Field(..., description="Time control string like '5+0' or '3+2'")
    Opening: str = Field(..., description="Opening name")

class PredictionOut(BaseModel):
    white_win_probability: float
    black_win_probability: float

# Utilities
def parse_timecontrol(tc: str):
    """Return (game_length, increment) given a timecontrol like '5+0'."""
    try:
        base, inc = tc.split('+', 1)
        return int(base), int(inc)
    except Exception:
        raise ValueError(f"Invalid TimeControl format: {tc!r}. Expected like '5+0' or '3+2'.")

def group_opening(string: str) -> str:
    """Remove variations of the same opening family."""
    if string is None:
        return ""
    s = string.split(':', 1)[0].split('|', 1)[0].strip()
    pattern = r'^(.*?(?:Defense|Attack|Game|Opening|Gambit|Countergambit)\b)'
    match = re.match(pattern, s)
    if match:
        return match.group(1)
    return s

# Health check
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

# Prediction endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(match: MatchInput):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        game_length, increment = parse_timecontrol(match.TimeControl)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rating_diff = match.WhiteElo - match.BlackElo
    opening_group = group_opening(match.Opening).replace("'", "")


    X = pd.DataFrame([{
        "rating_diff": rating_diff,
        "WhiteElo": match.WhiteElo,
        "opening_group": opening_group,
        "increment": increment,
        "game_length": game_length
    }])

    try:
        probs = _model.predict_proba(X)[0]
        white_prob = float(probs[1])
        black_prob = float(probs[0])
   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictionOut(
        white_win_probability=round(white_prob, 6),
        black_win_probability=round(black_prob, 6),
    )
