import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", "mental_health_pipeline.pkl"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH
if not MODEL_PATH.exists() and MODEL_PATH.name == "mental_health_pipeline.pkl":
    MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"

model = joblib.load(MODEL_PATH)
TOP_COUNTRIES = {
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France",
    "Other",
}

app = FastAPI(
    title="Y&S Student Mental Health Score Demo",
    description="A demo API for a trained student mental-health score regression model.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    Age: int = Field(ge=10, le=30)
    Study_Hours: float = Field(ge=0, le=12)
    Avg_Daily_Usage_Hours: float = Field(ge=0, le=10)
    Daily_Unlocks: int = Field(ge=0, le=500)
    Physical_Activity_Hours: float = Field(ge=0, le=6)
    Sleep_Hours_Per_Night: float = Field(ge=1, le=12)
    Stress_Level: Literal["Low", "Medium", "High", "Very High"]
    Gender: Literal["Female", "Male"]
    Academic_Level: Literal["High School", "Undergraduate", "Graduate"]
    Most_Used_Platform: Literal[
        "Facebook",
        "Instagram",
        "KakaoTalk",
        "LINE",
        "LinkedIn",
        "Snapchat",
        "TikTok",
        "Twitter",
        "VKontakte",
        "WeChat",
        "WhatsApp",
        "YouTube",
    ]
    Purpose_Of_Use: Literal["Education", "Entertainment", "Networking", "News"]
    Country: str = Field(min_length=2, max_length=64)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_PATH.name}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float | str]:
    features = {
        "Study_Hours": request.Study_Hours,
        "Age": request.Age,
        "Avg_Daily_Usage_Hours": request.Avg_Daily_Usage_Hours,
        "Daily_Unlocks": request.Daily_Unlocks,
        "Physical_Activity_Hours": request.Physical_Activity_Hours,
        "Sleep_Hours_Per_Night": request.Sleep_Hours_Per_Night,
        "Stress_Level": request.Stress_Level,
        "Gender": request.Gender,
        "Academic_Level": request.Academic_Level,
        "Most_Used_Platform": request.Most_Used_Platform,
        "Purpose_Of_Use": request.Purpose_Of_Use,
        "Grouped_Country": request.Country if request.Country in TOP_COUNTRIES else "Other",
    }
    prediction = float(model.predict(pd.DataFrame([features]))[0])
    return {
        "predicted_score": round(prediction, 2),
        "target": "Mental_Health_Score",
    }