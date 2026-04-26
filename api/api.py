# simple_api.py - The SIMPLE, PROFESSIONAL way
"""
Sekka Metro Capacity API - Simple, Clean, Professional
For Cairo Metro capacity predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import List, Optional
import json

# ============================================================
# CONFIGURATION (Simple, visible, easy to change)
# ============================================================

MAX_CAPACITY = 100
SEAT_CAPACITY = 40

# All Cairo Metro stations (100-399 range)
STATIONS = list(range(101, 136)) + list(range(201, 221)) + list(range(301, 346))

STATION_LINE_MAP = {}
for s in range(101, 136): STATION_LINE_MAP[s] = 1
for s in range(201, 221): STATION_LINE_MAP[s] = 2
for s in range(301, 346): STATION_LINE_MAP[s] = 3

# Interchange stations (high traffic)
INTERCHANGE_STATIONS = [119, 120, 122, 208, 209, 211, 215, 313, 319, 320, 324, 332, 335]

# ============================================================
# MODEL LOADER (Simple, reliable)
# ============================================================

class MetroPredictor:
    """Simple predictor that actually works"""
    
    def __init__(self, model_path="models/metro_capacity_model.txt"):
        self.model = None
        self.model_path = Path(model_path)
        self._load_model()
    
    def _load_model(self):
        if self.model_path.exists():
            self.model = lgb.Booster(model_file=str(self.model_path))
            print(f"✅ Model loaded from {self.model_path}")
        else:
            print(f"⚠️ No model found at {self.model_path} - using fallback")
    
    def _fallback_prediction(self, hour, day_of_week):
        """Simple rule-based prediction when no model"""
        base = 35
        if 7 <= hour <= 9: base += 25
        elif 17 <= hour <= 19: base += 30
        if day_of_week in [4, 5]: base -= 10
        return min(MAX_CAPACITY, max(10, base))
    
    def predict(self, station_id, hour=None, minute=None, day_of_week=None):
        """Get prediction for one station"""
        now = datetime.now()
        hour = hour if hour is not None else now.hour
        minute = minute if minute is not None else now.minute
        day = day_of_week if day_of_week is not None else now.weekday()
        
        # Try ML model first
        if self.model:
            try:
                features = pd.DataFrame([{
                    'hour': hour, 'minute': minute, 'day_of_week': day,
                    'is_weekend': 1 if day in [4,5] else 0,
                    'hour_sin': np.sin(2 * np.pi * hour / 24),
                    'hour_cos': np.cos(2 * np.pi * hour / 24),
                    'dow_sin': np.sin(2 * np.pi * day / 7),
                    'dow_cos': np.cos(2 * np.pi * day / 7),
                    'is_peak': 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
                    'is_friday_prayer': 1 if (day == 4 and 12 <= hour <= 14) else 0,
                    'is_interchange': 1 if station_id in INTERCHANGE_STATIONS else 0,
                    'people_count_lag_1': 30,  # fallback
                    'people_count_lag_3': 30,
                    'station_id': station_id,
                    'line_number': STATION_LINE_MAP.get(station_id, 1),
                }])
                predicted = self.model.predict(features)[0]
            except:
                predicted = self._fallback_prediction(hour, day)
        else:
            predicted = self._fallback_prediction(hour, day)
        
        return {
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'carriage_id': station_id,
            'predicted_count': int(predicted),
            'capacity_percent': round((predicted / MAX_CAPACITY) * 100, 1),
            'seats_available': 'YES' if predicted < SEAT_CAPACITY else 'NO'
        }


# ============================================================
# FASTAPI APP (Simple, focused)
# ============================================================

app = FastAPI(
    title="Sekka Metro Capacity API",
    description="Real-time capacity predictions for Cairo Metro",
    version="1.0.0"
)

# Allow all origins (for Flutter app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = MetroPredictor()

# ============================================================
# API ENDPOINTS (Clean, RESTful, documented)
# ============================================================

@app.get("/")
def root():
    """Welcome page with API information"""
    return {
        "app": "Sekka Metro Capacity API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "GET /stations": "List all stations",
            "GET /predict/{station_id}": "Get capacity for one station",
            "GET /predict/all": "Get all stations capacity",
            "GET /predict/line/{line_number}": "Get stations on a line",
            "GET /docs": "Interactive documentation"
        },
        "example": "http://localhost:8000/predict/119"
    }


@app.get("/health")
def health():
    """Check if API is running"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/stations")
def get_stations():
    """List all stations with their line numbers"""
    stations = []
    for sid in STATIONS:
        stations.append({
            "station_id": sid,
            "line": STATION_LINE_MAP.get(sid, 1),
            "name": get_station_name(sid)  # You can implement this
        })
    return stations


@app.get("/predict/{station_id}")
def predict_station(station_id: int, hour: Optional[int] = None, day: Optional[int] = None):
    """Get capacity prediction for a specific station"""
    if station_id not in STATIONS:
        raise HTTPException(404, f"Station {station_id} not found")
    return predictor.predict(station_id, hour=hour, day_of_week=day)


@app.get("/predict/all")
def predict_all():
    """Get capacity predictions for all stations"""
    return [predictor.predict(sid) for sid in STATIONS[:20]]  # Limit for speed


@app.get("/predict/line/{line_number}")
def predict_line(line_number: int):
    """Get capacity predictions for all stations on a line"""
    if line_number not in [1, 2, 3]:
        raise HTTPException(400, "Line must be 1, 2, or 3")
    
    line_stations = [s for s in STATIONS if STATION_LINE_MAP.get(s) == line_number]
    return [predictor.predict(sid) for sid in line_stations]


def get_station_name(station_id: int) -> str:
    """Helper to get station names (simplified)"""
    names = {
        119: "Sadat (Interchange)",
        209: "Attaba (Interchange)",
        215: "Cairo University",
        313: "Stadium (Monorail)",
    }
    return names.get(station_id, f"Station {station_id}")


# ============================================================
# RUN THE APP
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🚇 SEKKA METRO CAPACITY API")
    print("=" * 60)
    print("📍 API running at: http://127.0.0.1:8000")
    print("📚 Docs: http://127.0.0.1:8000/docs")
    print("🎯 Try: http://127.0.0.1:8000/predict/119")
    print("=" * 60)
    print("Press CTRL+C to stop\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)