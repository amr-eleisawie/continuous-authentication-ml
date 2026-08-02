import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests


MODEL_PATH = Path(__file__).resolve().parents[1] / "tuned_behavior_focused_pipeline.pkl"
MODEL = joblib.load(MODEL_PATH)


def build_features(data):
    timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
    hour = timestamp.hour
    speed = max(float(data["Keyboard_Speed_WPM"]), 1e-6)
    return pd.DataFrame([{
        "Keyboard_Speed_WPM": data["Keyboard_Speed_WPM"],
        "Keystroke_Duration_ms": data["Keystroke_Duration_ms"],
        "Typing_Accuracy_pct": data["Typing_Accuracy_pct"],
        "Mouse_Speed_pxs": data["Mouse_Speed_pxs"],
        "Network_Status": data["Network_Status"],
        "CPU_Usage_pct": data["CPU_Usage_pct"],
        "Memory_Usage_pct": data["Memory_Usage_pct"],
        "Previous_Authentication_Score": data["Previous_Authentication_Score"],
        "Hour": hour,
        "Is_Weekend": int(timestamp.weekday() >= 5),
        "Hour_Sin": np.sin(2 * np.pi * hour / 24),
        "Hour_Cos": np.cos(2 * np.pi * hour / 24),
        "Keystroke_Efficiency": data["Keystroke_Duration_ms"] / speed,
        "Mouse_to_Keyboard_Ratio": data["Mouse_Speed_pxs"] / speed,
        "Typing_Accuracy_Adjusted_Speed": (
            data["Keyboard_Speed_WPM"] * data["Typing_Accuracy_pct"] / 100
        ),
    }])


def get_decision(risk_score):
    if risk_score < 0.30:
        return "ALLOW"
    if risk_score <= 0.70:
        return "REQUEST_OTP"
    return "BLOCK_AND_ALERT"


class handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_json(200, {"status": "ok"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length))
            features = build_features(data)
            risk_score = float(MODEL.predict_proba(features)[0, 1])
            decision = get_decision(risk_score)
            payload = {
                "session_id": data.get("session_id", "WEB_SESSION"),
                "timestamp": data["timestamp"],
                "risk_score": round(risk_score, 6),
                "decision": decision,
                "features": features.iloc[0].to_dict(),
            }

            webhook_url = os.getenv("N8N_WEBHOOK_URL")
            if webhook_url:
                requests.post(webhook_url, json=payload, timeout=10).raise_for_status()

            self.send_json(200, payload)
        except Exception as error:
            self.send_json(400, {"error": str(error)})
