import asyncio
import base64
import os
import re

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from gloss_api import thai_to_gloss, AVAILABLE_GLOSSES
from holistic_data_extract import extract_keypoints
from sign_model import DEFAULT_MODEL_PATH, load_trained_model, predict_sign


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await manager.broadcast(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
VIDEO_DIR = os.path.join(BASE_DIR, "sign_videos")

if os.path.exists(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

if os.path.exists(VIDEO_DIR):
    app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")


def normalize_token(token: str) -> str:
    return re.sub(r"[^\wก-๙]", "", (token or "").strip())


def resolve_video_sequence(thai_text: str, gloss_string: str):
    video_urls = []
    resolved_glosses = []

    def append_if_exists(token: str):
        normalized = normalize_token(token)
        if not normalized:
            return

        video_path = os.path.join(VIDEO_DIR, f"{normalized}.mp4")
        if os.path.exists(video_path):
            resolved_glosses.append(normalized)
            video_urls.append(f"/videos/{normalized}.mp4")

    normalized_text = normalize_token(thai_text)
    if normalized_text and os.path.exists(os.path.join(VIDEO_DIR, f"{normalized_text}.mp4")):
        return [normalized_text], [f"/videos/{normalized_text}.mp4"]

    if gloss_string:
        for gloss in gloss_string.replace("\n", " ").split():
            append_if_exists(gloss)

    return resolved_glosses, video_urls


@app.post("/api/text-to-sign")
async def process_text_to_sign(request: Request):
    data = await request.json()
    thai_text = (data.get("text") or "").strip()
    
    if not thai_text:
        return {"glosses": "", "video_sequence": []}

    # Fast path: check if there's an exact match video for the full text
    normalized_text = normalize_token(thai_text)
    if normalized_text and os.path.exists(os.path.join(VIDEO_DIR, f"{normalized_text}.mp4")):
        await asyncio.sleep(0.8)
        return {
            "glosses": normalized_text,
            "video_sequence": [f"/videos/{normalized_text}.mp4"],
        }

    # Medium path: try to tokenize using greedy matching with available glosses
    sorted_glosses = sorted(AVAILABLE_GLOSSES, key=len, reverse=True)
    def greedy_tokenize(text):
        tokens = []
        text = text.replace(" ", "")
        i = 0
        while i < len(text):
            match = None
            for word in sorted_glosses:
                if text[i:].startswith(word):
                    match = word
                    break
            if match:
                tokens.append(match)
                i += len(match)
            else:
                i += 1  # skip unknown character
        return tokens

    tokens = greedy_tokenize(normalized_text)
    if tokens:
        # Filter only tokens that have corresponding videos
        valid_tokens = [t for t in tokens if os.path.exists(os.path.join(VIDEO_DIR, f"{t}.mp4"))]
        if valid_tokens:
            return {
                "glosses": " ".join(valid_tokens),
                "video_sequence": [f"/videos/{t}.mp4" for t in valid_tokens],
            }

    # Slow path: use Gemini to translate sentence to glosses
    gloss_string = await run_in_threadpool(thai_to_gloss, thai_text)
    resolved_glosses, video_urls = resolve_video_sequence(thai_text, gloss_string)

    return {
        "glosses": " ".join(resolved_glosses) if resolved_glosses else gloss_string,
        "video_sequence": video_urls,
    }


mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
sign_classifier = load_trained_model(DEFAULT_MODEL_PATH)


@app.post("/api/sign-to-text")
async def process_sign_to_text(request: Request):
    global sign_classifier
    data = await request.json()
    image_base64 = data.get("image")

    if not image_base64:
        return {"error": "No image received"}

    try:
        encoded_data = image_base64.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic_model.process(image_rgb)
        keypoints = extract_keypoints(results)

        if not (results.right_hand_landmarks or results.left_hand_landmarks):
            return {"prediction": "", "confidence": 0.0, "status": "no_hands_detected"}

        if sign_classifier is None:
            sign_classifier = load_trained_model(DEFAULT_MODEL_PATH)

        if sign_classifier is None:
            return {
                "prediction": "",
                "confidence": 0.0,
                "status": "model_not_trained",
                "model_path": DEFAULT_MODEL_PATH,
            }

        predicted_word, confidence = predict_sign(keypoints, sign_classifier)
        return {"prediction": predicted_word, "confidence": confidence, "status": "ok"}

    except Exception as exc:
        print(f"Error processing frame: {exc}")
        return {"prediction": "", "confidence": 0.0, "status": "error"}
