from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import base64
import cv2
import numpy as np
import os

# โหลดฟังก์ชันของคุณจากไฟล์อื่นๆ
import mediapipe as mp
from holistic_data_extract import extract_keypoints
from gloss_api import thai_to_gloss # 📍 นำเข้าตัวเรียก Gemini

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ==========================================
# 📍 1. การเชื่อมโฟลเดอร์ (Mount Static Files)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
VIDEO_DIR = os.path.join(BASE_DIR, "sign_videos") # โฟลเดอร์เก็บวิดีโอ

# สั่งให้ FastAPI รู้จักโฟลเดอร์ frontend และ videos
if os.path.exists(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
else:
    print(f"❌ หาโฟลเดอร์ Frontend ไม่เจอที่: {FRONTEND_DIR}")

if os.path.exists(VIDEO_DIR):
    app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")
else:
    print(f"❌ หาโฟลเดอร์ sign_videos ไม่เจอที่: {VIDEO_DIR}")


# ==========================================
# 📍 2. API: Text-to-Sign (แปลข้อความเป็นวิดีโอ)
# ==========================================
@app.post("/api/text-to-sign")
async def process_text_to_sign(request: Request):
    data = await request.json()
    thai_text = data.get("text")
    
    # 1. เรียก Gemini ของคุณแปลเป็น Gloss
    gloss_string = thai_to_gloss(thai_text)
    
    # 2. ค้นหาวิดีโอที่ตรงกับคำ
    video_urls = []
    if gloss_string:
        glosses = gloss_string.strip().split()
        for gloss in glosses:
            video_path = os.path.join(VIDEO_DIR, f"{gloss}.mp4")
            if os.path.exists(video_path):
                # ส่ง URL กลับไปให้หน้าเว็บเล่น
                video_urls.append(f"/videos/{gloss}.mp4")
            else:
                print(f"⚠️ ไม่พบไฟล์วิดีโอ: {gloss}.mp4")
                
    return {"glosses": gloss_string, "video_sequence": video_urls}


# ==========================================
# 📍 3. API: Sign-to-Text (MediaPipe)
# ==========================================
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.post("/api/sign-to-text")
async def process_sign_to_text(request: Request):
    data = await request.json()
    image_base64 = data.get("image")
    
    if not image_base64:
        return {"error": "No image received"}

    try:
        # แปลงภาพเข้า MediaPipe
        encoded_data = image_base64.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = holistic_model.process(image_rgb)
        keypoints = extract_keypoints(results)
        
        # จำลองคำตอบกลับไป
        if results.right_hand_landmarks or results.left_hand_landmarks:
            predicted_word = "✅👌จับท่าทางได้แล้ว!" 
        else:
            predicted_word = "❌👋ไม่พบมือในกล้อง"

        return {"prediction": predicted_word}

    except Exception as e:
        print(f"Error processing frame: {e}")
        return {"prediction": ""}