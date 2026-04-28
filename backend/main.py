from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import base64
import cv2
import numpy as np
import os

# Import ฟังก์ชันของคุณ
from gloss_api import thai_to_gloss
# จาก holistic_data_extract ต้อง import ฟังก์ชัน extract_keypoints (คุณอาจต้องแยกฟังก์ชันออกจากลูป while ให้เรียกใช้ง่ายๆ)

app = FastAPI()

# อนุญาตให้หน้าเว็บยิง API เข้ามาได้
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ให้ FastAPI เปิดไฟล์วิดีโอและไฟล์ HTML ได้
app.mount("/videos", StaticFiles(directory="sign_videos"), name="videos")
app.mount("/frontend", StaticFiles(directory="../frontend"), name="frontend")

@app.get("/")
def read_root():
    return {"message": "SignBridge API is running!"}

# ---------------------------------------------------------
# API 1: Text-to-Sign (ฝั่งคนหูดีพิมพ์ -> แปลเป็นวิดีโอภาษามือ)
# ---------------------------------------------------------
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
            video_path = f"sign_videos/{gloss}.mp4"
            if os.path.exists(video_path):
                video_urls.append(f"/videos/{gloss}.mp4") # ส่ง URL ให้หน้าเว็บไปเล่นเอง
                
    return {"glosses": gloss_string, "video_sequence": video_urls}

# ---------------------------------------------------------
# API 2: Sign-to-Text (ฝั่งคนหูหนวกทำท่า -> แปลเป็นข้อความ)
# ---------------------------------------------------------
@app.post("/api/sign-to-text")
async def process_sign_to_text(request: Request):
    data = await request.json()
    image_base64 = data.get("image")
    
    # แปลง Base64 จากเว็บเป็นภาพ OpenCV
    encoded_data = image_base64.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # ตรงนี้คุณต้องนำ frame ไปเข้า mediapipe (ดัดแปลงจาก holistic_data_extract.py)
    # keypoints = extract_keypoints(frame)
    # prediction = random_forest_model.predict(keypoints)
    
    predicted_word = "สวัสดี" # Mockup ไว้ก่อนสำหรับ Demo
    
    return {"prediction": predicted_word}