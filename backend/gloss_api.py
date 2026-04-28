# -*- coding: utf-8 -*-
from dotenv import load_dotenv
from google import genai
import os

# ===== โหลด API Key จากไฟล์ .env =====
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("ไม่พบ API_KEY ในไฟล์ .env หรือ Environment Variable")

# ===== Setup Gemini (Updated to google-genai SDK) =====
client = genai.Client(api_key=API_KEY)

# ===== Sign Gloss Dictionary =====
AVAILABLE_GLOSSES = [
    "สวัสดี", "ขอบคุณ", "ใช่", "ไม่ใช่",
    "ฉัน", "คุณ", "เขา", "เรา",
    "กิน", "ดื่ม", "ไป", "มา", "ต้องการ",
    "น้ำ", "อาหาร", "บ้าน", "โรงพยาบาล",
    "ช่วย", "เจ็บ", "ดี", "ไม่ดี",
    "อาจารย์", "หล่อ", "มาก"
]

# ===== Prompt สำหรับ Gemini =====
def build_prompt(thai_text):
    glosses_str = ", ".join(AVAILABLE_GLOSSES)
    return f"""คุณคือระบบแปลภาษาไทยเป็น Thai Sign Language Gloss

กฎการแปล:
1. แปลงประโยคภาษาไทยเป็น Gloss โดยใช้เฉพาะคำที่มีในรายการนี้เท่านั้น: {glosses_str}
2. ตัดคำที่ไม่จำเป็นออก เช่น ครับ ค่ะ นะ เลย
3. เรียงคำตาม grammar ภาษามือไทย (ประธาน + กริยา + กรรม)
4. ตอบกลับเป็น Gloss เท่านั้น คั่นด้วยช่องว่าง ไม่มีคำอธิบายเพิ่มเติม
5. ถ้าคำไหนไม่มีในรายการ ให้ข้ามคำนั้นไป

ประโยคที่ต้องแปล: {thai_text}

Gloss:"""

# ===== แปลประโยคไทยเป็น Gloss =====
def thai_to_gloss(thai_text):
    print(f"\n{'='*50}")
    print(f"ประโยคไทย : {thai_text}")
    prompt = build_prompt(thai_text)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        gloss = response.text.strip()
        print(f"Sign Gloss : {gloss}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: {e}")
        gloss = ""
        
    print(f"{'='*50}")
    return gloss

# ===== test =====
if __name__ == "__main__":
    test_sentences = [
        "อาจารย์หล่อมากเลยครับ"
    ]
    for sentence in test_sentences:
        thai_to_gloss(sentence)