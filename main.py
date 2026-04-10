import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from pymongo import MongoClient
from google import genai
from google.genai import types

# Database setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
try:
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    mongo_db = mongo_client["thaiproof_db"]
    logs_collection = mongo_db["rearrange_logs"]
    
    # Test connection
    mongo_client.admin.command('ping')
    db_available = True
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Warning: Could not connect to MongoDB database at {MONGODB_URI}: {e}")
    db_available = False

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDghdpHGryRKilKSvPYae4LxuU1qF91vnI")
client = genai.Client(api_key=GEMINI_API_KEY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

# Mount static files
# Make sure the static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class ToolRequest(BaseModel):
    text: str
    tool_id: str = "rearrange"
    option: str = ""

class ToolResponse(BaseModel):
    result_text: str

TOOLS_PROMPTS = {
    "check_spelling": "คุณคือผู้เชี่ยวชาญด้านการพิสูจน์อักษรภาษาไทย ตรวจสอบและแก้ไขคำผิด ไวยากรณ์ และการเว้นวรรคให้ถูกต้องตามหลักภาษาไทย ตอบกลับเฉพาะข้อความที่แก้ไขแล้ว หรือบอกว่า 'ไม่พบคำผิด' หากข้อความถูกต้องอยู่แล้ว ห้ามอธิบายเพิ่ม",
    "rearrange": "คุณคือผู้เชี่ยวชาญด้านภาษาไทย นำประโยคที่ผู้ใช้พิมพ์มาเรียบเรียงใหม่ให้สละสลวย เป็นธรรมชาติ อ่านง่าย ถูกต้องตามหลักภาษาไทย รักษาความหมายเดิม 100% ตอบกลับเฉพาะข้อความที่เรียบเรียงแล้วเท่านั้น ห้ามอธิบายเพิ่ม",
    "think_sentence": "คุณคือนักเขียนมืออาชีพและครีเอทีฟ นำเนื้อหาหรือไอเดียสั้นๆ ที่ผู้ใช้ให้มา แต่งให้เป็นประโยคที่สมบูรณ์ น่าสนใจ และสละสลวย ตอบกลับเฉพาะประโยคที่คิดให้เท่านั้น ห้ามอธิบายเพิ่ม",
    "official_letter": "คุณคือผู้เชี่ยวชาญด้านงานสารบรรณและภาษาหนังสือราชการไทย นำข้อความที่ผู้ใช้พิมพ์มาเรียบเรียงใหม่ให้อยู่ในรูปแบบภาษาทางการ เหมาะสำหรับใช้ในหนังสือราชการหรือจดหมายทางการ ตอบกลับเฉพาะข้อความที่เรียบเรียงแล้วเท่านั้น ห้ามอธิบายเพิ่ม",
    "speech_script": "คุณคือนักเขียนบทพูดและพิธีกรมืออาชีพ นำเนื้อหาที่ผู้ใช้ให้มาเรียบเรียงเป็นสคริปต์บทพูดที่ลื่นไหล เป็นธรรมชาติ เหมาะกับการพูดออกเสียง มีจังหวะหายใจที่ดี ตอบกลับเฉพาะบทพูดเท่านั้น ห้ามอธิบายเพิ่ม",
    "help_word": "คุณคือนักคัดสรรคำศัพท์และก็อปปี้ไรเตอร์ภาษาไทย นำความหมายหรือบริบทที่ผู้ใช้ให้มา แนะนำคำศัพท์ที่เหมาะสม สละสลวย หรือดูเป็นมืออาชีพมากขึ้น (ให้มาสัก 3-5 คำ) ตอบกลับเฉพาะคำศัพท์ที่แนะนำ ห้ามเกริ่นนำหรืออธิบายเพิ่มยาวๆ"
}

@app.post("/api/generate", response_model=ToolResponse)
def generate_text(request: ToolRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        system_instruction = TOOLS_PROMPTS.get(request.tool_id, TOOLS_PROMPTS["rearrange"])
        
        if request.option:
            system_instruction += f"\n\nเงื่อนไขเพิ่มเติมสำหรับการทำงานครั้งนี้: ให้เน้นและอ้างอิงตามรูปแบบ: {request.option}"
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            )
        )
        result_text = response.text.strip()
        
        # Save to database (we save all types of logs in the same collection for now)
        if db_available:
            try:
                log_entry = {
                    "tool_id": request.tool_id,
                    "original_text": request.text,
                    "rearranged_text": result_text,
                    "created_at": datetime.now(timezone.utc)
                }
                logs_collection.insert_one(log_entry)
            except Exception as db_e:
                print(f"Database error: {db_e}")
            
        return ToolResponse(result_text=result_text)
    except Exception as e:
        print(f"Gemini API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")
