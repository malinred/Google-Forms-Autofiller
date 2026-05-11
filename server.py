import os
import shutil
import fitz
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn
import cv2
import numpy as np

from main import run_inference
from model_test_inference import (
    preprocess_document, 
    get_ocr_words_and_boxes, 
    extract_table_data,
    classify_document,
    extract_aadhaar_data,
    extract_resume_data
)

app = FastAPI(title="Google Forms Autofiller API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup dirs
STATIC_DIR = "static"
TEMP_DIR = "temp_uploads"
DB_DIR = "database"
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend not found")
    with open(index_path, "r") as f:
        return f.read()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Upload doc
    file_path = os.path.join(TEMP_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        final_results = []
        if file.filename.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                native_text = page.get_text("text").strip()

                if native_text:
                    words_data = page.get_text("words")
                    words = []
                    boxes = []
                    p_width, p_height = page.rect.width, page.rect.height
                    for wd in words_data:
                        words.append(wd[4])
                        box = [
                            max(0, min(1000, int(1000 * wd[0] / p_width))),
                            max(0, min(1000, int(1000 * wd[1] / p_height))),
                            max(0, min(1000, int(1000 * wd[2] / p_width))),
                            max(0, min(1000, int(1000 * wd[3] / p_height)))
                        ]
                        boxes.append(box)
                    pix = page.get_pixmap(dpi=300)
                    page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                else:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    processed_img_bgr = preprocess_document(img)
                    words, boxes = get_ocr_words_and_boxes(processed_img_bgr)
                    page_img = Image.fromarray(cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB))

                doc_type = classify_document(words)

                if doc_type == "aadhaar":
                    pairs = extract_aadhaar_data(page_img)
                    # If QR failed, fall back to LayoutLMv3
                    if pairs and pairs[0][0] == "Error":
                        pairs = run_inference(page_img, words, boxes)
                elif doc_type == "resume":
                    pairs = extract_resume_data(raw_text=native_text, words=words)
                else:
                    pairs = run_inference(page_img, words, boxes)
                
                img_for_table = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
                tables = extract_table_data(img_for_table, words, boxes)
                
                final_results.append({
                    "page": page_num + 1, 
                    "type": doc_type,
                    "pairs": pairs,
                    "tables": tables
                })
        else:
            original_img_pil = Image.open(file_path)
            processed_img_bgr = preprocess_document(file_path)
            words, boxes = get_ocr_words_and_boxes(processed_img_bgr)
            page_img = Image.fromarray(cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB))
            doc_type = classify_document(words)

            if doc_type == "aadhaar":
                pairs = extract_aadhaar_data(original_img_pil)
                # If QR failed, fall back to LayoutLMv3
                if pairs and pairs[0][0] == "Error":
                    pairs = run_inference(page_img, words, boxes)
            elif doc_type == "resume":
                pairs = extract_resume_data(raw_text=None, words=words)
            else:
                pairs = run_inference(page_img, words, boxes)
            
            tables = extract_table_data(processed_img_bgr, words, boxes)
            final_results.append({
                "page": 1, 
                "type": doc_type,
                "pairs": pairs,
                "tables": tables
            })

        return {"filename": file.filename, "results": final_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/profiles")
async def get_profiles():
    # List profiles
    try:
        profiles = [f.replace(".json", "") for f in os.listdir(DB_DIR) if f.endswith(".json")]
        return {"profiles": sorted(profiles)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list profiles: {str(e)}")

@app.get("/profiles/{profile_name}")
async def get_profile(profile_name: str):
    # Get profile
    safe_name = "".join(c for c in profile_name if c.isalnum() or c in ('-', '_')).strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid profile name")

    profile_path = os.path.join(DB_DIR, f"{safe_name}.json")
    if not os.path.exists(profile_path):
        raise HTTPException(status_code=404, detail=f"Profile '{safe_name}' not found")

    with open(profile_path, "r") as f:
        return json.load(f)

@app.post("/save")
async def save_profile(payload: dict):
    # Save profile
    try:
        profile_name = payload.get("profile_name")
        if not profile_name:
            raise HTTPException(status_code=400, detail="Profile name is required")

        document_data = payload.get("document_data", {})
        new_fields = {}
        for res in document_data.get("results", []):
            for pair in res.get("pairs", []):
                if len(pair) == 2 and pair[0].strip():
                    new_fields[pair[0].strip()] = pair[1].strip()

        safe_profile_name = "".join(c for c in profile_name if c.isalnum() or c in ('-', '_')).strip()
        if not safe_profile_name:
            raise HTTPException(status_code=400, detail="Invalid profile name")

        save_path = os.path.join(DB_DIR, f"{safe_profile_name}.json")
        profile_data = {}
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                profile_data = json.load(f)

        profile_data.update(new_fields)
        with open(save_path, "w") as f:
            json.dump(profile_data, f, indent=4)
        return {"status": "success", "message": f"Successfully saved to profile: {safe_profile_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save profile: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
