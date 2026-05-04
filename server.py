import os
import shutil
import fitz  # PyMuPDF
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import uvicorn

# Import inference functions from existing modules
from main import run_inference
from model_test_inference import preprocess_document, get_ocr_words_and_boxes

app = FastAPI(title="Google Forms Autofiller API")

# Setup directories
STATIC_DIR = "static"
TEMP_DIR = "temp_uploads"
DB_DIR = "database"
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Mount static files
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
    """
    Upload a document, process it through the LayoutLMv3 pipeline, and return extracted fields.
    """
    file_path = os.path.join(TEMP_DIR, file.filename)
    
    # Save the uploaded file temporarily
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        final_results = []
        
        if file.filename.lower().endswith(".pdf"):
            # Handle PDF (multi-page)
            doc = fitz.open(file_path)
            print(f"[Server] Processing PDF: {file.filename} ({len(doc)} pages)")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Try native PDF text first (fastest/most accurate)
                words_data = page.get_text("words")
                
                if not words_data:
                    # Fallback to OCR if no native text
                    processed_img = preprocess_document(img)
                    words, boxes = get_ocr_words_and_boxes(processed_img)
                    page_img = processed_img
                else:
                    words = []
                    boxes = []
                    p_width, p_height = page.rect.width, page.rect.height
                    for wd in words_data:
                        # wd: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
                        words.append(wd[4])
                        box = [
                            int(1000 * wd[0] / p_width),
                            int(1000 * wd[1] / p_height),
                            int(1000 * wd[2] / p_width),
                            int(1000 * wd[3] / p_height)
                        ]
                        boxes.append(box)
                    page_img = img
                
                pairs = run_inference(page_img, words, boxes)
                final_results.append({"page": page_num + 1, "pairs": pairs})
        
        else:
            # Handle Image
            print(f"[Server] Processing Image: {file.filename}")
            processed_img = preprocess_document(file_path)
            words, boxes = get_ocr_words_and_boxes(processed_img)
            pairs = run_inference(processed_img, words, boxes)
            final_results.append({"page": 1, "pairs": pairs})

        return {
            "filename": file.filename,
            "results": final_results
        }

    except Exception as e:
        print(f"[Error] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/profiles")
async def get_profiles():
    """
    List all available profiles in the database directory.
    """
    try:
        profiles = [f.replace(".json", "") for f in os.listdir(DB_DIR) if f.endswith(".json")]
        return {"profiles": sorted(profiles)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list profiles: {str(e)}")

@app.post("/save")
async def save_profile(payload: dict):
    """
    Save or update the edited profile data into a specific profile JSON file.
    """
    try:
        profile_name = payload.get("profile_name")
        if not profile_name:
            raise HTTPException(status_code=400, detail="Profile name is required")
            
        document_data = payload.get("document_data", {})
        
        # Flatten the extracted key-value pairs into a simple dictionary
        new_fields = {}
        for res in document_data.get("results", []):
            for pair in res.get("pairs", []):
                if len(pair) == 2 and pair[0].strip():
                    new_fields[pair[0].strip()] = pair[1].strip()

        # Sanitize profile name to prevent directory traversal
        safe_profile_name = "".join(c for c in profile_name if c.isalnum() or c in ('-', '_')).strip()
        if not safe_profile_name:
            raise HTTPException(status_code=400, detail="Invalid profile name")

        save_path = os.path.join(DB_DIR, f"{safe_profile_name}.json")
        
        profile_data = {}
        # If profile exists, load it to append/update
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                profile_data = json.load(f)
                
        # Update existing profile with new fields (overwrites existing keys, adds new ones)
        profile_data.update(new_fields)
        
        with open(save_path, "w") as f:
            json.dump(profile_data, f, indent=4)
            
        print(f"[Server] Saved/Updated profile: {save_path}")
        return {"status": "success", "message": f"Successfully saved to profile: {safe_profile_name}"}
    except Exception as e:
        print(f"[Error] Save profile error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save profile: {str(e)}")

if __name__ == "__main__":
    print("Starting server on http://localhost:8000")
    # For reload to work, app must be passed as an import string
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
