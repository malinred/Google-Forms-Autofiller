import os
import re
import paddle
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch
import numpy as np
import cv2
from paddleocr import PaddleOCR, PPStructure

try:
    from pyaadhaar.decode import AadhaarSecureQr, AadhaarOldQr
    from pyaadhaar.utils import isSecureQr
except ImportError:
    pass

# OCR engines
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False, enable_mkldnn=False)
table_engine = PPStructure(show_log=False, layout=True, lang='en', image_orientation=False, use_gpu=False)

# Preprocess doc
def preprocess_document(image_input):
    print("DEBUG: Starting document preprocessing")
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        img_cv = cv2.imread(image_input)
    elif isinstance(image_input, Image.Image):
        img_cv = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    else:
        img_cv = image_input

    if img_cv is None:
        raise ValueError("Failed to load image")

    h, w = img_cv.shape[:2]
    print(f"DEBUG: Image loaded, size: {w}x{h}")

    # Resize
    min_long_edge = 1200
    if max(h, w) < min_long_edge:
        scale = min_long_edge / max(h, w)
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        print(f"DEBUG: Image resized to {int(w * scale)}x{int(h * scale)}")

    # CLAHE
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Denoise
    final_img = cv2.fastNlMeansDenoisingColored(final_img, None, 10, 10, 7, 21)
    print("DEBUG: Document preprocessing completed")
    return final_img

# PaddleOCR inference
def get_ocr_words_and_boxes(image_cv):
    print("DEBUG: Starting OCR")
    words = []
    boxes = []
    h, w = image_cv.shape[:2]

    try:
        results = ocr_engine.ocr(image_cv, cls=True)
        if results and results[0]:
            for line in results[0]:
                poly = line[0]
                text, conf = line[1]
                if conf >= 0.60:
                    words.append(text)
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    box = [
                        max(0, min(1000, int(1000 * min(xs) / w))),
                        max(0, min(1000, int(1000 * min(ys) / h))),
                        max(0, min(1000, int(1000 * max(xs) / w))),
                        max(0, min(1000, int(1000 * max(ys) / h)))
                    ]
                    boxes.append(box)
        print(f"DEBUG: OCR completed, found {len(words)} words")
    except Exception:
        print("DEBUG: OCR failed")

    return words, boxes

# Classify doc
def classify_document(words):
    print("DEBUG: Starting document classification")
    text = " ".join(words).lower()
    
    
    aadhaar_keywords = ["government of india", "aadhaar", "unique identification", "enrollment", "male", "female"]
    aadhaar_pattern = r"\d{4}\s\d{4}\s\d{4}"
    if any(k in text for k in aadhaar_keywords) or re.search(aadhaar_pattern, text):
        print("DEBUG: Classified as Aadhaar")
        return "aadhaar"
        
    
    resume_keywords = ["experience", "education", "skills", "projects", "summary", "objective", "achievement", "curriculum vitae"]
    resume_hits = sum(1 for k in resume_keywords if k in text)
    if resume_hits >= 3:
        print("DEBUG: Classified as Resume")
        return "resume"
        
    print("DEBUG: Classified as General")
    return "general"

# Aadhaar QR extraction
def extract_aadhaar_data(image_pil):
    print("DEBUG: Starting Aadhaar QR extraction")
    try:
        detector = cv2.QRCodeDetector()
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        raw_data, points, _ = detector.detectAndDecode(image_cv)

        if not raw_data:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            raw_data, points, _ = detector.detectAndDecode(thresh)

        if not raw_data:
            print("DEBUG: No QR code found")
            return []

        print("DEBUG: QR code detected")
        if isSecureQr(raw_data):
            obj = AadhaarSecureQr(raw_data)
        else:
            obj = AadhaarOldQr(raw_data)

        data = obj.decodeddata()
        pairs = []
        for key, value in data.items():
            if key not in ["image", "signature", "adhaar_last_4_digit"]:
                pairs.append((str(key).replace("_", " ").title(), str(value)))

        if isSecureQr(raw_data) and hasattr(obj, "isImage") and obj.isImage():
            try:
                obj.saveimage("aadhaar_photo.jpg")
            except Exception:
                pass

        print(f"DEBUG: Aadhaar extraction completed, found {len(pairs)} fields")
        return pairs
    except Exception:
        print("DEBUG: Aadhaar extraction failed")
        return []

# Table extraction
def extract_table_data(image_cv, words=None, boxes=None):
    print("DEBUG: Starting table extraction")
    tables = []
    try:
        result = table_engine(image_cv)
        for region in result:
            if region['type'] == 'table':
                table_res = region.get('res', {})
                html = table_res.get('html', '')
                if html:
                    tables.append({
                        "type": "table",
                        "bbox": region.get('bbox', []),
                        "html": html
                    })
        print(f"DEBUG: Table extraction completed, found {len(tables)} tables")
    except Exception:
        print("DEBUG: Table extraction failed")

    # Heuristic fallback
    if not tables and words and boxes:
        print("DEBUG: Using heuristic table fallback")
        try:
            keywords = ["item", "qty", "quantity", "price", "amount", "total"]
            header_indices = [i for i, w in enumerate(words) if w.lower() in keywords]
            
            if len(header_indices) >= 2:
                h, w = image_cv.shape[:2]
                min_y = min([boxes[i][1] for i in header_indices])
                pixel_min_y = int(min_y * h / 1000) - 10
                crop_img = image_cv[max(0, pixel_min_y):, :]
                single_table_engine = PPStructure(show_log=False, layout=False, lang='en')
                res = single_table_engine(crop_img)
                
                if res and res[0]['type'] == 'table':
                    table_res = res[0].get('res', {})
                    html = table_res.get('html', '')
                    if html:
                        tables.append({
                            "type": "table",
                            "bbox": [0, min_y, 1000, 1000],
                            "html": html,
                            "note": "heuristic"
                        })
                        print("DEBUG: Heuristic table found")
        except Exception:
            print("DEBUG: Heuristic table extraction failed")
        
    return tables

# Resume extraction
def extract_resume_data(raw_text=None, words=None):
    print("DEBUG: Starting resume extraction")
    if raw_text and raw_text.strip():
        full_text = raw_text
    elif words:
        lines = []
        current_line = []
        for i, word in enumerate(words):
            current_line.append(word)
            if i > 0 and i % 8 == 0:
                lines.append(" ".join(current_line))
                current_line = []
        if current_line:
            lines.append(" ".join(current_line))
        full_text = "\n".join(lines)
    else:
        print("DEBUG: No text available for resume extraction")
        return []
    
    text_flat = " ".join(full_text.splitlines())
    all_lines = [l.rstrip() for l in full_text.splitlines()]
    pairs = []

    # Detect headers
    SECTION_KEYWORDS = {
        "skills": ["skills", "technical skills", "technologies", "tech stack", "core competencies"],
        "education": ["education", "academic background", "qualification", "academics"],
        "experience": ["experience", "work experience", "employment history", "internship", "internships"],
        "projects": ["projects", "project work", "personal projects", "academic projects"],
        "publications": ["publications", "papers", "research", "research work"],
        "certifications": ["certifications", "certificates", "courses", "training"],
        "achievements": ["achievements", "awards", "honors", "accomplishments"],
    }

    def identify_section(line):
        clean = re.sub(r"[^a-zA-Z\s]", "", line).strip().lower()
        if not clean or len(clean.split()) > 5:
            return None
        for section, keywords in SECTION_KEYWORDS.items():
            for kw in keywords:
                if kw == clean or clean.startswith(kw) or clean.endswith(kw):
                    return section
        return None

    # Parse sections
    sections = {"header": []}
    current_section = "header"
    for line in all_lines:
        sec = identify_section(line)
        if sec:
            current_section = sec
            sections[sec] = []
        else:
            sections.setdefault(current_section, []).append(line)

    # Extraction logic
    email_match = re.findall(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text_flat)
    if email_match: pairs.append(("Email", email_match[0]))

    phones = re.findall(r"(?<!\d)(\+?91[\s\-]?[6-9]\d{9}|[6-9]\d{9})(?!\d)", text_flat)
    if phones: pairs.append(("Phone", phones[0]))

    # Name check
    HEADER_SKIP = {"resume", "curriculum", "vitae", "cv", "profile", "contact", "education", "experience", "skills", "projects"}
    for line in [l.strip() for l in sections.get("header", []) if l.strip()][:5]:
        clean_words = [re.sub(r"[^a-zA-Z]", "", w) for w in line.split() if re.sub(r"[^a-zA-Z]", "", w)]
        if 2 <= len(clean_words) <= 4 and all(len(w) > 1 for w in clean_words) and not any(w.lower() in HEADER_SKIP for w in clean_words) and not re.search(r"\d", line) and not re.search(r"[@|/\\]", line):
            pairs.append(("Name", line.strip()))
            break

    linkedin = re.findall(r"linkedin\.com/in/[\w\-]+", text_flat, re.IGNORECASE)
    if linkedin: pairs.append(("LinkedIn", linkedin[0]))

    github = re.findall(r"github\.com/[\w\-]+", text_flat, re.IGNORECASE)
    if github: pairs.append(("GitHub", github[0]))

    if "skills" in sections:
        raw_skills = re.sub(r"[•\-–|●▪◦âĢ¢âĢĵÂ§]+", ",", " ".join([l.strip() for l in sections["skills"] if l.strip()]))
        raw_skills = re.sub(r",\s*,", ",", raw_skills)
        raw_skills = re.sub(r"\s+", " ", raw_skills).strip().strip(",")
        if raw_skills: pairs.append(("Skills", raw_skills))

    if "education" in sections:
        content = [l.strip() for l in sections["education"] if len(l.strip()) > 5]
        if content:
            pairs.append(("Education", content[0]))
            edu_text = " ".join(content)
            cgpa = re.findall(r"cgpa[:\s]+(\d+\.?\d*)", edu_text, re.IGNORECASE)
            if cgpa: pairs.append(("CGPA", cgpa[0]))
            pct = re.findall(r"(\d{2,3}\.?\d*)\s*%", edu_text)
            if pct: pairs.append(("Percentage", pct[0] + "%"))

    if "experience" in sections:
        content = [l.strip() for l in sections["experience"] if len(l.strip()) > 5]
        if content: pairs.append(("Experience", content[0]))
        if any(kw in " ".join(sections["experience"]).lower() for kw in ["internship", "intern", "interned", "internships", "trainee", "apprentice"]):
            pairs.append(("Done internship", "yes"))

    if "projects" in sections:
        proj_names = []
        for line in [l.strip() for l in sections["projects"] if l.strip()]:
            stripped = re.sub(r"^[•\-–âĢ¢\s]+", "", line).strip()
            if stripped and not re.match(r"^(tools?|tech|using|languages?)[:\s]", stripped, re.IGNORECASE) and 5 < len(stripped) and len(stripped.split()) <= 10:
                proj_names.append(stripped)
        if proj_names: pairs.append(("Projects", " | ".join(proj_names[:5])))
        tool_lines = re.findall(r"Tools?[:\s]+([^\n]+)", "\n".join(sections["projects"]), re.IGNORECASE)
        if tool_lines: pairs.append(("Project Tools", " | ".join(t.strip() for t in tool_lines)))

    if "certifications" in sections:
        content = [l.strip() for l in sections["certifications"] if len(l.strip()) > 5]
        if content: pairs.append(("Certifications", content[0]))

    if "achievements" in sections:
        content = [l.strip() for l in sections["achievements"] if len(l.strip()) > 5]
        if content: pairs.append(("Achievements", content[0]))

    print(f"DEBUG: Resume extraction completed, found {len(pairs)} fields")
    return pairs

if __name__ == "__main__":
    pass
