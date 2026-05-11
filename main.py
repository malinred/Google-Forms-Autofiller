import sys
import math
import fitz
from PIL import Image
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import numpy as np
import cv2

from model_test_inference import (
    preprocess_document, 
    get_ocr_words_and_boxes,
    extract_table_data,
    classify_document,
    extract_aadhaar_data,
    extract_resume_data
)

# Model config
MODEL_PATH = "./model_info/checkpoint-800"
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH)
id2label = model.config.id2label

# Merge entities
def get_entities(tokens, predictions, encoding):
    entities = []
    current_entity = None
    bboxes = encoding['bbox'][0].tolist()
    
    for i, (token, pred) in enumerate(zip(tokens, predictions)):
        label = id2label[pred]
        if token in ["<s>", "</s>", "<pad>"]:
            continue
        
        clean_token = token.replace("Ġ", " ").strip()
        if not clean_token and not token.startswith("Ġ"):
             continue
            
        box = bboxes[i]
        
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text": clean_token,
                "label": label[2:],
                "boxes": [box]
            }
        elif label.startswith("I-") and current_entity and label[2:] == current_entity["label"]:
            if token.startswith("Ġ"):
                current_entity["text"] += " " + clean_token
            else:
                current_entity["text"] += clean_token
            current_entity["boxes"].append(box)
        else:
            if current_entity:
                entities.append(current_entity)
            current_entity = None
            
    if current_entity:
        entities.append(current_entity)

    for ent in entities:
        xs = [b[0] for b in ent["boxes"]] + [b[2] for b in ent["boxes"]]
        ys = [b[1] for b in ent["boxes"]] + [b[3] for b in ent["boxes"]]
        ent["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
        ent["center"] = [(ent["bbox"][0] + ent["bbox"][2]) / 2, 
                        (ent["bbox"][1] + ent["bbox"][3]) / 2]
        ent["text"] = " ".join(ent["text"].split())

    if not entities:
        return []
        
    merged = []
    curr = entities[0]
    for next_ent in entities[1:]:
        same_label = curr["label"] == next_ent["label"]
        same_line = abs(curr["center"][1] - next_ent["center"][1]) < 15
        close_horiz = next_ent["bbox"][0] - curr["bbox"][2] < 50
        
        if same_label and same_line and close_horiz:
            curr["text"] += " " + next_ent["text"]
            curr["boxes"].extend(next_ent["boxes"])
            xs = [curr["bbox"][0], curr["bbox"][2], next_ent["bbox"][0], next_ent["bbox"][2]]
            ys = [curr["bbox"][1], curr["bbox"][3], next_ent["bbox"][1], next_ent["bbox"][3]]
            curr["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
            curr["center"] = [(curr["bbox"][0] + curr["bbox"][2]) / 2, 
                             (curr["bbox"][1] + curr["bbox"][3]) / 2]
        else:
            merged.append(curr)
            curr = next_ent
    merged.append(curr)
        
    return merged

# Pair Q&A
def pair_entities(entities):
    questions = [e for e in entities if e["label"].upper() == "QUESTION"]
    answers = [e for e in entities if e["label"].upper() == "ANSWER"]
    q_to_a = {}
    
    for ans in answers:
        best_q = None
        min_score = float('inf')
        ax, ay = ans["center"]
        
        for q in questions:
            qx, qy = q["center"]
            dist = math.sqrt((ax - qx)**2 + (ay - qy)**2)
            score = dist
            
            if ay < q["bbox"][1] - 5:
                score *= 20.0
            if ax > q["bbox"][0] and abs(ay - qy) < 20:
                score *= 0.3
            elif ay > q["bbox"][3] and abs(ax - qx) < 100:
                score *= 0.5
                
            if score < min_score:
                min_score = score
                best_q = q
        
        if best_q:
            q_text = best_q["text"]
            if q_text not in q_to_a:
                q_to_a[q_text] = []
            q_to_a[q_text].append(ans["text"])
            
    results = []
    for q, a_list in q_to_a.items():
        results.append((q, ", ".join(a_list)))
            
    return results

# Run inference
def run_inference(image, words, boxes):
    if not words:
        return []

    encoding = processor(
        image,
        text=words,
        boxes=boxes,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze(0).tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(
        encoding["input_ids"].squeeze(0).tolist()
    )
    
    entities = get_entities(tokens, predictions, encoding)
    pairs = pair_entities(entities)
    return pairs

def main():
    if len(sys.argv) < 2:
        return

    file_path = sys.argv[1]
    
    if file_path.lower().endswith(".pdf"):
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
                img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                processed_img_bgr = preprocess_document(img_pil)
                words, boxes = get_ocr_words_and_boxes(processed_img_bgr)
                page_img = Image.fromarray(cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB))
            
            doc_type = classify_document(words)
            
            if doc_type == "aadhaar":
                pairs = extract_aadhaar_data(page_img)
                if not pairs:
                    pairs = run_inference(page_img, words, boxes)
            elif doc_type == "resume":
                native_text = page.get_text("text").strip()
                pairs = extract_resume_data(raw_text=native_text, words=words)
            else:
                pairs = run_inference(page_img, words, boxes)
            
            img_for_table = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
            tables = extract_table_data(img_for_table, words, boxes)
            
            if pairs:
                for q, a in pairs:
                    print(f"{q}: {a}")

            if tables:
                for table in tables:
                    print(f"Table HTML: {table['html'][:100]}...")
    else:
        processed_img_bgr = preprocess_document(file_path)
        words, boxes = get_ocr_words_and_boxes(processed_img_bgr)
        page_img = Image.fromarray(cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB))
        doc_type = classify_document(words)
        
        if doc_type == "aadhaar":
            pairs = extract_aadhaar_data(page_img)
            if not pairs:
                pairs = run_inference(page_img, words, boxes)
        elif doc_type == "resume":
            pairs = extract_resume_data(raw_text=None, words=words)
        else:
            pairs = run_inference(page_img, words, boxes)
        
        if pairs:
            for q, a in pairs:
                print(f"{q}: {a}")
        
        tables = extract_table_data(processed_img_bgr, words, boxes)
        if tables:
            for table in tables:
                print(f"Table HTML: {table['html'][:100]}...")

if __name__ == "__main__":
    main()
