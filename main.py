import sys
import math
import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import numpy as np
import cv2

from model_test_inference import (
    preprocess_document, 
    get_ocr_words_and_boxes
)

# -----------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------

MODEL_PATH = "./model_info/checkpoint-800"
print(f"[Init] Loading model and processor from {MODEL_PATH}...")
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH)
id2label = model.config.id2label

# -----------------------------------------------------------------------
# Entity Extraction & Pairing
# -----------------------------------------------------------------------

def get_entities(tokens, predictions, encoding):
    """
    Merge tokens into full entities (Question, Answer, etc.) with bounding boxes.
    """
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
        
        # Merge B- and I- tokens
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

    # Post-process: Consolidate boxes and clean text
    for ent in entities:
        xs = [b[0] for b in ent["boxes"]] + [b[2] for b in ent["boxes"]]
        ys = [b[1] for b in ent["boxes"]] + [b[3] for b in ent["boxes"]]
        ent["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
        ent["center"] = [(ent["bbox"][0] + ent["bbox"][2]) / 2, 
                        (ent["bbox"][1] + ent["bbox"][3]) / 2]
        ent["text"] = " ".join(ent["text"].split())

    # HEURISTIC: Merge adjacent entities of the same label on the same line
    if not entities:
        return []
        
    merged = []
    curr = entities[0]
    for next_ent in entities[1:]:
        # If same label, same approximate Y level, and close horizontally
        same_label = curr["label"] == next_ent["label"]
        same_line = abs(curr["center"][1] - next_ent["center"][1]) < 15
        close_horiz = next_ent["bbox"][0] - curr["bbox"][2] < 50
        
        if same_label and same_line and close_horiz:
            curr["text"] += " " + next_ent["text"]
            curr["boxes"].extend(next_ent["boxes"])
            # Update bbox and center
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

def pair_entities(entities):
    """
    Spatial pairing of Questions and Answers.
    """
    questions = [e for e in entities if e["label"].upper() == "QUESTION"]
    answers = [e for e in entities if e["label"].upper() == "ANSWER"]
    
    # Use a dictionary to group answers by question
    q_to_a = {}
    
    for ans in answers:
        best_q = None
        min_score = float('inf')
        ax, ay = ans["center"]
        
        for q in questions:
            qx, qy = q["center"]
            dist = math.sqrt((ax - qx)**2 + (ay - qy)**2)
            score = dist
            
            # Heavy penalty if answer is above the question
            if ay < q["bbox"][1] - 5:
                score *= 20.0
            
            # Favor horizontal alignment (Answer to the right of Question)
            if ax > q["bbox"][0] and abs(ay - qy) < 20:
                score *= 0.3
            # Favor vertical alignment (Answer below Question)
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
            
    # Format as list of strings "Field: Value"
    results = []
    for q, a_list in q_to_a.items():
        results.append((q, ", ".join(a_list)))
            
    return results

# -----------------------------------------------------------------------
# Processing Functions
# -----------------------------------------------------------------------

def run_inference(image, words, boxes):
    """
    Run LayoutLM inference and return paired Q&A.
    """
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

    # Use squeeze(0) to only remove batch dimension, avoids 0-dim tensor issues
    predictions = outputs.logits.argmax(-1).squeeze(0).tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(
        encoding["input_ids"].squeeze(0).tolist()
    )
    
    entities = get_entities(tokens, predictions, encoding)
    pairs = pair_entities(entities)
    return pairs

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <file_path>")
        return

    file_path = sys.argv[1]
    
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        print(f"[PDF] Opened: {file_path} ({len(doc)} pages)")
        
        for page_num in range(len(doc)):
            print(f"\n--- Processing PDF Page {page_num + 1} ---")
            page = doc[page_num]
            
            # Step 0: PDF Page Routing
            native_text = page.get_text("text").strip()
            
            if native_text:
                print(f"[PDF] Page {page_num + 1}: Using native text layer")
                words_data = page.get_text("words")
                words = []
                boxes = []
                p_width, p_height = page.rect.width, page.rect.height
                for wd in words_data:
                    # wd: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
                    words.append(wd[4])
                    # Normalize and CLIP for LayoutLMv3
                    box = [
                        max(0, min(1000, int(1000 * wd[0] / p_width))),
                        max(0, min(1000, int(1000 * wd[1] / p_height))),
                        max(0, min(1000, int(1000 * wd[2] / p_width))),
                        max(0, min(1000, int(1000 * wd[3] / p_height)))
                    ]
                    boxes.append(box)
                
                # Render page for LayoutLMv3 background
                pix = page.get_pixmap(dpi=300)
                page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            else:
                print(f"[PDF] Page {page_num + 1}: No native text found, using OCR pipeline")
                pix = page.get_pixmap(dpi=300)
                img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                processed_img_bgr = preprocess_document(img_pil)
                words, boxes = get_ocr_words_and_boxes(processed_img_bgr)
                # Convert BGR back to RGB PIL for LayoutLMv3
                page_img = Image.fromarray(cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB))
            
            pairs = run_inference(page_img, words, boxes)
            
            if pairs:
                print(f"\nExtracted Fields (Page {page_num + 1}):")
                for q, a in pairs:
                    print(f"  {q}: {a}")
            else:
                print(f"No Q&A pairs detected on Page {page_num + 1}.")
    else:
        # Assume Image input
        print(f"[Image] Processing: {file_path}")
        processed_img_bgr = preprocess_document(file_path)
        words, boxes = get_ocr_words_and_boxes(processed_img_bgr)
        # Convert BGR back to RGB PIL for LayoutLMv3
        page_img = Image.fromarray(cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB))
        pairs = run_inference(page_img, words, boxes)
        
        if pairs:
            print("\nExtracted Fields:")
            for q, a in pairs:
                print(f"  {q}: {a}")
        else:
            print("No Q&A pairs detected.")

if __name__ == "__main__":
    main()
