import os
import paddle
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch
import numpy as np
import cv2
from paddleocr import PaddleOCR

# -----------------------------------------------------------------------
# PaddleOCR Singleton Setup (Step 9)
# -----------------------------------------------------------------------

# Initialize PaddleOCR singleton using the stable 2.8.1 engine
# Use enable_mkldnn=False to avoid the specific CPU instruction bug
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False, enable_mkldnn=False)

# -----------------------------------------------------------------------
# Preprocessing Pipeline (Steps 1-8)
# -----------------------------------------------------------------------

def preprocess_document(image_input):
    """
    Revised Pipeline:
    1. Upscale (DPI Booster)
    2. CLAHE (Illumination)
    3. Noise Removal
    *Skip Hard Binarization* - Let PaddleOCR handle the colors.
    """
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

    # Step 1: DPI Booster (Resolution Normalization)
    min_long_edge = 1200
    if max(h, w) < min_long_edge:
        scale = min_long_edge / max(h, w)
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Step 2: Illumination Correction (CLAHE) in LAB space to preserve color
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Step 3: Denoising (Soft)
    final_img = cv2.fastNlMeansDenoisingColored(final_img, None, 10, 10, 7, 21)
    
    return final_img

def get_ocr_words_and_boxes(image_cv):
    """
    Step 9: PaddleOCR Inference (0.60 threshold)
    """
    words = []
    boxes = []
    h, w = image_cv.shape[:2]

    try:
        results = ocr_engine.ocr(image_cv, cls=True)
        if results and results[0]:
            for line in results[0]:
                poly = line[0]
                text, conf = line[1]
                # Lower threshold to 0.60 for real-world phone photos
                if conf >= 0.60:
                    words.append(text)
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    # Normalize and CLIP for LayoutLMv3
                    box = [
                        max(0, min(1000, int(1000 * min(xs) / w))),
                        max(0, min(1000, int(1000 * min(ys) / h))),
                        max(0, min(1000, int(1000 * max(xs) / w))),
                        max(0, min(1000, int(1000 * max(ys) / h)))
                    ]
                    boxes.append(box)
    except Exception as e:
        print(f"[OCR] PaddleOCR runtime error: {e}")

    return words, boxes

if __name__ == "__main__":
    # Test script
    SAMPLE_IMAGE = "sample_form.jpg"
    if os.path.exists(SAMPLE_IMAGE):
        processed = preprocess_document(SAMPLE_IMAGE)
        words, boxes = get_ocr_words_and_boxes(processed)
        print(f"\nExtracted {len(words)} words using PaddleOCR 2.8.1")
        for w, b in zip(words[:10], boxes[:10]):
            print(f"  Word: {w} | Box: {b}")
    else:
        print(f"Sample image {SAMPLE_IMAGE} not found for testing.")
