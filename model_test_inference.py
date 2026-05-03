from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image, ImageEnhance
import pytesseract
import torch
import numpy as np
import cv2


def fix_orientation(img_pil):
    """
    Step 1 — Use Tesseract OSD to detect orientation and rotate accordingly.
    Handles 0, 90, 180, 270 degree rotations.
    """
    try:
        osd = pytesseract.image_to_osd(img_pil, output_type=pytesseract.Output.DICT)
        angle = osd["rotate"]
        print(f"[Orientation] Detected rotation needed: {angle} degrees")
        if angle != 0:
            # PIL rotate is counter-clockwise, so negate the angle
            img_pil = img_pil.rotate(-angle, expand=True)
            print(f"[Orientation] Rotated image by {angle} degrees")
    except Exception as e:
        print(f"[Orientation] OSD failed, skipping rotation: {e}")
    return img_pil


def deskew(img_cv):
    """
    Step 2 — Fix tilts. Now handles 90-degree rotations if OSD fails.
    """
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                             minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        median_angle = np.median(angles)
        print(f"[Deskew] Detected skew angle: {median_angle:.2f} degrees")
        
        # Correct for small tilts
        if 0.5 < abs(median_angle) < 45:
            h, w = img_cv.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            img_cv = cv2.warpAffine(img_cv, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            print(f"[Deskew] Corrected skew by {median_angle:.2f} degrees")
        # Correct for 90-degree rotations (common if OSD fails)
        elif abs(abs(median_angle) - 90) < 5:
            print("[Deskew] Significant 90-degree rotation detected. Correcting...")
            if median_angle < 0:
                img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
            else:
                img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            print("[Deskew] No actionable skew detected, skipping")
    else:
        print("[Deskew] No lines detected, skipping")
    return img_cv


def denoise(img_cv):
    """
    Step 3 — Remove camera noise using non-local means denoising.
    Helps with photos taken in suboptimal lighting.
    """
    print("[Denoise] Applying denoising")
    return cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)


def binarize(img_cv):
    """
    Step 4 — Convert to grayscale and apply Otsu thresholding.
    Produces crisp black text on white background for better OCR.
    Converts back to RGB since LayoutLMv3 expects RGB input.
    """
    print("[Binarize] Applying grayscale + Otsu thresholding")
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Convert back to RGB
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def enhance_contrast(img_pil):
    """
    Step 5 — Boost contrast to make faded or low light text clearer.
    """
    print("[Contrast] Enhancing contrast")
    enhancer = ImageEnhance.Contrast(img_pil)
    return enhancer.enhance(2.0)


def upscale_if_needed(img_pil, min_size=1000):
    """
    Step 6 — Upscale if the image is too small.
    Tesseract struggles with images where text is less than ~20px tall.
    """
    w, h = img_pil.size
    if min(w, h) < min_size:
        scale = min_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"[Upscale] Image too small ({w}x{h}), upscaling to {new_w}x{new_h}")
        img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
    else:
        print(f"[Upscale] Image size OK ({w}x{h}), skipping")
    return img_pil


def preprocess_document(image_input):
    """
    Full preprocessing pipeline:
    1. Fix orientation (OSD)
    2. Deskew
    3. Denoise
    4. Binarize
    5. Enhance contrast
    6. Upscale if needed
    """
    if isinstance(image_input, str):
        print(f"\n--- Preprocessing: {image_input} ---")
        img_pil = Image.open(image_input).convert("RGB")
    else:
        print("\n--- Preprocessing: Image Object ---")
        img_pil = image_input.convert("RGB")

    print(f"[Load] Original size: {img_pil.size}")

    # Step 1 — Fix orientation using Tesseract OSD
    img_pil = fix_orientation(img_pil)

    # Convert to OpenCV format for steps 2-4
    img_cv = np.array(img_pil)

    # Step 2 — Deskew
    img_cv = deskew(img_cv)

    # Step 3 — Denoise
    img_cv = denoise(img_cv)

    # Step 4 — Binarize
    img_cv = binarize(img_cv)

    # Convert back to PIL for steps 5-6
    img_pil = Image.fromarray(img_cv)

    # Step 5 — Enhance contrast
    img_pil = enhance_contrast(img_pil)

    # Step 6 — Upscale if needed
    img_pil = upscale_if_needed(img_pil)

    print(f"[Done] Final size: {img_pil.size}\n")
    return img_pil


def get_ocr_words_and_boxes(image):
    """
    Run Tesseract OCR to get words and bounding boxes normalized to 0-1000.
    """
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Extract valid words and their bounding boxes
    words = []
    boxes = []

    width, height = image.size

    for i, word in enumerate(ocr_data["text"]):
        if word.strip() != "":
            words.append(word)
            x, y, w, h = (ocr_data["left"][i], ocr_data["top"][i],
                          ocr_data["width"][i], ocr_data["height"][i])
            box = [
                int(1000 * x / width),
                int(1000 * y / height),
                int(1000 * (x + w) / width),
                int(1000 * (y + h) / height)
            ]
            boxes.append(box)
    
    return words, boxes


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Main inference pipeline
    # -----------------------------------------------------------------------

    model = LayoutLMv3ForTokenClassification.from_pretrained("./model_info/checkpoint-800")
    processor = LayoutLMv3Processor.from_pretrained("./model_info/checkpoint-800")

    # Preprocess the document
    image = preprocess_document("sample_form.jpg")

    # Run Tesseract OCR to get words and bounding boxes
    words, boxes = get_ocr_words_and_boxes(image)

    # Pass to processor
    encoding = processor(
        image,
        text=words,
        boxes=boxes,
        return_tensors="pt"
    )

    # Run inference
    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(
        encoding["input_ids"].squeeze().tolist()
    )

    # -----------------------------------------------------------------------
    # Post processing — group tokens into key-value pairs
    # -----------------------------------------------------------------------

    id2label = model.config.id2label

    current_word = ""
    current_label = ""
    results = []

    for token, pred in zip(tokens, predictions):
        label = id2label[pred]

        if token in ["<s>", "</s>", "<pad>"]:
            continue

        clean_token = token.replace("Ġ", " ").strip()

        if label.startswith("B-"):
            if current_word:
                results.append((current_word.strip(), current_label))
            current_word = clean_token
            current_label = label[2:]
        elif label.startswith("I-"):
            current_word += clean_token
        else:
            if current_word:
                results.append((current_word.strip(), current_label))
            current_word = ""
            current_label = ""

    # Print raw token labels
    print("--- Raw Token Labels ---")
    for token, pred in zip(tokens, predictions):
        print(f"{token}: {id2label[pred]}")

    # Print grouped key-value results
    print("\n--- Extracted Fields ---")
    for text, label in results:
        print(f"{label}: {text}")
