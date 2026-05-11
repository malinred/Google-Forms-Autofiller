import sys
import cv2
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode
import zxingcpp
import fitz  # PyMuPDF

try:
    from pyaadhaar.decode import AadhaarSecureQr, AadhaarOldQr
    from pyaadhaar.utils import isSecureQr
    PYAADHAAR_AVAILABLE = True
except ImportError:
    PYAADHAAR_AVAILABLE = False
    print("[WARN] pyaadhaar not installed — raw QR bytes will be printed instead.")


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def decode_with_zxing(image_cv):
    """
    Try zxing-cpp on a cv2 image.
    Returns the longest result string if any result is > 100 chars, else None.
    """
    # zxing works on grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if len(image_cv.shape) == 3 else image_cv
    results = zxingcpp.read_barcodes(gray)
    if not results:
        return None
    best = max(results, key=lambda r: len(r.text))
    return best.text if len(best.text) > 100 else None


def decode_with_pyzbar(image_cv):
    """
    Try pyzbar on a cv2 image through multiple pre-processing variants.
    Returns the longest result string if any result is > 100 chars, else None.
    """
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if len(image_cv.shape) == 3 else image_cv

    variants = [image_cv, gray]

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    variants.append(adaptive)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    _, sharp_otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(sharp_otsu)

    for img in variants:
        results = pyzbar_decode(img)
        if results:
            best = max(results, key=lambda r: len(r.data))
            decoded = best.data.decode("utf-8", errors="ignore")
            if len(decoded) > 100:
                return decoded

    return None


def try_decode(image_cv, label=""):
    """
    Try zxing first, fall back to pyzbar.
    Returns decoded string or None.
    """
    result = decode_with_zxing(image_cv)
    if result:
        print(f"  [OK] zxing decoded{' (' + label + ')' if label else ''} — {len(result)} chars")
        return result

    result = decode_with_pyzbar(image_cv)
    if result:
        print(f"  [OK] pyzbar decoded{' (' + label + ')' if label else ''} — {len(result)} chars")
        return result

    return None


# ---------------------------------------------------------------------------
# QR region detection + perspective correction (Option 2)
# ---------------------------------------------------------------------------

def detect_and_correct_qr(image_cv):
    """
    Use OpenCV's QRCodeDetector to locate the QR finder pattern and
    perspective-warp it into a clean square before decoding.
    Returns a corrected crop (BGR) or None if not found.
    """
    detector = cv2.QRCodeDetectorAruco() if hasattr(cv2, "QRCodeDetectorAruco") else cv2.QRCodeDetector()

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # detectMulti finds all QR codes; we want the one with the largest area
    try:
        found, points = detector.detectMulti(gray)
    except Exception:
        found = False
        points = None

    if not found or points is None:
        return None

    best_crop = None
    best_area = 0

    for qr_points in points:
        pts = qr_points.reshape(4, 2).astype(np.float32)
        area = cv2.contourArea(pts)
        if area <= best_area:
            continue
        best_area = area

        # Perspective warp to a 600x600 square
        side = 600
        dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image_cv, M, (side, side))
        best_crop = warped

    return best_crop


# ---------------------------------------------------------------------------
# PDF rasterisation
# ---------------------------------------------------------------------------

def extract_pages_from_pdf(pdf_path, dpi=300):
    """Rasterize every PDF page at the given DPI. Returns list of (page_num, bgr_image)."""
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        pages.append((i + 1, img_cv))
    return pages


# ---------------------------------------------------------------------------
# Main scanning strategy
# ---------------------------------------------------------------------------

def scan_qr_from_image(image_cv, page_label=""):
    """
    Full scanning strategy on one cv2 image.
    1. Perspective-correct via OpenCV detector → decode
    2. Full image → zxing + pyzbar
    3. Bottom-right crops → zxing + pyzbar (no upscaling — too slow)
    Returns raw QR string or None.
    """
    h, w = image_cv.shape[:2]
    prefix = f"[Page {page_label}] " if page_label else ""

    # Strategy 1: detect QR location, perspective-correct, then decode
    print(f"{prefix}[*] Attempting QR detection + perspective correction ...")
    corrected = detect_and_correct_qr(image_cv)
    if corrected is not None:
        raw = try_decode(corrected, "perspective-corrected")
        if raw:
            return raw
        print(f"      Detection found QR region but decode failed — continuing ...")
    else:
        print(f"      No QR region detected by OpenCV.")

    # Strategy 2: full image
    print(f"{prefix}[*] Trying full image ...")
    raw = try_decode(image_cv, "full image")
    if raw:
        return raw

    # Strategy 3: bottom-right crops (no upscaling to keep it fast)
    for fraction in [0.35, 0.45, 0.55, 0.65]:
        cy = int(h * (1 - fraction))
        cx = int(w * (1 - fraction))
        crop = image_cv[cy:, cx:]
        print(f"{prefix}[*] Trying bottom-right crop {int(fraction * 100)}% ...")
        raw = try_decode(crop, f"crop {int(fraction * 100)}%")
        if raw:
            return raw

    return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_aadhaar_data(raw_data):
    if not PYAADHAAR_AVAILABLE:
        print("\n[RAW DATA]\n", raw_data[:500], "..." if len(raw_data) > 500 else "")
        return

    try:
        secure = isSecureQr(raw_data)
        print(f"[QR TYPE] {'Secure (XML compressed)' if secure else 'Old (plain XML)'}\n")

        obj  = AadhaarSecureQr(raw_data) if secure else AadhaarOldQr(raw_data)
        data = obj.decodeddata()

        SKIP = {"image", "signature", "adhaar_last_4_digit"}
        print("=" * 40)
        print("  AADHAAR DATA")
        print("=" * 40)
        for key, value in data.items():
            if key not in SKIP:
                label = str(key).replace("_", " ").title()
                print(f"  {label:<22}: {value}")
        print("=" * 40)

        if secure and hasattr(obj, "isImage") and obj.isImage():
            try:
                obj.saveimage("aadhaar_photo.jpg")
                print("[INFO] Embedded photo saved as aadhaar_photo.jpg")
            except Exception as e:
                print(f"[WARN] Could not save photo: {e}")

    except Exception as e:
        print(f"[ERROR] pyaadhaar parsing failed: {e}")
        print("[RAW DATA]\n", raw_data[:500])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_aadhaar_qr.py <file.pdf or file.png/jpg>")
        sys.exit(1)

    path = sys.argv[1]
    raw_data = None

    if path.lower().endswith(".pdf"):
        for dpi in [300, 600]:
            print(f"\n[*] Rasterizing PDF at {dpi} DPI ...")
            pages = extract_pages_from_pdf(path, dpi=dpi)
            for page_num, image_cv in pages:
                raw_data = scan_qr_from_image(image_cv, page_label=str(page_num))
                if raw_data:
                    break
            if raw_data:
                break
    else:
        image_pil = Image.open(path).convert("RGB")
        image_cv  = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        raw_data  = scan_qr_from_image(image_cv)

    if not raw_data:
        print("\n[FAIL] QR code could not be read from this file.")
        sys.exit(1)

    print(f"\n[RAW QR LENGTH] {len(raw_data)} chars")
    print_aadhaar_data(raw_data)


if __name__ == "__main__":
    main()
