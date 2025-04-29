import cv2
import pytesseract
import numpy as np
import os

# Configure the Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'D:\Users\Admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def detect_and_recognize_text(image_path):
    if not os.path.exists(image_path):
        print("❌ Image file not found!")
        return

    image = cv2.imread(image_path)
    orig = image.copy()

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    cv2.imshow("Threshold", thresh)

    # Dilate to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out very small detections
        if w < 10 or h < 10:
            continue

        print(f"Found region: x={x}, y={y}, w={w}, h={h}")

        # Draw rectangles
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Region of Interest (ROI) for OCR
        roi = gray[y:y+h, x:x+w]

        # OCR on each detected region
        text = pytesseract.image_to_string(roi, config='--psm 11')
        cleaned = text.strip()
        if cleaned:
            print(f"🔍 Detected Text: {cleaned}")

    # Full image OCR
    print("\n📄 Full Image OCR Result:")
    full_text = pytesseract.image_to_string(thresh, config='--psm 6').strip()
    print(full_text)

    # Show final image
    cv2.imshow("Text Detection", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main block to run
if __name__ == "__main__":
    image_file = r"C:\Users\Admin\Pictures\Screenshots\Screenshot 2025-03-14 204105.png"
    detect_and_recognize_text(image_file)
