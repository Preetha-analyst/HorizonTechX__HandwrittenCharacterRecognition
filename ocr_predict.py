import os
import cv2
import numpy as np
import tensorflow as tf

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "advanced_emnist_model.h5"
IMAGE_PATH = "word_input.jpeg"
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# EMNIST Character Mapping
# ==========================================================

EMNIST_MAPPING = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# ==========================================================
# Load Model
# ==========================================================

print("=" * 70)
print("ADVANCED HANDWRITTEN OCR SYSTEM")
print("=" * 70)

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ==========================================================
# Read Image
# ==========================================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Cannot open image : {path}")

    return image

# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10
    )

    kernel = np.ones((3,3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    cv2.imwrite(
        os.path.join(OUTPUT_FOLDER, "01_threshold.png"),
        thresh
    )

    return thresh

# ==========================================================
# Remove Small Noise
# ==========================================================

def remove_noise(binary):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    clean = np.zeros(binary.shape, dtype=np.uint8)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if 80 < area < 12000:

            clean[labels == i] = 255

    cv2.imwrite(
        os.path.join(OUTPUT_FOLDER, "02_clean.png"),
        clean
    )

    return clean

# ==========================================================
# Find Text Region
# ==========================================================

def find_text_region(binary):

    coords = cv2.findNonZero(binary)

    if coords is None:
        return binary

    x, y, w, h = cv2.boundingRect(coords)

    margin = 20

    x = max(0, x - margin)
    y = max(0, y - margin)

    w = min(binary.shape[1] - x, w + 2 * margin)
    h = min(binary.shape[0] - y, h + 2 * margin)

    cropped = binary[y:y+h, x:x+w]

    cv2.imwrite(
        os.path.join(OUTPUT_FOLDER, "03_cropped.png"),
        cropped
    )

    return cropped
# ==========================================================
# Character Segmentation
# ==========================================================

def segment_characters(binary):

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        area = w * h

        # Remove tiny noise
        if area < 150:
            continue

        # Ignore extremely thin objects
        if w < 5 or h < 12:
            continue

        # Ignore huge objects
        if w > 250 or h > 250:
            continue

        boxes.append((x, y, w, h))

    # Left to Right
    boxes = sorted(boxes, key=lambda b: b[0])

    print(f"\nCharacters Detected : {len(boxes)}")

    return boxes


# ==========================================================
# Prepare Character for CNN
# ==========================================================

def prepare_character(binary, box):

    x, y, w, h = box

    roi = binary[y:y+h, x:x+w]

    pad = 10

    size = max(w, h) + pad * 2

    square = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    square[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = roi

    roi = cv2.resize(square, (28, 28))

    # EMNIST Orientation Fix
    roi = np.transpose(roi)
    roi = np.fliplr(roi)

    roi = roi.astype(np.float32) / 255.0

    roi = roi.reshape(1, 28, 28, 1)

    return roi


# ==========================================================
# Predict Character
# ==========================================================

def predict_character(roi):

    prediction = model.predict(roi, verbose=0)

    index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    character = EMNIST_MAPPING[index]

    return character, confidence


# ==========================================================
# Draw Bounding Boxes
# ==========================================================

def draw_boxes(image, boxes, predictions):

    result = image.copy()

    for (box, pred) in zip(boxes, predictions):

        x, y, w, h = box

        char, conf = pred

        cv2.rectangle(
            result,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

        cv2.putText(
            result,
            f"{char}",
            (x, y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            "04_segmented.png"
        ),
        result
    )

    return result
# ==========================================================
# OCR Pipeline
# ==========================================================

def run_ocr(image_path):

    # Load Original Image
    original = load_image(image_path)

    # Preprocessing
    binary = preprocess_image(original)

    # Remove Noise
    clean = remove_noise(binary)

    # Crop only text region
    cropped = find_text_region(clean)

    # Find contours again after crop
    contours, _ = cv2.findContours(
        cropped,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        area = w * h

        if area < 150:
            continue

        if w < 5 or h < 12:
            continue

        if w > 250 or h > 250:
            continue

        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    print("\nCharacters Found :", len(boxes))

    predictions = []

    final_text = ""

    display = cv2.cvtColor(cropped.copy(), cv2.COLOR_GRAY2BGR)

    for box in boxes:

        roi = prepare_character(cropped, box)

        character, confidence = predict_character(roi)

        predictions.append((character, confidence))

        x, y, w, h = box

        if confidence >= 0.60:
            final_text += character
        else:
            final_text += "?"

        cv2.rectangle(
            display,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

        cv2.putText(
            display,
            f"{character} {confidence:.2f}",
            (x, y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,255),
            1
        )

        print(
            f"{character}   Confidence : {confidence:.4f}"
        )
    draw_boxes(display, boxes, predictions)

    print("\n" + "="*60)
    print("Recognized Text")
    print("="*60)
    print(final_text)
    print("="*60)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "05_final_output.png"
    )

    cv2.imwrite(output_path, display)

    print(f"\nSaved : {output_path}")

    enlarged = cv2.resize(
        display,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imshow("OCR Result", enlarged)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    run_ocr(IMAGE_PATH)