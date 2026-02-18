# openCV_detector.py
# YOLO-based object detection with OpenCV visualization
from google.colab.patches import cv2_imshow
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ======================
# CONFIG
# ======================
MODEL_PATH = "/content/best-2.pt"
INPUT_IMAGE = "/content/Warehouse/test/images/72_JPG.rf.5489e6f989b002cc240b9ffafa7c9ef5.jpg"   # change or pass dynamically
OUTPUT_FOLDER = "/content/results/detections"

# ======================
# Load YOLO model
# ======================
model = YOLO(MODEL_PATH)

def detect_and_visualize(image_path):

    image_path = str(image_path)
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found.")
        return

    h_img, w_img = img.shape[:2]

    results = model(image_path)
    result = results[0]

    if len(result.boxes) == 0:
        print("No objects detected.")
        return

    for box in result.boxes:

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        # Width and height
        width = x2 - x1
        height = y2 - y1

        # Center coordinates
        cx = x1 + width // 2
        cy = y1 + height // 2

        confidence = float(box.conf[0].cpu().item())

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display dimensions
        cv2.putText(img,
                    f"W:{width}px H:{height}px",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

        # Display center coordinates
        cv2.putText(img,
                    f"Center: ({cx},{cy})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2)

        # Display confidence
        cv2.putText(img,
                    f"Conf: {confidence:.2f}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2)

        print("\nDetected Object:")
        print(f"Bounding Box: ({x1}, {y1}), ({x2}, {y2})")
        print(f"Width: {width}px")
        print(f"Height: {height}px")
        print(f"Center: ({cx}, {cy})")
        print(f"Confidence: {confidence:.3f}")

    # Create output folder
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    output_path = Path(OUTPUT_FOLDER) / Path(image_path).name
    cv2.imwrite(str(output_path), img)

    print(f"\nSaved detected image to: {output_path}")

    # Show original and detected image
    cv2_imshow(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_and_visualize(INPUT_IMAGE)
