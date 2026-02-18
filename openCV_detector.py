import cv2
import numpy as np

def detect(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, f"{w}x{h}", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(img, f"Center:({cx},{cy})",
                        (x,y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    cv2.imshow("Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect("sample.jpg")
