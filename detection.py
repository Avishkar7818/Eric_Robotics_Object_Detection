
import cv2
import time
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# PLACEHOLDER values: calibrate your actual camera.
FOCAL_LENGTH_PX = 700.0

# Approximate physical heights in metres.
# Replace with measurements of your actual target objects.
OBJECT_HEIGHTS = {
    "cone": 0.50,
    "barrier": 0.80,
    "stop_sign": 0.75,
    "stop-sign": 0.75,
}

CONFIDENCE = 0.25

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(
        frame,
        imgsz=416,
        conf=CONFIDENCE,
        device="cpu",
        verbose=False
    )

    output = frame.copy()

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        normalized_name = class_name.lower().replace(" ", "_")

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        box_height_px = y2 - y1

        if box_height_px <= 0:
            continue

        real_height_m = OBJECT_HEIGHTS.get(normalized_name)

        if real_height_m is not None:
            distance_m = (
                FOCAL_LENGTH_PX * real_height_m / box_height_px
            )
            label = f"{class_name}, {distance_m:.1f}m"
        else:
            label = f"{class_name}, {confidence:.2f}"

        cv2.rectangle(
            output, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    cv2.imshow("Object Detection + Distance Estimation", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
