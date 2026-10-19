
import cv2
from ultralytics import YOLO

# Load your trained object detection model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    raise SystemExit

# Camera calibration value (example only)
FOCAL_LENGTH_PX = 700.0

# Approximate real-world object heights in metres.
# Adjust these to the actual object being detected.
OBJECT_HEIGHTS = {
    "cone": 0.50,
    "barrier": 0.80,
    "stop_sign": 0.75,
    "stop-sign": 0.75,
}

# Lower confidence threshold if detections are being missed
CONFIDENCE = 0.25

print("Distance measurement started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam.")
        break

    results = model.predict(
        source=frame,
        imgsz=416,
        conf=CONFIDENCE,
        device="cpu",
        verbose=False
    )

    output = frame.copy()

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        normalized_name = class_name.lower().replace(" ", "_")

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        box_height_px = y2 - y1

        if box_height_px <= 0:
            continue

        real_height_m = OBJECT_HEIGHTS.get(normalized_name)

        # Estimate distance only when object height is known
        if real_height_m is not None:
            distance_m = (
                FOCAL_LENGTH_PX * real_height_m / box_height_px
            )
            label = f"{class_name}: {distance_m:.2f} m"
        else:
            label = f"{class_name}: distance unavailable"

        # Draw bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw class and estimated distance
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    cv2.imshow("Distance Measurement", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
