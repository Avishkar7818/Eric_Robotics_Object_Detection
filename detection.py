
import cv2
import time
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Open the laptop's default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Close other camera apps or try camera index 1.")
    raise SystemExit

# Request a smaller frame size for CPU performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

confidence_threshold = 0.25
previous_time = time.perf_counter()

print("Live detection started. Press Q to quit.")
print("Model classes:", model.names)

while True:
    success, frame = cap.read()

    if not success:
        print("Error: Could not read webcam frame.")
        break

    # Run YOLO inference on the current frame
    results = model.predict(
        source=frame,
        imgsz=416,
        conf=confidence_threshold,
        device="cpu",
        verbose=False
    )

    # Draw bounding boxes and class labels
    output = results[0].plot()

    # Calculate approximate displayed FPS
    current_time = time.perf_counter()
    elapsed = current_time - previous_time
    previous_time = current_time
    fps = 1.0 / max(elapsed, 1e-6)

    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Show the live output window
    cv2.imshow("ERIC Robotics - Live Object Detection", output)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
