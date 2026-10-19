import sys
from ultralytics import YOLO

img, known_dist_m, real_h_m = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
model = YOLO("weights/best.pt")
r = model.predict(img, verbose=False)[0]
b = max(r.boxes, key=lambda b: float(b.conf[0]))        # most confident box
x1, y1, x2, y2 = b.xyxy[0].tolist()
h = y2 - y1
print(f"box height = {h:.1f}px  ->  focal_px = {h * known_dist_m / real_h_m:.1f}")