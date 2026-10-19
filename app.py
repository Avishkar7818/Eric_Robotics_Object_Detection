
import cv2
import av
import time
import threading
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Robotics Object Detection",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Robotics Object Detection & Distance Estimation")
st.caption("YOLO object detection | Live camera | Estimated distance")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load best.pt: {e}")
    st.stop()

# ---------------- SIDEBAR SETTINGS ----------------
st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Detection confidence",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

image_size = st.sidebar.select_slider(
    "Inference image size",
    options=[320, 416, 512, 640],
    value=416
)

focal_length = st.sidebar.number_input(
    "Focal length (pixels)",
    min_value=1.0,
    max_value=5000.0,
    value=700.0,
    step=10.0
)

st.sidebar.warning(
    "Focal length and object heights are example values. "
    "Calibrate your camera for meaningful distance estimates."
)

st.sidebar.subheader("Object heights (metres)")

cone_height = st.sidebar.number_input(
    "Cone height",
    min_value=0.01,
    max_value=5.0,
    value=0.50,
    step=0.05
)

barrier_height = st.sidebar.number_input(
    "Barrier height",
    min_value=0.01,
    max_value=5.0,
    value=0.80,
    step=0.05
)

stop_sign_height = st.sidebar.number_input(
    "Stop sign height",
    min_value=0.01,
    max_value=5.0,
    value=0.75,
    step=0.05
)

# Normalize common class-name variants
OBJECT_HEIGHTS = {
    "cone": cone_height,
    "trafficcone": cone_height,
    "movable_object-trafficcone": cone_height,
    "barrier": barrier_height,
    "movable_object-barrier": barrier_height,
    "stop_sign": stop_sign_height,
    "stop-sign": stop_sign_height,
    "stop sign": stop_sign_height,
}

# ---------------- SHARED METRICS ----------------
if "detections" not in st.session_state:
    st.session_state.detections = []

if "inference_ms" not in st.session_state:
    st.session_state.inference_ms = 0.0

if "fps" not in st.session_state:
    st.session_state.fps = 0.0

metrics_lock = threading.Lock()


# ---------------- VIDEO PROCESSING ----------------
class VideoProcessor:
    def recv(self, frame):
        start = time.perf_counter()

        # streamlit-webrtc supplies an AV frame
        image = frame.to_ndarray(format="bgr24")

        results = model.predict(
            source=image,
            imgsz=image_size,
            conf=confidence,
            device="cpu",
            verbose=False
        )

        result = results[0]
        output = image.copy()
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            score = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            box_height = y2 - y1

            # Convert class names to a consistent format
            normalized_name = class_name.lower().replace(" ", "_")
            real_height = OBJECT_HEIGHTS.get(normalized_name)

            if real_height is not None and box_height > 0:
                distance = focal_length * real_height / box_height
                label = f"{class_name}: {distance:.2f} m"
                detections.append({
                    "Object": class_name,
                    "Confidence": round(score, 3),
                    "Distance (m)": round(distance, 2)
                })
            else:
                distance = None
                label = f"{class_name}: distance unavailable"
                detections.append({
                    "Object": class_name,
                    "Confidence": round(score, 3),
                    "Distance (m)": None
                })

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
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

        elapsed = time.perf_counter() - start
        inference_ms = elapsed * 1000
        fps = 1.0 / max(elapsed, 1e-6)

        with metrics_lock:
            st.session_state.detections = detections
            st.session_state.inference_ms = inference_ms
            st.session_state.fps = fps

        return av.VideoFrame.from_ndarray(
            output, format="bgr24"
        )


# ---------------- MAIN APP ----------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Live Camera")
    st.write(
        "Allow camera access in your browser. "
        "Click START to begin detection."
    )

    ctx = webrtc_streamer(
        key="robotics-distance-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

with right:
    st.subheader("Performance")

    fps_placeholder = st.empty()
    latency_placeholder = st.empty()
    count_placeholder = st.empty()

    st.subheader("Detected Objects")
    table_placeholder = st.empty()

    st.info(
        "Distance estimates depend on correct focal length "
        "and real object dimensions."
    )

# Refresh metrics while video is running
if ctx.state.playing:
    while ctx.state.playing:
        with metrics_lock:
            detections = list(st.session_state.detections)
            inference_ms = st.session_state.inference_ms
            fps = st.session_state.fps

        fps_placeholder.metric("Approx. FPS", f"{fps:.1f}")
        latency_placeholder.metric(
            "Processing time", f"{inference_ms:.1f} ms"
        )
        count_placeholder.metric(
            "Objects detected", len(detections)
        )

        if detections:
            table_placeholder.dataframe(
                detections,
                use_container_width=True,
                hide_index=True
            )
        else:
            table_placeholder.write(
                "No objects detected yet."
            )

        time.sleep(1)
