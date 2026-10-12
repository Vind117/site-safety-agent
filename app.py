import streamlit as st
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import requests
import base64
import json
import os
import tempfile
import threading
from datetime import datetime

st.set_page_config(
    page_title="Site Safety Incident Reasoner",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Multimodal Site Safety Incident Reasoner")
st.markdown("AI-powered construction safety monitoring using **YOLOv8**, **ByteTrack**, and **Qwen2.5-VL**.")

# Sidebar setup
st.sidebar.header("Configuration Panel")
uploaded_file = st.sidebar.file_uploader("Upload Construction Video (.mp4)", type=["mp4", "avi", "mov"])

LOG_FILE = "incidents_log.json"
os.makedirs("incidents", exist_ok=True)

@st.cache_resource
def load_detection_model():
    model = YOLO("yolov8n.pt")
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    return model, tracker, box_annotator, label_annotator

model, tracker, box_annotator, label_annotator = load_detection_model()

def log_incident_to_json(data):
    log_data = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            log_data = []
    log_data.append(data)
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=4)

def async_ask_qwen(frame_snapshot, tracker_id):
    """Runs Qwen reasoning in a background thread so the video doesn't freeze."""
    def worker():
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        snapshot_filename = f"incidents/violation_worker_{tracker_id}_{timestamp_str}.jpg"
        cv2.imwrite(snapshot_filename, frame_snapshot)

        _, buffer = cv2.imencode('.jpg', frame_snapshot)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "model": "qwen2.5vl",
            "prompt": f"Analyze this construction site snapshot. Worker #{tracker_id} is in the designated danger zone. Are they wearing proper safety equipment (harness, hard hat)? Describe any safety compliance risk in 2 sentences.",
            "images": [img_base64],
            "stream": False
        }

        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
            result = response.json()
            assessment = result.get('response', 'No response text.')
            
            incident_record = {
                "timestamp": datetime.now().isoformat(),
                "worker_id": int(tracker_id),
                "snapshot": snapshot_filename,
                "assessment": assessment
            }
            log_incident_to_json(incident_record)
        except Exception as e:
            print(f"❌ Error querying Qwen model in background thread: {e}")

    threading.Thread(target=worker, daemon=True).start()

tab1, tab2 = st.tabs(["📹 Live Detection & Stream", "📋 Audit Logs & Incidents"])

with tab1:
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        if st.sidebar.button("Run Safety Analysis"):
            st.info("Running smooth streaming analysis with background AI reasoning...")
            
            video_placeholder = st.empty()
            cap = cv2.VideoCapture(video_path)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            polygon_zone_coords = np.array([
                [int(width * 0.3), int(height * 0.3)],
                [int(width * 0.7), int(height * 0.3)],
                [int(width * 0.8), int(height * 0.8)],
                [int(width * 0.2), int(height * 0.8)]
            ])

            zone = sv.PolygonZone(polygon=polygon_zone_coords)
            zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2)

            alert_cooldown = {}
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # Process every 2nd frame to boost playback smoothness
                if frame_count % 2 != 0:
                    continue

                results = model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)

                person_detections = detections[detections.class_id == 0]
                tracked_detections = tracker.update_with_detections(person_detections)

                is_in_zone = zone.trigger(tracked_detections)

                annotated_frame = zone_annotator.annotate(scene=frame.copy())

                labels = [
                    f"Worker #{tracker_id}" if tracker_id is not None else "Worker"
                    for tracker_id in tracked_detections.tracker_id
                ]

                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

                for i, in_zone in enumerate(is_in_zone):
                    if in_zone:
                        t_id = tracked_detections.tracker_id[i]
                        if t_id is not None:
                            if t_id not in alert_cooldown or alert_cooldown[t_id] == 0:
                                async_ask_qwen(frame, t_id)
                                alert_cooldown[t_id] = 150  # Cooldown frames

                for t_id in list(alert_cooldown.keys()):
                    if alert_cooldown[t_id] > 0:
                        alert_cooldown[t_id] -= 1

                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", width="stretch")

            cap.release()
            st.success("Video safety analysis completed smoothly!")
    else:
        st.info("👈 Please upload a construction video via the sidebar to begin.")

with tab2:
    st.subheader("Automated Compliance Audit Trail")
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
            
            if logs:
                for entry in reversed(logs):
                    with st.expander(f"🚨 Incident Worker #{entry.get('worker_id')} - {entry.get('timestamp')}"):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if os.path.exists(entry.get('snapshot', '')):
                                st.image(entry['snapshot'], caption="Violation Snapshot", width="stretch")
                        with col2:
                            st.write(f"**Worker ID:** #{entry.get('worker_id')}")
                            st.write(f"**Timestamp:** {entry.get('timestamp')}")
                            st.markdown(f"**Qwen Assessment:**\n> {entry.get('assessment')}")
            else:
                st.write("No safety violations recorded yet.")
        except Exception as e:
            st.error(f"Error reading log file: {e}")
    else:
        st.write("No incident logs found yet.")