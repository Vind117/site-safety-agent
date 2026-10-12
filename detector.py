import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import requests
import base64
import json
import os
from datetime import datetime

# Initialize YOLO model
model = YOLO("yolov8n.pt")

video_path = "sample_site.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create incident storage directory
os.makedirs("incidents", exist_ok=True)
LOG_FILE = "incidents_log.json"

# Define danger zone polygon
polygon_zone_coords = np.array([
    [int(width * 0.3), int(height * 0.3)],
    [int(width * 0.7), int(height * 0.3)],
    [int(width * 0.8), int(height * 0.8)],
    [int(width * 0.2), int(height * 0.8)]
])

zone = sv.PolygonZone(polygon=polygon_zone_coords)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2)

# Initialize ByteTrack for persistent worker IDs
tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def log_incident_to_json(data):
    """Appends safety incident data to a local JSON audit log."""
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

def ask_qwen_safety_agent(frame, tracker_id):
    print(f"\n🚨 Safety Zone Breached by Worker #{tracker_id}! Sending snapshot to Qwen2.5-VL...")
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_filename = f"incidents/violation_worker_{tracker_id}_{timestamp_str}.jpg"
    
    # Save snapshot image to disk
    cv2.imwrite(snapshot_filename, frame)

    _, buffer = cv2.imencode('.jpg', frame)
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
        
        print(f"\n🤖 Qwen Safety Assessment:\n{assessment}\n" + "-"*50)

        # Save structured log entry
        incident_record = {
            "timestamp": datetime.now().isoformat(),
            "worker_id": int(tracker_id),
            "snapshot": snapshot_filename,
            "assessment": assessment
        }
        log_incident_to_json(incident_record)

    except Exception as e:
        print(f"❌ Error connecting to Qwen via Ollama: {e}")

print("Starting Advanced Site Safety Agent with ByteTrack & Auto-Logging... Press 'q' to exit.")

alert_cooldown = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Filter for persons (class_id 0) and pass through ByteTrack
    person_detections = detections[detections.class_id == 0]
    tracked_detections = tracker.update_with_detections(person_detections)

    # Check zone triggers
    is_in_zone = zone.trigger(tracked_detections)

    annotated_frame = zone_annotator.annotate(scene=frame.copy())
    
    # Generate labels containing Tracker IDs
    labels = [
        f"Worker #{tracker_id}" if tracker_id is not None else "Worker"
        for tracker_id in tracked_detections.tracker_id
    ]

    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

    # Trigger logic per unique worker ID
    for i, in_zone in enumerate(is_in_zone):
        if in_zone:
            t_id = tracked_detections.tracker_id[i]
            if t_id is not None:
                if t_id not in alert_cooldown or alert_cooldown[t_id] == 0:
                    ask_qwen_safety_agent(frame, t_id)
                    alert_cooldown[t_id] = 120  # Cooldown frames per specific worker

    # Decrease cooldown counters
    for t_id in list(alert_cooldown.keys()):
        if alert_cooldown[t_id] > 0:
            alert_cooldown[t_id] -= 1

    cv2.imshow("Multimodal Site Safety Incident Reasoner", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()