# 🛡️ Multimodal Site Safety Incident Reasoner

An AI-powered, local-first industrial safety compliance monitoring system that combines real-time Computer Vision (`YOLOv8` + `ByteTrack`), spatial zone triggering (`Supervision`), and Multimodal Vision-Language Models (`Qwen2.5-VL` via Ollama) with an interactive `Streamlit` web dashboard.

---

## 🚀 Key Features
- **Real-Time Object Detection & Tracking:** Leverages `YOLOv8` for fast person detection and `ByteTrack` to assign persistent worker IDs (`Worker #1`, `Worker #2`, etc.) across video frames.
- **Dynamic Polygon Zoning:** Uses `Supervision` to define restricted high-risk construction zones and detect breaches instantly.
- **Asynchronous AI Safety Reasoning:** When a worker enters a danger zone, a background thread captures a snapshot and queries `Qwen2.5-VL` locally to analyze personal protective equipment (PPE) compliance and assess safety risks without freezing the UI.
- **Automated Audit Logging:** Instantly logs violation timestamps, worker IDs, image snapshots, and AI assessments into a structured JSON audit trail.
- **Interactive Streamlit UI:** A clean, multi-tab web dashboard for video uploads, live streaming, and compliance review.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Computer Vision & Tracking:** OpenCV, Ultralytics YOLOv8, Supervision (ByteTrack & PolygonZone)
- **Multimodal AI:** Ollama (`Qwen2.5-VL`)
- **Frontend / UI:** Streamlit

---

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/vind_117/site-safety-agent.git](https://github.com/vind_117/site-safety-agent.git)
   cd site-safety-agent