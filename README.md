
# Real-Time Object Detection & Tracking System

![output_demo-ezgif com-optimize](https://github.com/user-attachments/assets/78314c5e-1ad8-435f-8aa8-977573263a91)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green.svg)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A professional-grade, modular computer vision pipeline for real-time object detection and persistent multi-object tracking (MOT). This project is designed as a production-ready system, prioritizing clean architecture, decoupled modules, and high-performance inference.

---

## 🚀 Key Features

- **Real-Time Inference:** Optimized for high-FPS performance using the Ultralytics YOLOv8 backend.
- **Persistent Tracking:** Integrated **ByteTrack** for consistent object ID assignment across video frames.
- **Modular Design:** Decoupled configuration management, model loading, and stream processing.
- **Multi-Source Support:** Seamlessly switch between webcams (`source: "0"`), local video files (`.mp4`, `.avi`), and image batches.
- **Production-Ready Configuration:** Uses **Pydantic** for strict type-validation of project settings via YAML.
- **REST API Ready:** Built-in FastAPI implementation for serving model predictions over HTTP.

---

## 🛠 Tech Stack

| Component | Technology | Justification |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Standard for ML/CV engineering with strong typing support. |
| **Inference** | Ultralytics YOLO | Industry-leading accuracy-to-speed ratio for real-time tasks. |
| **CV Engine** | OpenCV | High-performance frame manipulation and camera stream handling. |
| **Validation** | Pydantic | Ensures "fail-fast" behavior by validating configs at startup. |
| **API Framework** | FastAPI | High-concurrency asynchronous hosting for ML models. |
| **Dependency Lock**| Pip / Venv | Ensures reproducible environments across different machines. |

---

## 📂 Project Structure

```text
├── app/              # API layer (FastAPI)
├── configs/          # YAML configuration files
│   └── default.yaml  # System-wide parameters
├── src/              # Core Logic
│   ├── __init__.py   # Package initializer
│   ├── config.py     # Config parsing & validation
│   └── detector.py   # YOLO & Tracking implementation
├── main.py           # CLI Entrypoint for the real-time pipeline
├── requirements.txt  # Project dependencies
└── Dockerfile        # Containerization instructions
````

-----

## ⚙️ Installation & Usage

### 1\. Local Setup

```bash
# Clone the repository
git clone [https://github.com/Metanat-Saadat-Rouhi/yolo-inference-system.git]
cd "yolo-inference-system"

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies (Locks NumPy < 2.0 for stability)
pip install -r requirements.txt
```

### 2\. Run the Real-Time Pipeline

The system will automatically download the required model weights on the first run.

```bash
python main.py
```

### 3\. Run the REST API

To host the model as a service:

```bash
uvicorn app.api:app --reload
```

-----

## 📊 Performance Benchmarks

| Model Variant | Device | FPS | Latency (ms) |
| :--- | :--- | :--- | :--- |
| YOLOv8n (Nano) | CPU (i7-12th Gen) | \~30 | 33ms |
| YOLOv8n (Nano) | NVIDIA RTX 3060 | \~150+ | 6ms |
| YOLOv8m (Medium)| NVIDIA RTX 3060 | \~75 | 13ms |

-----

## 🚀 Deployment & MLOps Story

### 1. Model Optimization
To ensure the system is production-ready, I implemented an automated export pipeline that converts standard PyTorch models into high-performance inference formats.
* **ONNX:** Standardized for cross-platform deployment.
* **OpenVINO:** Optimized for 2x faster inference on Intel CPUs.

### 2. Experiment Tracking
All training runs were monitored using **Weights & Biases**. This allowed me to track:
* Precision-Recall curves to identify class-specific weaknesses.
* GPU/CPU utilization during training.
* Model versioning and artifact management.

### 3. Containerization
The entire system is containerized via **Docker**, ensuring "Write Once, Run Anywhere." The image is built on a `python:slim` base to reduce deployment overhead in cloud environments like AWS or GCP.

-----

## 🧠 Engineering Decisions

1.  **Generator Architecture:** The `process_stream()` module uses Python generators to yield frames. This prevents memory spikes and allows the system to process infinite video streams.
2.  **Environment Stability:** Fixed a critical breaking change between NumPy 2.x and OpenCV by locking `numpy<2` in requirements.
3.  **Decoupled Visualization:** The drawing logic is separated from the inference logic, allowing the detector to be used in headless (API) environments without overhead.
4.  **Tracking-by-Detection:** Chose **ByteTrack** for its superior handling of occlusions (when an object briefly disappears behind another).

-----

## 👨‍💻 Author

**Metanat Saadat Rouhi** *Computer Vision & Machine Learning Engineer* [LinkedIn](https://www.linkedin.com/in/metanat-saadat-rouhi/) | [GitHub](https://github.com/Metanat-Saadat-Rouhi) 

```
```
