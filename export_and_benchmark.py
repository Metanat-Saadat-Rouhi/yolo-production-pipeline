import time
import os
import torch
from ultralytics import YOLO

def benchmark(model_path, name):
    model = YOLO(model_path)
    # Warmup
    model.predict("https://ultralytics.com/images/bus.jpg", verbose=False)
    
    start_time = time.time()
    for _ in range(100):
        model.predict("https://ultralytics.com/images/bus.jpg", verbose=False)
    avg_time = (time.time() - start_time) / 100 * 1000 # ms
    
    size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"RESULT | {name:10} | Latency: {avg_time:.2f}ms | Size: {size:.2f}MB")

# 1. Load Base Model
model = YOLO("yolov8n.pt")

# 2. Export to different formats
print("[INFO] Exporting to ONNX...")
model.export(format="onnx")

print("[INFO] Exporting to OpenVINO (CPU Optimized)...")
model.export(format="openvino")

# 3. Benchmark results
print("\n--- PERFORMANCE BENCHMARK ---")
benchmark("yolov8n.pt", "PyTorch")
benchmark("yolov8n.onnx", "ONNX")
benchmark("yolov8n_openvino_model", "OpenVINO")