from ultralytics import YOLO
import cv2

class StreamProcessor:
    def __init__(self, config):
        print(f"[INFO] Loading YOLO model: {config.model.weights}")
        self.config = config
        self.model = YOLO(config.model.weights)
        self.model.to(config.model.device)
        
    def process_stream(self):
        source = self.config.pipeline.source
        # Handle webcam (numeric) vs video file (string)
        source = int(source) if source.isdigit() else source
        
        # The .track() method handles both detection AND tracking
        results = self.model.track(
            source=source,
            conf=self.config.model.conf_threshold,
            iou=self.config.model.iou_threshold,
            tracker=self.config.pipeline.tracker_type,
            stream=True,  # This makes it a generator for real-time speed
            show=False,   # We will handle the display ourselves in main.py
            persist=self.config.pipeline.enable_tracking
        )
        
        for r in results:
            yield r