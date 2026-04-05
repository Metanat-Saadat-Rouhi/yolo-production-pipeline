import cv2
import os
from src.config import load_config
from src.detector import StreamProcessor

def main():
    # 1. Load config
    config = load_config("configs/default.yaml")
    processor = StreamProcessor(config)
    
    # Create output directory if it doesn't exist
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    print(f"[INFO] Processing: {config.pipeline.source}")
    
    # Initialize Video Writer variables
    writer = None
    
    try:
        for result in processor.process_stream():
            # Get the annotated frame (numpy array)
            frame = result.plot()

            # --- FIX 1: RESIZE FOR DISPLAY ---
            # Resize so it fits on your screen (e.g., 1280 width)
            display_frame = cv2.resize(frame, (1280, 720)) 
            cv2.imshow("YOLO Real-Time Detection", display_frame)

            # --- FIX 2: SAVE THE VIDEO ---
            if writer is None:
                # Use the ORIGINAL frame size for saving so we don't lose quality
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
                writer = cv2.VideoWriter("outputs/output_demo.mp4", fourcc, 30, (w, h))
                print(f"[INFO] Saving output to: outputs/output_demo.mp4")

            writer.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Video saved and cleanup complete.")

if __name__ == "__main__":
    main()