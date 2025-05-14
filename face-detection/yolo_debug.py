import cv2
import time
import collections
import psutil
import os
from ultralytics import YOLO
import GPUtil  # For GPU usage monitoring

def main():
    # Load the YOLOv11 model weights
    model_path = "models/yolov8n-face.pt"  # Replace with your actual model file
    model = YOLO(model_path)
    # Move the model to GPU
    model.to("cuda")
    
    # In case I want to switch back to CPU
    # model.to("cpu")
    
    # Open the default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    process = psutil.Process(os.getpid())
    logical_cores = psutil.cpu_count(logical=True)
    
    # Initialize buffers for FPS, CPU load, memory usage, and GPU usage
    prev_frame_time = time.time()
    fps_buffer = collections.deque(maxlen=30)
    cpu_buffer = collections.deque(maxlen=30)
    mem_buffer = collections.deque(maxlen=30)
    gpu_buffer = collections.deque(maxlen=30)
    
    frame_count = 0
    avg_gpu = 0.0  # Initial GPU load value

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        frame_count += 1
        
        # -------------------------
        # System Resource Measurements
        # -------------------------
        # 1) CPU Memory (RAM) usage
        mem_usage = process.memory_info().rss / (1024 * 1024)
        mem_buffer.append(mem_usage)
        avg_mem = sum(mem_buffer) / len(mem_buffer)
        
        # 2) FPS calculation
        current_time = time.time()
        fps = 1.0 / (current_time - prev_frame_time)
        prev_frame_time = current_time
        fps_buffer.append(fps)
        avg_fps = sum(fps_buffer) / len(fps_buffer)
        
        # 3) CPU usage
        cpu_usage = process.cpu_percent(interval=None)
        norm_cpu = cpu_usage / logical_cores
        cpu_buffer.append(norm_cpu)
        avg_cpu = sum(cpu_buffer) / len(cpu_buffer)
        
        # 4) GPU usage (via GPUtil) - update every 10 frames
        if frame_count % 10 == 0:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_load = gpu.load * 100.0
                gpu_buffer.append(gpu_load)
                avg_gpu = sum(gpu_buffer) / len(gpu_buffer)
            else:
                avg_gpu = 0.0
        # -------------------------
        # YOLO Inference
        # -------------------------
        results = model(frame)
        
        # Draw bounding boxes on detections
        preds = results[0]  # First batch prediction
        for box in preds.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "{:.2f}".format(score),
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # -------------------------
        # Overlay Metrics on Frame
        # -------------------------
        cv2.putText(
            frame,
            "FPS: {}".format(int(avg_fps)),
            (7, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "CPU: {:.1f}% per core".format(avg_cpu),
            (7, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Mem: {:.2f} MB".format(avg_mem),
            (7, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "GPU: {:.1f}%".format(avg_gpu),
            (7, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Cores: {}".format(logical_cores),
            (7, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        cv2.imshow("YOLOv8-N - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
