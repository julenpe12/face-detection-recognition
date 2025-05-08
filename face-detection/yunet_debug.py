import cv2
import time
import psutil
import collections

def main():
    # Path to YuNet ONNX model
    model_path = "models/face_detection_yunet_2023mar.onnx"
    
    face_detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (0, 0),
        0.5
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize system metrics
    process = psutil.Process()
    logical_cores = psutil.cpu_count(logical=True)
    prev_frame_time = time.time()
    fps_buffer = collections.deque(maxlen=30)
    cpu_buffer = collections.deque(maxlen=30)
    mem_buffer = collections.deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # -------------------------
        # System Metrics Calculations
        # -------------------------
        current_time = time.time()
        fps = 1.0 / (current_time - prev_frame_time)
        prev_frame_time = current_time
        fps_buffer.append(fps)
        avg_fps = sum(fps_buffer) / len(fps_buffer)

        cpu_usage = process.cpu_percent(interval=None)
        norm_cpu = cpu_usage / logical_cores
        cpu_buffer.append(norm_cpu)
        avg_cpu = sum(cpu_buffer) / len(cpu_buffer)

        mem_usage = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        mem_buffer.append(mem_usage)
        avg_mem = sum(mem_buffer) / len(mem_buffer)

        # -------------------------
        # Face Detection
        # -------------------------
        h, w = frame.shape[:2]
        face_detector.setInputSize((w, h))
        rc, faces = face_detector.detect(frame)

        if rc > 0 and faces is not None:
            for face in faces:
                x, y, box_w, box_h = face[:4].astype(int)
                score = face[-1]
                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                cv2.putText(frame, f"{score:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # -------------------------
        # Display System Metrics
        # -------------------------
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"CPU: {avg_cpu:.2f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Memory: {avg_mem:.2f} MB", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("YuNet Face Detection (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()