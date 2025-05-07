import cv2

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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        
        # YuNet requires updating the input size if the frame size changes
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
        
        cv2.imshow("YuNet Face Detection (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
