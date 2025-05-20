# yolo38.py
import cv2
from ultralytics import YOLO

MODEL_PATH="models/yolov8n-face.pt"
DEVICE_ID=0

def main():
    model = YOLO(MODEL_PATH).to("cuda")  # o "cpu"
    cap = cv2.VideoCapture(DEVICE_ID)
    if not cap.isOpened():
        print("Error: no webcam."); return

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model(frame)
        for box in results[0].boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{score:.2f}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        cv2.imshow("YOLO38 – q to quit", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
