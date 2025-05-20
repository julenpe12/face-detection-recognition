# yolo38_debug.py
import cv2, time, collections, psutil, os
from ultralytics import YOLO
import GPUtil

MODEL_PATH="models/yolov8n-face.pt"
DEVICE_ID=0

def main():
    model = YOLO(MODEL_PATH).to("cuda")  # o .to("cpu")
    cap = cv2.VideoCapture(DEVICE_ID)
    if not cap.isOpened(): 
        print("Error: no webcam."); return

    proc=psutil.Process(os.getpid())
    cores=psutil.cpu_count()
    fps_buf, cpu_buf, mem_buf, gpu_buf = [collections.deque(maxlen=30) for _ in range(4)]
    prev_t=time.time(); frame_count=0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # Métricas
        mem_buf.append(proc.memory_info().rss/(1024*1024))
        cpu_buf.append(proc.cpu_percent()/cores)
        if frame_count%10==0:
            g=GPUtil.getGPUs()
            gpu_buf.append(g[0].load*100 if g else 0)
        t=time.time()
        fps_buf.append(1/(t-prev_t)); prev_t=t

        # Inferencia Ultralytics
        results = model(frame)
        for box in results[0].boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            score=float(box.conf[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{score:.2f}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        # Overlay métricas
        cv2.putText(frame,f"FPS:{sum(fps_buf)/len(fps_buf):.1f}",(5,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,255,0),2)
        cv2.putText(frame,f"CPU:{sum(cpu_buf)/len(cpu_buf):.1f}%",(5,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,255,0),2)
        cv2.putText(frame,f"RAM:{sum(mem_buf)/len(mem_buf):.1f}MB",(5,90),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,255,0),2)
        cv2.putText(frame,f"GPU:{sum(gpu_buf)/len(gpu_buf):.1f}%",(5,120),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,255,0),2)

        cv2.imshow("YOLO38 Debug – q to quit", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
