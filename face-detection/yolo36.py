# yolo36.py
from __future__ import division, print_function
import cv2, numpy as np, onnxruntime as ort, time, sys
from pathlib import Path

MODEL_PATH="models/yolov8n-face.onnx"; INPUT_SIZE=640
CONF_THRES=0.25; NMS_THRES=0.45; DEVICE_ID=0

# (define aquí letterbox, xywh2xyxy, iou, non_max_suppression, preprocess, postprocess)

def get_execution_providers():
    pref=["CUDAExecutionProvider","CPUExecutionProvider"]
    avail=ort.get_available_providers()
    return [p for p in pref if p in avail]

def main():
    if not Path(MODEL_PATH).is_file(): sys.exit("Modelo ONNX no encontrado")
    sess=ort.InferenceSession(MODEL_PATH, providers=get_execution_providers())
    inp, outp = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    cap=cv2.VideoCapture(DEVICE_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    if not cap.isOpened(): sys.exit("No se puede abrir cámara.")

    while True:
        ret, frame = cap.read(); 
        if not ret: break

        img, sc, l, t0 = preprocess(frame)
        pred = sess.run([outp], {inp: img})[0]
        dets = postprocess(pred, sc, l, t0, frame.shape)

        for (x1,y1,x2,y2),s in dets:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{s:.2f}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        cv2.imshow("YOLO36 – q to quit", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
