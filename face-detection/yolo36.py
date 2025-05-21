# yolo36.py
from __future__ import division, print_function
import cv2, numpy as np, onnxruntime as ort, time, sys
from pathlib import Path

MODEL_PATH="models/yolov8n-face.onnx"; INPUT_SIZE=640
CONF_THRES=0.25; NMS_THRES=0.45; DEVICE_ID=0

def letterbox(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = img_resized
    return canvas, scale, left, top

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def iou(box, boxes):
    inter = (np.maximum(0, np.minimum(boxes[:, 2], box[2]) -
                           np.maximum(boxes[:, 0], box[0])) *
             np.maximum(0, np.minimum(boxes[:, 3], box[3]) -
                           np.maximum(boxes[:, 1], box[1])))
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter + 1e-6
    return inter / union

def non_max_suppression(boxes, scores, iou_thres=0.45):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep

def preprocess(frame):
    img, scale, left, top = letterbox(frame, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return img, scale, left, top

def postprocess(pred, scale, left, top, orig_shape):
    pred = np.squeeze(pred).transpose(1, 0)
    boxes = xywh2xyxy(pred[:, :4])
    scores = pred[:, 4]
    mask = scores > CONF_THRES
    boxes, scores = boxes[mask], scores[mask]
    if boxes.size == 0:
        return []

    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes /= scale
    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    keep = non_max_suppression(boxes, scores, NMS_THRES)
    return [(boxes[i].astype(int), float(scores[i])) for i in keep]

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
