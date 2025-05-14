# -*- coding: utf-8 -*-
"""
Dependencias mínimas (Python 3.6):

pip install opencv-python==4.5.5.64 \
            onnxruntime==1.6.0 \
            psutil==5.9.5 \
            GPUtil==1.4.0
"""

import cv2
import time
import collections
import psutil
import os
import numpy as np
import GPUtil            # Información GPU
import onnxruntime as ort

# -------------------------  CONFIGURACIÓN  -------------------------

MODEL_PATH   = "models/yolov8n-face.onnx"
INPUT_SIZE   = 640          # Entrada cuadrada 640×640 (la que usó al exportar)
CONF_THRESH  = 0.25
NMS_THRESH   = 0.45         # Por si usa NMS manual

# Preferencias de aceleración
CUDA = True                 # Ponga a False si sólo dispone de CPU

# -------------------------  PREPROCESADO  -------------------------

def letterbox(img, new_size=INPUT_SIZE):
    """Redimensiona con márgenes manteniendo proporción (estilo YOLO)."""
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(scale * h), int(scale * w)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, scale, (nw, nh)

def preprocess(frame):
    img, scale, (nw, nh) = letterbox(frame)
    blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (INPUT_SIZE, INPUT_SIZE),
                                 swapRB=True, crop=False)
    return blob, scale, (nw, nh)

# -------------------------  POSTPROCESADO  -------------------------

def postprocess(onnx_output, scale, pad, conf_th=CONF_THRESH):
    """Convierte la salida [num_det, 6] a listas de cajas y puntuaciones."""
    boxes, scores = [], []
    detections = onnx_output[0]  # (num,6) -> x1,y1,x2,y2,conf,class_id
    for det in detections:
        conf = det[4]
        if conf < conf_th:
            continue
        x1, y1, x2, y2 = det[:4]
        # Deshacer letterbox
        x1 = (x1 - 0) / scale
        y1 = (y1 - 0) / scale
        x2 = (x2 - 0) / scale
        y2 = (y2 - 0) / scale
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
        scores.append(float(conf))
    return boxes, scores

# -------------------------  CARGA DEL MODELO  -------------------------

providers = ['CUDAExecutionProvider'] if CUDA else ['CPUExecutionProvider']
session = ort.InferenceSession(MODEL_PATH, providers=providers)

# -------------------------  RECURSOS DEL SISTEMA  -------------------------

process = psutil.Process(os.getpid())
logical_cores = psutil.cpu_count(True)

prev_time   = time.time()
fps_buffer  = collections.deque(maxlen=30)
cpu_buffer  = collections.deque(maxlen=30)
mem_buffer  = collections.deque(maxlen=30)
gpu_buffer  = collections.deque(maxlen=30)

# -------------------------  BUCLE PRINCIPAL  -------------------------

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Inferencia ---
    blob, scale, _ = preprocess(frame)
    onnx_out = session.run(None, {session.get_inputs()[0].name: blob})
    boxes, scores = postprocess(onnx_out, scale, (0, 0))

    # Dibujar resultados
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "%.2f" % sc, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Métricas de rendimiento ---
    mem = process.memory_info().rss / (1024 * 1024)
    mem_buffer.append(mem)

    now = time.time()
    fps_buffer.append(1.0 / (now - prev_time))
    prev_time = now

    cpu = process.cpu_percent(None) / logical_cores
    cpu_buffer.append(cpu)

    if len(fps_buffer) and not len(gpu_buffer) % 10:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_buffer.append(gpus[0].load * 100)

    # Overlay
    cv2.putText(frame, "FPS: %d" % int(sum(fps_buffer)/len(fps_buffer)), (7, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2)
    cv2.putText(frame, "CPU: %.1f%% core" % (sum(cpu_buffer)/len(cpu_buffer)),
                (7, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2)
    cv2.putText(frame, "Mem: %.1f MB" % (sum(mem_buffer)/len(mem_buffer)),
                (7, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2)
    if gpu_buffer:
        cv2.putText(frame, "GPU: %.1f%%" % (sum(gpu_buffer)/len(gpu_buffer)),
                    (7, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2)

    cv2.imshow("YOLOv8‑ONNX (Python 3.6) – 'q' para salir", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
