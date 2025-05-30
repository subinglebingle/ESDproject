# live_display.py

import torch
import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
YOLOV5_PATH = os.path.join(ROOT)
sys.path.append(YOLOV5_PATH)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

device = select_device('cpu')
model = DetectMultiBackend(os.path.join(YOLOV5_PATH, 'best.pt'), device=device)
model.eval()
names = model.names

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

prev_time = 0

try:
    while True:
        frame = picam2.capture_array()
        orig = frame.copy()

        img = cv2.resize(frame, (640, 640))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.7, iou_thres=0.45)[0]

        for *box, conf, cls in pred:
            label = names[int(cls)]
            if label.lower() != "drinks":
                continue

            x1, y1, x2, y2 = map(int, box)
            x1 = int(x1 / 640 * orig.shape[1])
            x2 = int(x2 / 640 * orig.shape[1])
            y1 = int(y1 / 640 * orig.shape[0])
            y2 = int(y2 / 640 * orig.shape[0])

            cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig, f'Drink ({conf:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(orig, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Live Drink Detection", orig)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
