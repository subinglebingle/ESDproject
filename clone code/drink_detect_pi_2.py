import torch
import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
YOLOV5_PATH = os.path.join(ROOT, 'yolov5')
sys.path.append(YOLOV5_PATH)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
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
        # capture_array() 대신 capture_image().to_array() 사용
        frame = picam2.capture_image().to_array()
        orig = frame.copy()

        # 입력 이미지 크기 320x320으로 변경 (YOLO 입력 크기 맞춤)
        img = cv2.resize(frame, (320, 320))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():  # no_grad 적용
            pred = model(img_tensor)
            pred = non_max_suppression(pred, conf_thres=0.7, iou_thres=0.6)[0]

        if pred is not None:
            for *box, conf, cls in pred:
                label = names[int(cls)]
                if label.lower() != "drinks":
                    continue

                x1, y1, x2, y2 = map(int, box)
                # 좌표를 원본 크기 기준으로 다시 스케일링
                x1 = int(x1 / 320 * orig.shape[1])
                x2 = int(x2 / 320 * orig.shape[1])
                y1 = int(y1 / 320 * orig.shape[0])
                y2 = int(y2 / 320 * orig.shape[0])

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
