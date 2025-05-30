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

# 모델 1: 사람 인식용 (yolov5n.pt)
person_model = DetectMultiBackend(os.path.join(YOLOV5_PATH, 'yolov5n.pt'), device=device)
person_model.eval()
person_names = person_model.names

# 모델 2: drinks 인식용 (best.pt)
drink_model = DetectMultiBackend(os.path.join(YOLOV5_PATH, 'best.pt'), device=device)
drink_model.eval()
drink_names = drink_model.names

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

        # 공통 입력 이미지 전처리
        def preprocess(img):
            img = cv2.resize(img, (320, 320))
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return img_tensor.unsqueeze(0).to(device)

        # -------- 사람 검출 --------
        person_input = preprocess(frame)
        with torch.no_grad():
            person_pred = person_model(person_input)
            person_pred = non_max_suppression(person_pred, conf_thres=0.6, iou_thres=0.45)[0]

        persons = []
        scale_x = orig.shape[1] / 320
        scale_y = orig.shape[0] / 320

        if person_pred is not None:
            for *box, conf, cls in person_pred:
                label = person_names[int(cls)].lower()
                if label == "person":
                    x1, y1, x2, y2 = map(int, box)
                    x1 = int(x1 * scale_x)
                    x2 = int(x2 * scale_x)
                    y1 = int(y1 * scale_y)
                    y2 = int(y2 * scale_y)
                    persons.append((x1, y1, x2, y2))
                    cv2.rectangle(orig, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 파란색 박스

        # -------- drinks 검출 --------
        drink_input = preprocess(frame)
        with torch.no_grad():
            drink_pred = drink_model(drink_input)
            drink_pred = non_max_suppression(drink_pred, conf_thres=0.7, iou_thres=0.45)[0]

        drinks = []
        if drink_pred is not None:
            for *box, conf, cls in drink_pred:
                label = drink_names[int(cls)].lower()
                if label == "drinks":
                    x1, y1, x2, y2 = map(int, box)
                    x1 = int(x1 * scale_x)
                    x2 = int(x2 * scale_x)
                    y1 = int(y1 * scale_y)
                    y2 = int(y2 * scale_y)
                    drinks.append(((x1, y1, x2, y2), conf))

        # -------- ROI 기반 매칭 --------
        for (px1, py1, px2, py2) in persons:
            for (dx1, dy1, dx2, dy2), conf in drinks:
                if dx1 >= px1 and dy1 >= py1 and dx2 <= px2 and dy2 <= py2:
                    cv2.rectangle(orig, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                    cv2.putText(orig, f'Drink ({conf:.2f})', (dx1, dy1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(orig, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Live Person + Drink Detection", orig)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
