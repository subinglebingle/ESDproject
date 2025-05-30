import torch
import cv2
import numpy as np
import time
import sys
import os
import pathlib

# WindowsPath 오류 방지
pathlib.PosixPath = pathlib.WindowsPath
# 경로 설정
ROOT = os.path.dirname(os.path.abspath(__file__))
YOLOV5_PATH = os.path.join(ROOT,'yolov5')
sys.path.append(YOLOV5_PATH)

# YOLOv5 모델 로딩 관련 모듈
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

device = select_device('cpu')

# 사람 인식용 모델 로딩
person_model = DetectMultiBackend(os.path.join(YOLOV5_PATH, 'yolov5n.pt'), device=device)
person_model.eval()
person_names = person_model.names

# 음료 감지용 모델 로딩
drink_model = DetectMultiBackend(os.path.join(YOLOV5_PATH, 'best1.pt'), device=device)
drink_model.eval()
drink_names = drink_model.names

# 카메라 설정 (PC용)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    sys.exit()

prev_time = 0

def preprocess(img):
    img = cv2.resize(img, (320, 320))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img_tensor.unsqueeze(0).to(device)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다.")
            break

        orig = frame.copy()

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
                    cv2.rectangle(orig, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # -------- ROI 내에서 drink 검출 --------
        for (px1, py1, px2, py2) in persons:
            roi = frame[py1:py2, px1:px2]

            if roi.size == 0:
                continue

            drink_input = preprocess(roi)
            with torch.no_grad():
                drink_pred = drink_model(drink_input)
                drink_pred = non_max_suppression(drink_pred, conf_thres=0.7, iou_thres=0.45)[0]

            roi_scale_x = (px2 - px1) / 320
            roi_scale_y = (py2 - py1) / 320

            if drink_pred is not None:
                for *box, conf, cls in drink_pred:
                    label = drink_names[int(cls)].lower()
                    if label == "drinks":
                        x1, y1, x2, y2 = map(int, box)
                        gx1 = int(x1 * roi_scale_x) + px1
                        gx2 = int(x2 * roi_scale_x) + px1
                        gy1 = int(y1 * roi_scale_y) + py1
                        gy2 = int(y2 * roi_scale_y) + py1

                        cv2.rectangle(orig, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                        cv2.putText(orig, f'Drink ({conf:.2f})', (gx1, gy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS 표시
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(orig, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Live Person ROI + Drink Detection", orig)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
