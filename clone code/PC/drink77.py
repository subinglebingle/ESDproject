import cv2
import torch
import numpy as np
import mediapipe as mp
import os
import sys
import time
import pathlib

# WindowsPath 오류 방지
pathlib.PosixPath = pathlib.WindowsPath

# YOLOv5 모델 경로 설정
ROOT = os.path.dirname(os.path.abspath(__file__))
YOLOV5_PATH = os.path.join(ROOT, 'yolov5')
sys.path.append(YOLOV5_PATH)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# 장치 설정
device = select_device('cpu')  # CPU 사용

# YOLOv5 음료수 감지 모델 로드
drink_model = DetectMultiBackend(os.path.join(YOLOV5_PATH, 'best.pt'), device=device)
drink_model.eval()
drink_names = drink_model.names

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# 웹캠 초기화
cap = cv2.VideoCapture(0)
prev_time = 0
frame_count = 0

# 전처리 함수
def preprocess(img):
    img_resized = cv2.resize(img, (320, 320))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    return img_tensor.unsqueeze(0).to(device)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig = frame.copy()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        roi_regions = []
        hand_detected = False

        # frame_count에 따라 실행할 모듈 분기
        mod = frame_count % 3

        if mod == 0:
            # 손 감지 및 ROI 설정
            hand_results = hands.process(img_rgb)
            if hand_results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    x_list = [lm.x for lm in hand_landmarks.landmark]
                    y_list = [lm.y for lm in hand_landmarks.landmark]
                    cx = int(np.mean(x_list) * frame.shape[1])
                    cy = int(np.mean(y_list) * frame.shape[0])
                    size = 150  # 확대
                    x1, y1 = max(cx - size, 0), max(cy - size, 0)
                    x2, y2 = min(cx + size, frame.shape[1]), min(cy + size, frame.shape[0])
                    roi_regions.append((x1, y1, x2, y2))
                    cv2.rectangle(orig, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 파란색

        elif mod == 1:
            # 얼굴 감지 및 ROI 설정
            face_results = face_detection.process(img_rgb)
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw = frame.shape[:2]
                    pad = 80  # 확대
                    x1 = int(bboxC.xmin * iw) - pad
                    y1 = int(bboxC.ymin * ih) - pad
                    x2 = int((bboxC.xmin + bboxC.width) * iw) + pad
                    y2 = int((bboxC.ymin + bboxC.height) * ih) + pad
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(iw, x2), min(ih, y2)
                    roi_regions.append((x1, y1, x2, y2))
                    cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색

        else:
            # 포즈 감지 및 상반신 ROI 설정 (손이 없을 때만)
            pose_results = pose.process(img_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                keypoints = [landmarks[i] for i in [11, 12, 13, 14, 23, 24, 0]]  # 어깨, 팔, 허리, 코
                xs = [int(p.x * frame.shape[1]) for p in keypoints]
                ys = [int(p.y * frame.shape[0]) for p in keypoints]
                x1, y1 = max(min(xs) - 40, 0), max(min(ys) - 40, 0)
                x2, y2 = min(max(xs) + 40, frame.shape[1]), min(max(ys) + 40, frame.shape[0])
                roi_regions.append((x1, y1, x2, y2))
                cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 노란색

        # ROI 없으면 안내 문구 표시 및 다음 프레임으로 넘어감
        if not roi_regions:
            cv2.putText(orig, 'No ROI detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Detection", orig)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1
            continue

        # ROI 내 음료수 탐지
        for (x1, y1, x2, y2) in roi_regions:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            input_tensor = preprocess(roi)
            with torch.no_grad():
                pred = drink_model(input_tensor)
                pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.8)[0]

            scale_x = (x2 - x1) / 320
            scale_y = (y2 - y1) / 320

            if pred is not None:
                for *box, conf, cls in pred:
                    label = drink_names[int(cls)].lower()
                    if label == "drinks":
                        bx1, by1, bx2, by2 = map(int, box)
                        gx1 = int(bx1 * scale_x) + x1
                        gx2 = int(bx2 * scale_x) + x1
                        gy1 = int(by1 * scale_y) + y1
                        gy2 = int(by2 * scale_y) + y1
                        cv2.rectangle(orig, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                        cv2.putText(orig, f'Drink ({conf:.2f})', (gx1, gy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS 표시
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(orig, f'FPS: {fps:.2f}', (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Detection", orig)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

finally:
    cap.release()
    cv2.destroyAllWindows()
