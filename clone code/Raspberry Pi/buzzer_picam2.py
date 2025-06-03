import torch
import numpy as np
import mediapipe as mp
import os
import sys
import time
import cv2
import RPi.GPIO as GPIO  # 부저 제어용

from picamera2 import Picamera2

ROOT = os.path.dirname(os.path.abspath(__file__))
YOLOV5_PATH = os.path.join(ROOT)
sys.path.append(YOLOV5_PATH)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

device = select_device('cpu')  # GPU 없으면 CPU

drink_model = DetectMultiBackend(os.path.join(YOLOV5_PATH, 'best.pt'), device=device)
drink_model.eval()
drink_names = drink_model.names

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose

# Mediapipe 초기화 (pose는 complexity 0으로 가볍게)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

BUZZER_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
pwm = GPIO.PWM(BUZZER_PIN, 1000)

prev_time = 0

def preprocess(img):
    img_resized = cv2.resize(img, (320, 320))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    return img_tensor.unsqueeze(0).to(device)

frame_count = 0

try:
    while True:
        frame = picam2.capture_array()
        orig = frame.copy()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        roi_regions = []
        hand_detected = False
        face_detected = False
        pose_detected = False

        # frame_count 기준으로 3 프레임 주기로 모듈 각각 한 번씩 실행
        if frame_count % 3 == 0:
            # 손 검출
            hand_results = hands.process(img_rgb)
            if hand_results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    x_list = [lm.x for lm in hand_landmarks.landmark]
                    y_list = [lm.y for lm in hand_landmarks.landmark]
                    cx = int(np.mean(x_list) * frame.shape[1])
                    cy = int(np.mean(y_list) * frame.shape[0])
                    size = 150
                    x1, y1 = max(cx - size, 0), max(cy - size, 0)
                    x2, y2 = min(cx + size, frame.shape[1]), min(cy + size, frame.shape[0])
                    roi_regions.append((x1, y1, x2, y2))
                    cv2.rectangle(orig, (x1, y1), (x2, y2), (255, 0, 0), 2)

        elif frame_count % 3 == 1:
            # 얼굴 검출
            face_results = face_detection.process(img_rgb)
            if face_results.detections:
                face_detected = True
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw = frame.shape[:2]
                    pad = 80
                    x1 = int(bboxC.xmin * iw) - pad
                    y1 = int(bboxC.ymin * ih) - pad
                    x2 = int((bboxC.xmin + bboxC.width) * iw) + pad
                    y2 = int((bboxC.ymin + bboxC.height) * ih) + pad
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(iw, x2), min(ih, y2)
                    roi_regions.append((x1, y1, x2, y2))
                    cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 0, 255), 2)

        else:
            # 포즈 검출 (상반신)
            pose_results = pose.process(img_rgb)
            if pose_results.pose_landmarks:
                pose_detected = True
                landmarks = pose_results.pose_landmarks.landmark
                keypoints = [landmarks[i] for i in [11, 12, 13, 14, 23, 24, 0]]
                xs = [int(p.x * frame.shape[1]) for p in keypoints]
                ys = [int(p.y * frame.shape[0]) for p in keypoints]
                x1, y1 = max(min(xs) - 40, 0), max(min(ys) - 40, 0)
                x2, y2 = min(max(xs) + 40, frame.shape[1]), min(max(ys) + 40, frame.shape[0])
                roi_regions.append((x1, y1, x2, y2))
                cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # ROI에서 음료수 검출
        drink_detected = False
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
                        drink_detected = True
                        bx1, by1, bx2, by2 = map(int, box)
                        gx1 = int(bx1 * scale_x) + x1
                        gx2 = int(bx2 * scale_x) + x1
                        gy1 = int(by1 * scale_y) + y1
                        gy2 = int(by2 * scale_y) + y1
                        cv2.rectangle(orig, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                        cv2.putText(orig, f'Drink ({conf:.2f})', (gx1, gy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 부저 울리기
        if drink_detected:
            pwm.start(50)
            pwm.ChangeFrequency(1000)
            time.sleep(0.5)
            pwm.stop()

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
    picam2.stop()
    cv2.destroyAllWindows()
    pwm.stop()
    GPIO.cleanup()
