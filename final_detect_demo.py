import torch
import numpy as np
import mediapipe as mp
import os
import sys
import time
import cv2
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import smtplib
from email.mime.text import MIMEText

email=input('이메일을 입력해주세요':)
if '@' in email:
    email=email
else: email='embeddedsystemdesign@naver.com'

ROOT = os.path.dirname(os.path.abspath(__file__))
YOLOV5_PATH = os.path.join(ROOT)
sys.path.append(YOLOV5_PATH)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

device = select_device('cpu')
drink_model = DetectMultiBackend(os.path.join(YOLOV5_PATH, 'best (1).pt'), device=device)
drink_model.eval()
drink_names = drink_model.names

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

BUZZER_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
pwm = GPIO.PWM(BUZZER_PIN, 1000)

prev_time = 0
last_email_time = 0
EMAIL_INTERVAL = 60  # 초 단위, 이메일 전송 간 최소 간격

def send_email():
    try:
        smtp = smtplib.SMTP('smtp.gmail.com', 587)
        smtp.starttls()
        smtp.login('embeddedsystemdesign12@gmail.com', 'vfnfiybfutnwsbba') 
        msg = MIMEText('room no.10')
        msg['Subject'] = '경고: drinks detected (room no.10)'
        msg['From'] = 'embeddedsystemdesign12@gmail.com'
        msg['To'] = email
        smtp.sendmail('embeddedsystemdesign12@gmail.com', email, msg.as_string())
        smtp.quit()
        print("email sent.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def preprocess(img):
    img_resized = cv2.resize(img, (320, 320))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    return img_tensor.unsqueeze(0).to(device)

frame_count = 0

try:
    while True:
        frame = picam2.capture_array()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img_rgb, (320, 240))
        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 240

        roi_regions = []

        if frame_count % 2 == 0:
            hand_results = hands.process(resized_img)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    x_list = [lm.x * 320 for lm in hand_landmarks.landmark]
                    y_list = [lm.y * 240 for lm in hand_landmarks.landmark]
                    cx = int(np.mean(x_list) * scale_x)
                    cy = int(np.mean(y_list) * scale_y)
                    size = 150
                    x1, y1 = max(cx - size, 0), max(cy - size, 0)
                    x2, y2 = min(cx + size, frame.shape[1]), min(cy + size, frame.shape[0])
                    roi_regions.append((x1, y1, x2, y2))

        drink_detected = False
        for (x1, y1, x2, y2) in roi_regions:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            input_tensor = preprocess(roi)
            with torch.no_grad():
                pred = drink_model(input_tensor)
                pred = non_max_suppression(pred, conf_thres=0.7, iou_thres=0.8)[0]

            scale_roi_x = (x2 - x1) / 320
            scale_roi_y = (y2 - y1) / 320

            if pred is not None:
                for *box, conf, cls in pred:
                    label = drink_names[int(cls)].lower()
                    if label == "drinks":
                        drink_detected = True
                        break

        if drink_detected:
            pwm.start(50)
            pwm.ChangeFrequency(1000)
            time.sleep(0.5)
            pwm.stop()

            curr_email_time = time.time()
            if curr_email_time - last_email_time > EMAIL_INTERVAL:
                send_email()
                last_email_time = curr_email_time

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        print(f'FPS: {fps:.2f}')

        frame_count += 1

finally:
    picam2.stop()
    pwm.stop()
    GPIO.cleanup()
