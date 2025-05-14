import torch
import cv2
import numpy as np
import pathlib
import time

# WindowsPath 오류 방지
pathlib.PosixPath = pathlib.WindowsPath

# 모델 로드
person_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
drink_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("웹캠을 열 수 없습니다.")

# FPS 측정 초기화
prev_time = time.time()

# Threshold 설정 (0.7 고정)
CONF_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    person_results = person_model(rgb_image)

    for *box, conf, cls in person_results.xyxy[0]:
     

        cls_id = int(cls.item())
        if cls_id != 0:
            continue  # 사람 클래스만 처리

        x1, y1, x2, y2 = map(int, box)
        person_height = y2 - y1
        torso_y1 = y1 + int(person_height * 0.25)
        torso_y2 = y1 + int(person_height * 0.75)
        roi = frame[torso_y1:torso_y2, x1:x2]

        if roi.size == 0:
            continue

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        drink_results = drink_model(roi_rgb)

        drink_found = False
        for *dbox, dconf, dcls in drink_results.xyxy[0]:
            if dconf < CONF_THRESHOLD:
                continue

            dx1, dy1, dx2, dy2 = map(int, dbox)
            abs_x1 = x1 + dx1
            abs_y1 = torso_y1 + dy1
            abs_x2 = x1 + dx2
            abs_y2 = torso_y1 + dy2

            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
            cv2.putText(frame, "Drink", (abs_x1, abs_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            drink_found = True

        if not drink_found:
            cv2.putText(frame, "No Drink", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.rectangle(frame, (x1, torso_y1), (x2, torso_y2), (0, 255, 255), 2)

    # FPS 출력
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 화면 출력
    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Live Detection (Threshold=0.7)", resized_frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
