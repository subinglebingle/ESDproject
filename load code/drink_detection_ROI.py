import torch
import cv2
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 1. 모델 로드(통신 이용)
person_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
drink_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') 

# 2. 이미지 로드
image = cv2.imread("input1.jpg")
if image is None:
    raise ValueError("이미지를 불러올 수 없습니다.")

# BGR → RGB 변환
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. 사람 감지
person_results = person_model(rgb_image)

# 4. 사람 감지 결과 순회
for *box, conf, cls in person_results.xyxy[0]:
    cls_id = int(cls.item())
    label = person_model.names[cls_id] #yolov5에 있는 모델 
    if cls_id!= 0:
        continue  # 'person'=0 클래스만 처리

    x1, y1, x2, y2 = map(int, box)
    person_height = y2 - y1

    # 5. 몸통 영역 (25% ~ 75%) ROI 지정
    torso_y1 = y1 + int(person_height * 0.25)
    torso_y2 = y1 + int(person_height * 0.75)
    roi = image[torso_y1:torso_y2, x1:x2]

    if roi.size == 0:
        continue  # 빈 ROI는 무시

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # 6. 음료수 감지 (ROI에서만)
    drink_results = drink_model(roi_rgb)

    drink_found = False  # 음료수 감지 여부를 기록

    # 음료수를 감지한 결과를 추적
    for *dbox, dconf, dcls in drink_results.xyxy[0]:
        dx1, dy1, dx2, dy2 = map(int, dbox)

        abs_x1 = x1 + dx1
        abs_y1 = torso_y1 + dy1
        abs_x2 = x1 + dx2
        abs_y2 = torso_y1 + dy2

        # 음료수 박스 그리기
        cv2.rectangle(image, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
        cv2.putText(image, "Drink", (abs_x1, abs_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        drink_found = True  # 음료수 감지됨

    # 음료수 없을 경우 'No Drink' 표시
    if not drink_found:
        cv2.putText(image, "No Drink", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 사람의 ROI 그리기
    cv2.rectangle(image, (x1,  torso_y1), (x2,  torso_y2), (0, 255, 255), 2)  # 사람의 영역은 Yellow로 표시

# 7. 결과 출력
resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.3)
cv2.imshow("Detection Result", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
