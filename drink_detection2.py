import torch
import cv2
import numpy as np
import pathlib
import time  # ⬅️ 실행 시간 측정용

# 경로 문제 해결
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 1. 모델 로드
drink_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # 커스텀 모델

# 2. 이미지 로드
image = cv2.imread("input.jpg")
if image is None:
    raise ValueError("이미지를 불러올 수 없습니다.")

# BGR → RGB 변환
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. 실행 시간 측정 시작
start_time = time.time()

# 4. 음료수 감지 수행
drink_results = drink_model(rgb_image)

# 5. 실행 시간 측정 종료
end_time = time.time()

elapsed_time = end_time - start_time
fps = 1.0 / elapsed_time if elapsed_time > 0 else float('inf')

print(f"추론 시간: {elapsed_time:.4f}초")
print(f"FPS (프레임당 처리 속도): {fps:.2f} FPS")

# 6. 감지 결과 시각화
for *box, conf, cls in drink_results.xyxy[0]:
    x1, y1, x2, y2 = map(int, box)

    # 음료수 박스 그리기
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, "Drink", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 7. 결과 출력
resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.3)
cv2.imshow("Detection Result", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
