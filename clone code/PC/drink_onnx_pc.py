import cv2
import numpy as np
import mediapipe as mp
import os
import sys
import time
import pathlib
import onnxruntime as ort

# WindowsPath 오류 방지
pathlib.PosixPath = pathlib.WindowsPath

# 모델 경로 설정
ROOT = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(ROOT, 'best.onnx')

# 클래스 이름 수동 설정 (YOLOv5 학습 시 사용한 클래스 순서와 같게)
drink_names = ["drinks"]  # 클래스가 하나라면

# ONNX 모델 로드
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

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
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img_np, axis=0)

# NMS 함수
def non_max_suppression_onnx(prediction, conf_thres=0.6, iou_thres=0.8):
    boxes = []
    for det in prediction:
        if det[4] < conf_thres:
            continue
        scores = det[4] * det[5:]  # conf * class confidence
        cls = np.argmax(scores)
        score = scores[cls]
        if score > conf_thres:
            x, y, w, h = det[:4]
            box = [x - w/2, y - h/2, x + w/2, y + h/2]  # convert to (x1, y1, x2, y2)
            boxes.append((box, score, cls))
    # NMS
    final_boxes = []
    while boxes:
        boxes.sort(key=lambda x: x[1], reverse=True)
        chosen = boxes.pop(0)
        final_boxes.append(chosen)
        boxes = [b for b in boxes if iou(chosen[0], b[0]) < iou_thres]
    return final_boxes

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig = frame.copy()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        roi_regions = []
        mod = frame_count % 3

        if mod == 0:
            hand_results = hands.process(img_rgb)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    x_list = [lm.x for lm in hand_landmarks.landmark]
                    y_list = [lm.y for lm in hand_landmarks.landmark]
                    cx = int(np.mean(x_list) * frame.shape[1])
                    cy = int(np.mean(y_list) * frame.shape[0])
                    size = 100
                    vertical_bias = 80
                    x1 = max(cx - size, 0)
                    y1 = max(cy - size - vertical_bias, 0)
                    x2 = min(cx + size, frame.shape[1])
                    y2 = min(cy + size, frame.shape[0])
                    roi_regions.append((x1, y1, x2, y2))

        elif mod == 1:
            face_results = face_detection.process(img_rgb)
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw = frame.shape[:2]
                    pad = 80
                    x1 = int(bboxC.xmin * iw) - pad
                    y1 = int(bboxC.ymin * ih) - pad
                    x2 = int((bboxC.xmin + bboxC.width) * iw) + pad
                    y2 = int((bboxC.ymin + bboxC.height) * ih) + pad
                    roi_regions.append((max(0, x1), max(0, y1), min(iw, x2), min(ih, y2)))

        else:
            pose_results = pose.process(img_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                keypoints = [landmarks[i] for i in [11, 12, 13, 14, 23, 24, 0]]
                xs = [int(p.x * frame.shape[1]) for p in keypoints]
                ys = [int(p.y * frame.shape[0]) for p in keypoints]
                x1, y1 = max(min(xs) - 40, 0), max(min(ys) - 40, 0)
                x2, y2 = min(max(xs) + 40, frame.shape[1]), min(max(ys) + 40, frame.shape[0])
                roi_regions.append((x1, y1, x2, y2))

        if not roi_regions:
            cv2.imshow("Detection", orig)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for (x1, y1, x2, y2) in roi_regions:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            input_tensor = preprocess(roi)
            ort_outs = session.run([output_name], {input_name: input_tensor})[0][0]
            detections = non_max_suppression_onnx(ort_outs)

            scale_x = (x2 - x1) / 320
            scale_y = (y2 - y1) / 320

            for (box, conf, cls_id) in detections:
                if drink_names[int(cls_id)].lower() == 'drinks':
                    bx1, by1, bx2, by2 = box
                    gx1 = int(bx1 * scale_x) + x1
                    gx2 = int(bx2 * scale_x) + x1
                    gy1 = int(by1 * scale_y) + y1
                    gy2 = int(by2 * scale_y) + y1
                    cv2.rectangle(orig, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                    cv2.putText(orig, f'Drink ({conf:.2f})', (gx1, gy1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
