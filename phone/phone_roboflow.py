import cv2
import torch
import numpy as np
import os
import pickle
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from roboflow import Roboflow

# YOLO 모델 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model_path = 'yolo11l-pose.pt'  # YOLO-Pose 모델

# Roboflow 설정 (항상 CPU에서 실행)
rf = Roboflow(api_key="h2zqBG1dDssjtfUBnY7T")
project = rf.workspace("m17865515473-163-com").project("mobilephone-wusj2")
model = project.version(5).model

# YOLO 포즈 모델 로드 (CUDA 사용)
pose_model = YOLO(pose_model_path)
pose_model.to(device)

# DeepSort 추적기 초기화 (CUDA 사용)
tracker = DeepSort(
    max_age=15,
    n_init=8,
    max_iou_distance=0.5,
    max_cosine_distance=0.3,
    nn_budget=200,
    nms_max_overlap=1.0,
    embedder='mobilenet',
    embedder_gpu=True if device == 'cuda' else False,
    half=True if device == 'cuda' else False
)

# 시간 기반 조건 기록
time_window = 30  # 30프레임 (약 1초)
condition_history = deque(maxlen=time_window)

# 키포인트 연결 정의 (상수 추가)
SKELETON = [
    (0,1), (0,2), (1,3), (2,4),  # 얼굴
    (5,6), (5,7), (7,9), (6,8), (8,10),  # 팔
    (5,11), (6,12), (11,12),  # 몸통
    (11,13), (12,14), (13,15), (14,16)  # 다리
]

# 키포인트 색상 정의 (상수 추가)
KEYPOINT_COLORS = {
    'nose': (255, 0, 0),        # 파랑
    'eyes': (0, 255, 0),        # 초록
    'ears': (0, 0, 255),        # 빨강
    'shoulders': (255, 255, 0),  # 청록
    'elbows': (255, 0, 255),    # 마젠타
    'wrists': (0, 255, 255),    # 노랑
    'hips': (255, 128, 0),      # 주황
    'knees': (128, 255, 0),     # 연두
    'ankles': (0, 128, 255)     # 하늘
}

# 핸드폰 객체 감지 함수 (Roboflow 실행: CPU 사용)
def detect_phones_with_roboflow(image):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
        predictions = model.predict(image_rgb, confidence=40, overlap=30).json()
        
        phone_boxes = []
        for pred in predictions['predictions']:
            if pred['class'] == 'mobilephone':  # 핸드폰 클래스
                x1 = int(pred['x'] - pred['width'] / 2)
                y1 = int(pred['y'] - pred['height'] / 2)
                x2 = int(pred['x'] + pred['width'] / 2)
                y2 = int(pred['y'] + pred['height'] / 2)
                phone_boxes.append([x1, y1, x2, y2])
        return phone_boxes
    except Exception as e:
        print(f"Roboflow Error: {e}")
        return []

# YOLO 포즈 추출 함수 (CUDA 사용)
def detect_pose_with_yolo(frame):
    with torch.no_grad():
        pose_results = pose_model(frame, verbose=False)
        poses = []
        for result in pose_results:
            for bbox, keypoints in zip(result.boxes.xyxy, result.keypoints):
                bbox = list(map(int, bbox.cpu().numpy()))
                keypoints = keypoints.data.cpu().numpy().reshape(-1, 3)
                poses.append((bbox, keypoints))
        return poses

# 시각화 함수
def draw_skeleton(frame, keypoints, confidence_threshold=0.5):
    for start, end in SKELETON:
        if keypoints[start][2] > confidence_threshold and keypoints[end][2] > confidence_threshold:
            start_point = tuple(map(int, keypoints[start][:2]))
            end_point = tuple(map(int, keypoints[end][:2]))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

def draw_keypoints(frame, keypoints, confidence_threshold=0.5):
    for i, kp in enumerate(keypoints):
        if kp[2] > confidence_threshold:
            x, y = map(int, kp[:2])
            color = KEYPOINT_COLORS.get(i, (0, 255, 255))  # 기본 색상: 노랑
            cv2.circle(frame, (x, y), 5, color, -1)

def draw_status(frame, bbox, conditions, is_using_phone):
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 255, 0) if is_using_phone else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    label = "Phone Usage" if is_using_phone else "No Phone"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    for i, (cond, state) in enumerate(conditions.items()):
        if state:
            cv2.putText(frame, f"{cond.replace('_', ' ').title()}", (x1, y1 + 20 + i * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 핸드폰 객체와 손 위치 기반 판단 (조건 1)
def detect_phone_with_hands(phone_boxes, keypoints, threshold_phone_hand=30):
    if len(keypoints) < 17:
        return False

    left_hand = keypoints[9]  # 왼손목
    right_hand = keypoints[10]  # 오른손목

    if left_hand[2] < 0.5 or right_hand[2] < 0.5:
        return False

    left_hand_pos = np.array(left_hand[:2])
    right_hand_pos = np.array(right_hand[:2])

    for phone_box in phone_boxes:
        phone_center = np.array([(phone_box[0] + phone_box[2]) / 2, (phone_box[1] + phone_box[3]) / 2])
        dist_left = np.linalg.norm(left_hand_pos - phone_center)
        dist_right = np.linalg.norm(right_hand_pos - phone_center)

        if dist_left < threshold_phone_hand or dist_right < threshold_phone_hand:
            return True
    return False

# 손 위치가 얼굴/가슴 근처에 있음 (조건 2)
def detect_phone_pose_based(keypoints, threshold_face=80, min_distance_between_hands=40):
    if len(keypoints) < 17:
        return False

    nose = keypoints[0]  # 얼굴 중심 (코)
    left_hand = keypoints[9]  # 왼손목
    right_hand = keypoints[10]  # 오른손목

    if left_hand[2] < 0.5 or right_hand[2] < 0.5 or nose[2] < 0.5:
        return False

    face_left_hand_dist = np.linalg.norm(np.array(left_hand[:2]) - np.array(nose[:2]))
    face_right_hand_dist = np.linalg.norm(np.array(right_hand[:2]) - np.array(nose[:2]))
    hands_dist = np.linalg.norm(np.array(left_hand[:2]) - np.array(right_hand[:2]))

    return (face_left_hand_dist < threshold_face or face_right_hand_dist < threshold_face) and hands_dist > min_distance_between_hands

# 고개 숙임 각도 계산 (조건 3)
def detect_head_tilt(keypoints, head_angle_threshold=45):
    if len(keypoints) < 17:
        return False

    nose = keypoints[0]
    neck = keypoints[1]

    if nose[2] < 0.5 or neck[2] < 0.5:
        return False

    head_angle = np.abs(np.arctan2(nose[1] - neck[1], nose[0] - neck[0]) * 180 / np.pi)
    return head_angle > head_angle_threshold

# 양손 간 거리 계산 (조건 4)
def detect_hand_proximity(keypoints, min_hand_distance=20, max_hand_distance=70):
    if len(keypoints) < 17:
        return False

    left_hand = keypoints[9]
    right_hand = keypoints[10]

    if left_hand[2] < 0.5 or right_hand[2] < 0.5:
        return False

    hands_dist = np.linalg.norm(np.array(left_hand[:2]) - np.array(right_hand[:2]))
    return min_hand_distance <= hands_dist <= max_hand_distance

# 시간 기반 조건 계산 (조건 5)
def calculate_time_based_conditions(history, min_ratio=0.9):
    """
    조건 만족 비율 계산
    :param history: 조건 만족 여부 기록 (deque)
    :param min_ratio: 최소 만족 비율 (예: 90%)
    :return: 조건 만족 여부
    """
    return np.mean(history) >= min_ratio

# 메인 함수
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print(f"카메라가 시작되었습니다. 디바이스: {device}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 핸드폰 감지 (Roboflow, CPU 사용)
            phone_boxes = detect_phones_with_roboflow(frame)

            # 포즈 추출 (YOLO-Pose, CUDA 사용)
            poses = detect_pose_with_yolo(frame)

            # 추적기 업데이트 및 조건 확인
            for bbox, keypoints in poses:
                is_using_phone = False
                conditions = {
                    'phone_near_hand': detect_phone_with_hands(phone_boxes, keypoints),
                    'hand_near_face': detect_phone_pose_based(keypoints),
                    'head_tilted': detect_head_tilt(keypoints),
                    'hands_close': detect_hand_proximity(keypoints),
                }
                if any(conditions.values()):
                    is_using_phone = True

                # 시각화
                draw_skeleton(frame, keypoints)
                draw_keypoints(frame, keypoints)
                draw_status(frame, bbox, conditions, is_using_phone)

            # 핸드폰 바운딩 박스 시각화
            for box in phone_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Phone Usage Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
