import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO

# YOLO 모델 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model_path = 'yolo11l-pose.pt'  # YOLO-Pose 모델
object_model_path = 'yolo11x.pt'     # YOLO-핸드폰 감지 모델

pose_model = YOLO(pose_model_path)
pose_model.to(device)

object_model = YOLO(object_model_path)
object_model.to(device)

# 시간 기반 조건 기록
time_window = 30  # 30프레임 (약 1초)
condition_history = deque(maxlen=time_window)

# 키포인트 연결 정의
SKELETON = [
    (0,1), (0,2), (1,3), (2,4),  # 얼굴
    (5,6), (5,7), (7,9), (6,8), (8,10),  # 팔
    (5,11), (6,12), (11,12),  # 몸통
    (11,13), (12,14), (13,15), (14,16)  # 다리
]

# 키포인트 색상 정의
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

def draw_skeleton(frame, keypoints, confidence_threshold=0.5):
    """스켈레톤 그리기"""
    for start, end in SKELETON:
        if keypoints[start][2] > confidence_threshold and keypoints[end][2] > confidence_threshold:
            start_point = tuple(map(int, keypoints[start][:2]))
            end_point = tuple(map(int, keypoints[end][:2]))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

def draw_keypoints(frame, keypoints, confidence_threshold=0.5):
    """키포인트 그리기"""
    for i, kp in enumerate(keypoints):
        if kp[2] > confidence_threshold:
            x, y = map(int, kp[:2])
            # 키포인트 종류에 따라 다른 색상 사용
            if i == 0:  # nose
                color = KEYPOINT_COLORS['nose']
            elif i in [1, 2]:  # eyes
                color = KEYPOINT_COLORS['eyes']
            elif i in [3, 4]:  # ears
                color = KEYPOINT_COLORS['ears']
            elif i in [5, 6]:  # shoulders
                color = KEYPOINT_COLORS['shoulders']
            elif i in [7, 8]:  # elbows
                color = KEYPOINT_COLORS['elbows']
            elif i in [9, 10]:  # wrists
                color = KEYPOINT_COLORS['wrists']
            elif i in [11, 12]:  # hips
                color = KEYPOINT_COLORS['hips']
            elif i in [13, 14]:  # knees
                color = KEYPOINT_COLORS['knees']
            else:  # ankles
                color = KEYPOINT_COLORS['ankles']
            
            cv2.circle(frame, (x, y), 5, color, -1)

def draw_status(frame, bbox, conditions, is_using_phone):
    """상태 정보 표시"""
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 255, 0) if is_using_phone else (0, 0, 255)
    
    # 바운딩 박스
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # 상태 텍스트
    status_text = []
    if conditions['phone_near_hand']: status_text.append("Phone Near Hand")
    if conditions['hand_near_face']: status_text.append("Hand Near Face")
    if conditions['head_tilted']: status_text.append("Head Tilted")
    if conditions['hands_close']: status_text.append("Hands Close")
    
    # 메인 라벨
    label = "Phone Usage" if is_using_phone else "No Phone"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 세부 상태
    for i, text in enumerate(status_text):
        y_pos = y1 + 20 + (i * 20)
        cv2.putText(frame, text, (x1 + 5, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 핸드폰 객체와 손 위치 조건 (핸드폰 객체 탐지)
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

        # 핸드폰 객체가 양손 중 하나 근처에 있어야 함
        if dist_left < threshold_phone_hand or dist_right < threshold_phone_hand:
            return True
    return False

# 손 위치가 얼굴/가슴 근처에 있는 조건
def detect_phone_pose_based(keypoints, threshold_face=80):
    if len(keypoints) < 17:
        return False

    nose = keypoints[0]  # 얼굴 중심 (코)
    left_hand = keypoints[9]  # 왼손목
    right_hand = keypoints[10]  # 오른손목

    if left_hand[2] < 0.5 or right_hand[2] < 0.5 or nose[2] < 0.5:
        return False

    face_left_hand_dist = np.linalg.norm(np.array(left_hand[:2]) - np.array(nose[:2]))
    face_right_hand_dist = np.linalg.norm(np.array(right_hand[:2]) - np.array(nose[:2]))

    return face_left_hand_dist < threshold_face or face_right_hand_dist < threshold_face

# 양손 간 거리 조건
def detect_hand_proximity(keypoints, min_hand_distance=10, max_hand_distance=50):
    if len(keypoints) < 17:
        return False

    left_hand = keypoints[9]
    right_hand = keypoints[10]

    if left_hand[2] < 0.5 or right_hand[2] < 0.5:
        return False

    hands_dist = np.linalg.norm(np.array(left_hand[:2]) - np.array(right_hand[:2]))
    return min_hand_distance <= hands_dist <= max_hand_distance

# 시간 기반 조건 계산
def calculate_time_based_conditions(history, min_ratio=0.7):
    """
    조건 만족 비율 계산
    :param history: 조건 만족 여부 기록 (deque)
    :param min_ratio: 최소 만족 비율 (예: 70%)
    :return: 조건 만족 여부
    """
    return np.mean(history) >= min_ratio

# 고개 숙임 각도 계산 (조건 3)
def detect_head_tilt(keypoints, head_angle_threshold=30):
    """
    고개 숙임 감지
    :param keypoints: 포즈 키포인트
    :param head_angle_threshold: 고개 숙임으로 판단할 각도 임계값 (기본값: 30도)
    :return: 고개 숙임 여부
    """
    if len(keypoints) < 17:
        return False

    # 필요한 키포인트 추출
    nose = keypoints[0]  # 코
    left_shoulder = keypoints[5]  # 왼쪽 어깨
    right_shoulder = keypoints[6]  # 오른쪽 어깨

    # 키포인트 신뢰도 확인
    if nose[2] < 0.5 or left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5:
        return False

    # 어깨 중심점 계산
    shoulder_center = np.array([
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    ])

    # 코와 어깨 중심점 사이의 각도 계산
    dx = nose[0] - shoulder_center[0]
    dy = nose[1] - shoulder_center[1]
    angle = np.abs(np.degrees(np.arctan2(dy, dx)))

    # 수직 기준 각도 계산 (90도에서 얼마나 벗어났는지)
    tilt_angle = np.abs(90 - angle)

    return tilt_angle > head_angle_threshold

# 비디오 스트림 처리
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 모델 추론
    pose_results = pose_model(frame, verbose=False)
    object_results = object_model(frame, verbose=False)

    # 핸드폰 객체 검출
    phone_boxes = []
    for obj in object_results:
        for box, cls in zip(obj.boxes.xyxy, obj.boxes.cls):
            if int(cls) == 67:  # 핸드폰 클래스
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                phone_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Phone", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 포즈 검출 및 시각화
    for result in pose_results:
        for bbox, keypoints in zip(result.boxes.xyxy, result.keypoints):
            kpts = keypoints.data.cpu().numpy().reshape(-1, 3)
            
            # 조건 검사
            conditions = {
                'phone_near_hand': detect_phone_with_hands(phone_boxes, kpts),
                'hand_near_face': detect_phone_pose_based(kpts),
                'head_tilted': detect_head_tilt(kpts),
                'hands_close': detect_hand_proximity(kpts)
            }
            
            # 전체 상태 업데이트
            condition_history.append(any(conditions.values()))
            is_using_phone = calculate_time_based_conditions(condition_history)
            
            # 시각화
            draw_skeleton(frame, kpts)
            draw_keypoints(frame, kpts)
            draw_status(frame, bbox, conditions, is_using_phone)

    cv2.imshow("Phone Usage Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
