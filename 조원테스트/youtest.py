import cv2
import torch
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model_path = 'yolo11l-pose.pt'  # YOLO-Pose 모델 경로

pose_model = YOLO(pose_model_path)
pose_model.to(device)

# 스켈레톤 키포인트 연결 정의
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 얼굴
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
    (5, 11), (6, 12), (11, 12),  # 몸통
    (11, 13), (12, 14), (13, 15), (14, 16)  # 다리
]

def draw_skeleton(frame, keypoints, confidence_threshold=0.5):
    """스켈레톤 그리기"""
    for start, end in SKELETON:
        if keypoints[start][2] > confidence_threshold and keypoints[end][2] > confidence_threshold:
            start_point = tuple(map(int, keypoints[start][:2]))
            end_point = tuple(map(int, keypoints[end][:2]))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

def draw_keypoints(frame, keypoints, confidence_threshold=0.5):
    """키포인트 그리기"""
    for kp in keypoints:
        if kp[2] > confidence_threshold:
            x, y = map(int, kp[:2])
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

def detect_posture(keypoints, desk_height, confidence_threshold=0.5):
    """
    책상에서 엎드린 자세 감지 (조건 강화).
    """
    if len(keypoints) < 17:
        return None, None

    # 주요 키포인트 추출
    shoulder_l = keypoints[5]  # 왼쪽 어깨
    shoulder_r = keypoints[6]  # 오른쪽 어깨
    hip_l = keypoints[11]  # 왼쪽 엉덩이
    hip_r = keypoints[12]  # 오른쪽 엉덩이
    elbow_l = keypoints[7]  # 왼쪽 팔꿈치
    elbow_r = keypoints[8]  # 오른쪽 팔꿈치
    head = keypoints[0]     # 머리

    try:
        # 중심점 계산
        shoulder_center = np.array([(shoulder_l[0] + shoulder_r[0]) / 2, (shoulder_l[1] + shoulder_r[1]) / 2])
        hip_center = np.array([(hip_l[0] + hip_r[0]) / 2, (hip_l[1] + hip_r[1]) / 2])

        # 1. 상체 기울기 계산
        spine_vector = shoulder_center - hip_center
        vertical = np.array([0, -1])  # 수직 기준 벡터
        spine_length = np.linalg.norm(spine_vector)

        if spine_length == 0:
            return None, None  # 유효하지 않은 벡터

        cos_lean = np.dot(spine_vector, vertical) / spine_length
        lean_angle = np.degrees(np.arccos(np.clip(cos_lean, -1.0, 1.0)))

        # 2. 팔 위치 확인 (팔꿈치가 책상 높이 아래로 내려가면 엎드린 것으로 간주)
        arm_on_desk = (
            (elbow_l[1] > desk_height if elbow_l[2] > confidence_threshold else False) or
            (elbow_r[1] > desk_height if elbow_r[2] > confidence_threshold else False)
        )

        # 3. 머리 위치 확인 (머리가 어깨보다 아래에 있어야 엎드린 자세로 간주)
        is_head_down = head[1] > shoulder_center[1] if head[2] > confidence_threshold else False

        # 엎드린 자세 조건
        if 20 <= lean_angle <= 50 and arm_on_desk and is_head_down:
            return 'lying on desk', lean_angle

    except Exception as e:
        print(f"Error calculating posture: {e}")

    return None, None



def draw_status(frame, bbox, posture, angle=None):
    """상태 정보 표시"""
    if posture is None and angle is None:
        return

    x1, y1, x2, y2 = map(int, bbox)
    
    # 각도 정보 표시
    if angle is not None:
        angle_text = f"Angle: {angle:.1f}"
        cv2.putText(frame, angle_text, (x1, y1 - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 자세 정보 표시
    if posture == 'lying on desk':
        color = (255, 0, 0)  # 파란색
        status = "Lying on desk"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, status, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def estimate_desk_height(bboxes):
    """
    책상 높이 추정 (바운딩 박스의 하단 가장자리 사용)
    """
    if not bboxes:
        return 300  # 기본 책상 높이
    return int(np.mean([bbox[3] for bbox in bboxes]))

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print(f"카메라가 시작되었습니다. 디바이스: {device}")
    print("종료하려면 'q'를 누르세요.")

    desk_height = 300  # 기본 책상 높이

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 모델 추론
            pose_results = pose_model(frame, verbose=False)

            for result in pose_results:
                for bbox, keypoints in zip(result.boxes.xyxy, result.keypoints):
                    bbox = bbox.cpu().numpy()
                    keypoints = keypoints.data.cpu().numpy().reshape(-1, 3)

                    # 스켈레톤과 키포인트 그리기
                    draw_skeleton(frame, keypoints)
                    draw_keypoints(frame, keypoints)

                    # 자세 감지
                    posture, angle = detect_posture(keypoints, desk_height)

                    # 상태와 각도 표시
                    draw_status(frame, bbox, posture, angle)

            # 프레임 출력
            cv2.imshow("Desk Sleeping Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()