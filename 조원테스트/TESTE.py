import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import face_recognition

# 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FACE_FEATURES_DIR = 'features/'  # 얼굴 및 색상 특징 저장 디렉토리

# YOLO 모델 로드
yolo_model = YOLO('yolo11l-pose.pt')

# DeepSort 초기화
tracker = DeepSort(
    max_age=30,  # 객체가 사라졌을 때 ID 유지 시간
    n_init=3,
    max_iou_distance=0.7,
    max_cosine_distance=0.3,
    nn_budget=100,
    embedder='mobilenet',
    embedder_gpu=True if DEVICE == 'cuda' else False,
    half=True if DEVICE == 'cuda' else False
)

# 얼굴 특징 저장 함수
def save_features(features, object_id):
    """얼굴 및 색상 특징을 디렉토리에 저장"""
    if not os.path.exists(FACE_FEATURES_DIR):
        os.makedirs(FACE_FEATURES_DIR)
    filename = os.path.join(FACE_FEATURES_DIR, f"{object_id}.npz")
    np.savez(filename, **features)


# 얼굴 및 색상 특징 로드 함수
def load_features(object_id):
    """디렉토리에서 얼굴 및 색상 특징을 로드"""
    filename = os.path.join(FACE_FEATURES_DIR, f"{object_id}.npz")
    if os.path.exists(filename):
        return np.load(filename)
    return None


# 얼굴 특징 추출 함수
def extract_face_features(image):
    """얼굴 특징 추출"""
    if image is None or image.size == 0:  # 빈 이미지 확인
        return None
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_img)
    if len(face_encodings) > 0:
        return face_encodings[0]
    return None


# 색상 평균 계산 함수 (머리 색상, 옷 색상)
def calculate_average_color(image, top_ratio=0.3, bottom_ratio=0.7):
    """주어진 이미지에서 상단 (머리)과 하단 (옷) 색상 평균 계산"""
    height, width, _ = image.shape
    top_height = int(height * top_ratio)
    bottom_height = int(height * bottom_ratio)

    # 머리 영역 (상단)
    head_img = image[0:top_height, :]
    head_avg_color = np.mean(head_img, axis=(0, 1))

    # 옷 영역 (하단)
    body_img = image[bottom_height:, :]
    body_avg_color = np.mean(body_img, axis=(0, 1))

    return head_avg_color, body_avg_color


# 얼굴 특징과 색상 비교 함수
def compare_features(feature1, feature2):
    """얼굴 특징 벡터 비교"""
    if feature1 is None or feature2 is None:
        return float('inf')
    return np.linalg.norm(np.array(feature1) - np.array(feature2))


def compare_colors(color1, color2):
    """색상 평균 비교 (유클리드 거리)"""
    return np.linalg.norm(np.array(color1) - np.array(color2))


# 마우스 클릭 이벤트 처리 함수
def on_mouse(event, x, y, flags, param):
    """마우스 클릭 이벤트 처리"""
    global selected_object_id, selected_object_features

    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭
        frame, tracks = param
        for track in tracks:
            if track.is_confirmed():
                l, t, r, b = map(int, track.to_tlbr())
                if l <= x <= r and t <= y <= b:  # 클릭 위치가 바운딩 박스 안에 있으면
                    selected_object_id = int(track.track_id)
                    print(f"선택된 객체 ID: {selected_object_id}")

                    # 선택된 객체의 얼굴 특징 및 색상 정보를 추출하여 저장
                    object_img = frame[t:b, l:r]
                    face_feature = extract_face_features(object_img)
                    head_color, body_color = calculate_average_color(object_img)
                    
                    features = {
                        'face_feature': face_feature,
                        'head_color': head_color,
                        'body_color': body_color
                    }

                    if face_feature is not None:
                        save_features(features, selected_object_id)
                    break

# YOLO 감지 및 트래킹 결과 처리 함수
def process_detections(results, min_confidence=0.7):
    """YOLO 감지 결과 처리"""
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0 and conf > min_confidence:  # 사람 클래스
                w = x2 - x1
                h = y2 - y1
                if w * h > 400 and w/h < 1.5 and h/w < 2.5:  # 크기 및 비율 필터링
                    bbox = [x1, y1, w, h]
                    detections.append((bbox, conf, 0))  # bbox, confidence, 클래스 ID
    return detections

# 메인 함수
def main():
    global selected_object_id, selected_object_features

    selected_object_id = None  # 초기화: 객체 ID가 선택되기 전까지는 None
    selected_object_features = None  # 얼굴 특징도 초기화

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Object Tracking")
    print("'ESC' 키: 프로그램 종료")

    # 객체 ID와 얼굴 특징 저장
    face_features_dict = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 감지 및 DeepSort 트래킹
            with torch.no_grad():
                preds = yolo_model(frame, verbose=False)
            detections = process_detections(preds)
            tracks = tracker.update_tracks(detections, frame=frame)

            # 마우스 이벤트 연결
            cv2.setMouseCallback("Object Tracking", on_mouse, (frame, tracks))

            # 트랙 처리
            for track in tracks:
                track_id = int(track.track_id)

                # 확인된 트랙만 처리
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                l, t, r, b = map(int, track.to_tlbr())
                color = (0, 255, 0) if track_id == selected_object_id else (255, 0, 0)
                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                label = f"ID: {track_id}"
                cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 저장된 얼굴 및 색상 특징을 로드하여 비교
                if track_id == selected_object_id:
                    saved_features = load_features(track_id)
                    if saved_features is not None:
                        saved_face_feature = saved_features['face_feature']
                        saved_head_color = saved_features['head_color']
                        saved_body_color = saved_features['body_color']

                        # 현재 객체 이미지에서 얼굴 및 색상 정보 추출
                        current_object_img = frame[t:b, l:r]
                        current_face_feature = extract_face_features(current_object_img)
                        current_head_color, current_body_color = calculate_average_color(current_object_img)

                        # 얼굴 및 색상 특징 비교
                        feature_distance = compare_features(saved_face_feature, current_face_feature)
                        head_color_distance = compare_colors(saved_head_color, current_head_color)
                        body_color_distance = compare_colors(saved_body_color, current_body_color)

                        print(f"얼굴 특징 거리: {feature_distance}, 머리 색상 거리: {head_color_distance}, 몸 색상 거리: {body_color_distance}")

                        if feature_distance < 0.6 and head_color_distance < 30 and body_color_distance < 30:
                            print(f"동일 객체: {track_id} (얼굴, 머리, 몸 색상 비교)")

            cv2.imshow("Object Tracking", frame)

            # ESC 키로 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
