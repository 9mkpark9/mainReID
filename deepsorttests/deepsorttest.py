import cv2
import torch
import os
import pickle
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# YOLO 모델 로드
yolo_model = YOLO('yolo11x.pt')
yolo_model.to(DEVICE)

# DeepSort 초기화
tracker = DeepSort(
    max_age=30,        # 더 오래 트랙 유지
    n_init=3,          # 더 빠른 트랙 초기화
    max_iou_distance=0.7,
    max_cosine_distance=0.3,
    nn_budget=100,
    embedder='mobilenet',
    embedder_gpu=True if DEVICE == 'cuda' else False,
    half=True if DEVICE == 'cuda' else False
)

# 저장 디렉토리 설정
PERSON_DB_DIR = "person_database"
os.makedirs(PERSON_DB_DIR, exist_ok=True)

class PersonDB:
    def __init__(self):
        self.db = {}  # {track_id: {'images': [], 'features': []}}
        self.max_images = 100
        self.is_collecting = False
        self.is_matching = False
        self.matching_threshold = 0.6
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # CNN 특징 추출기 초기화 및 CUDA로 이동
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        ).to(self.device)
        
        self.load_db()
        self.precompute_features()
        print(f"로드된 ID 수: {len(self.db)}")

    def load_db(self):
        """기존 데이터베이스 로드"""
        for person_dir in os.listdir(PERSON_DB_DIR):
            if person_dir.startswith('ID_'):
                track_id = int(person_dir.split('_')[1])
                self.db[track_id] = {'images': [], 'features': []}
                person_path = os.path.join(PERSON_DB_DIR, person_dir)
                for img_file in os.listdir(person_path):
                    if img_file.endswith('.jpg'):
                        img_path = os.path.join(person_path, img_file)
                        self.db[track_id]['images'].append(img_path)

    def save_person_image(self, frame, bbox, track_id):
        """이미지 저장 및 특징 추출"""
        if not self.is_collecting:
            return None

        try:
            track_id = int(track_id)
            x1, y1, w, h = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            w = min(frame.shape[1] - x1, w)
            h = min(frame.shape[0] - y1, h)
            
            if w <= 0 or h <= 0:
                return None

            cropped = frame[y1:y1+h, x1:x1+w]
            if cropped.size == 0:
                return None

            person_dir = os.path.join(PERSON_DB_DIR, f"ID_{track_id}")
            os.makedirs(person_dir, exist_ok=True)

            if track_id not in self.db:
                self.db[track_id] = {'images': [], 'features': []}

            current_images = len(self.db[track_id]['images'])
            if current_images < self.max_images:
                img_path = os.path.join(person_dir, f"{current_images + 1}.jpg")
                cv2.imwrite(img_path, cropped)
                self.db[track_id]['images'].append(img_path)
                
                # 특징 즉시 추출 및 저장
                features = self.extract_features(cropped)
                if features is not None:
                    self.db[track_id]['features'].append(features.to(self.device))

                if len(self.db[track_id]['images']) >= self.max_images:
                    print(f"ID {track_id}의 이미지 수집 완료 (100장)")
                    self.is_collecting = False

            return cropped
        except Exception as e:
            print(f"이미지 저장 중 오류: {e}")
            return None

    def precompute_features(self):
        """��이터베이스의 모든 이미지 특징을 CUDA로 계산"""
        print("GPU로 특징 사전 계산 중...")
        with torch.no_grad():
            for track_id, data in self.db.items():
                data['features'] = []
                for img_path in data['images']:
                    img = cv2.imread(img_path)
                    if img is not None:
                        features = self.extract_features(img)
                        if features is not None:
                            # GPU 텐서로 저장
                            data['features'].append(features.to(self.device))

    def extract_features(self, image):
        """CUDA를 활용한 특징 추출"""
        try:
            if image is None or image.size == 0:
                return None

            # 이미지 전처리
            resized = cv2.resize(image, (128, 128))
            # BGR to RGB 및 정규화
            img_tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device) / 255.0

            # GPU에서 특징 추출
            with torch.no_grad():
                features = self.feature_extractor(img_tensor)
                features = features.view(features.size(0), -1)
                # L2 정규화
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                return features.squeeze()

        except Exception as e:
            print(f"특징 추출 중 오류: {e}")
            return None

    def find_matching_id(self, frame, bbox, current_id):
        """CUDA 기반 ID 매칭"""
        if not self.is_matching:
            return current_id

        try:
            x1, y1, w, h = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            w = min(frame.shape[1] - x1, w)
            h = min(frame.shape[0] - y1, h)
            
            if w <= 0 or h <= 0:
                return current_id

            current_img = frame[y1:y1+h, x1:x1+w]
            if current_img is None or current_img.size == 0:
                return current_id

            # 현재 이미지의 특징을 GPU에서 추출
            current_features = self.extract_features(current_img)
            if current_features is None:
                return current_id

            best_match_id = current_id
            best_match_score = float('inf')
            match_found = False

            # GPU에서 배치 처리로 비교
            for db_id, data in self.db.items():
                if len(data['features']) > 0:
                    # 모든 특징을 스택으로 쌓아서 한 번에 비교
                    db_features = torch.stack(data['features'])
                    # 코사인 유사도 계산 (GPU)
                    similarities = torch.nn.functional.cosine_similarity(
                        current_features.unsqueeze(0),
                        db_features
                    )
                    # 상위 3개 유사도 평균
                    top_similarities, _ = similarities.topk(min(3, len(similarities)))
                    avg_similarity = top_similarities.mean().item()
                    
                    # 유사도가 높을수록 좋음 (1에 가까울수록)
                    score = 1 - avg_similarity
                    
                    if score < best_match_score and score < self.matching_threshold:
                        best_match_score = score
                        best_match_id = db_id
                        match_found = True

            if match_found:
                print(f"ID 매칭: {current_id} -> {best_match_id} (점수: {best_match_score:.4f})")
            
            return best_match_id

        except Exception as e:
            print(f"ID 매칭 중 오류: {e}")
            return current_id

def process_detections(results, min_confidence=0.5):
    """YOLO 감지 결과 처리"""
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0 and conf > min_confidence:  # 사람 클래스
                w = x2 - x1
                h = y2 - y1
                if w * h > 200 and w/h < 2 and h/w < 3:  # 크기 및 비율 필터링
                    bbox = [x1, y1, w, h]
                    detections.append((bbox, conf, 0))
    return detections

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    person_db = PersonDB()
    print("'p' 키: 사진 수집 시작/중지")
    print("'l' 키: ID 매칭 시작/중지")
    print("'ESC' 키: 프로그램 종료")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                person_db.is_collecting = not person_db.is_collecting
                status = "시작" if person_db.is_collecting else "중지"
                print(f"사진 수집 {status}")
            elif key == ord('l'):
                person_db.is_matching = not person_db.is_matching
                status = "시작" if person_db.is_matching else "중지"
                print(f"ID 매칭 {status}")
            elif key == 27:  # ESC
                break

            # YOLO 감지 및 DeepSort 트래킹
            with torch.no_grad():
                preds = yolo_model(frame, verbose=False)
            detections = process_detections(preds)
            tracks = tracker.update_tracks(detections, frame=frame)

            # 트랙 처리
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = int(track.track_id)
                l, t, r, b = track.to_tlbr()
                bbox = [l, t, r - l, b - t]

                # 수집 모드일 때만 이미지 저장
                if person_db.is_collecting:
                    person_db.save_person_image(frame, bbox, track_id)

                # 매칭 모드일 때는 ID 매칭
                if person_db.is_matching:
                    track_id = person_db.find_matching_id(frame, bbox, track_id)

                # 박스 그리기
                color = ((track_id * 123) % 255, (track_id * 85) % 255, (track_id * 147) % 255)
                cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
                
                # 상태 표시
                saved_count = len(person_db.db.get(track_id, {'images': []})['images'])
                mode = "Collecting" if person_db.is_collecting else "Matching" if person_db.is_matching else "Tracking"
                label = f"ID: {track_id} ({saved_count}/100) - {mode}"
                cv2.putText(frame, label, (int(l), int(t) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("DeepSort Tracking", frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
