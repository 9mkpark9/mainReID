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
        self.db = {}  # {track_id: {'images': [], 'features': [], 'last_pos': None, 'last_seen': None}}
        self.max_images = 100
        self.is_collecting = False
        self.is_matching = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 얼굴 검출기
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 특징 추출기 (CUDA)
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        ).to(self.device)
        
        # 특징 캐시
        self.feature_cache = {}
        self.frame_count = 0
        
        self.load_db()
        self.precompute_features()
        print(f"로드된 ID 수: {len(self.db)}")

    def precompute_features(self):
        """데이터베이스 이미지의 특징을 미리 계산"""
        print("특징 계산 중...")
        for track_id, data in self.db.items():
            data['features'] = []
            for img_path in data['images']:
                img = cv2.imread(img_path)
                if img is not None:
                    features = self.extract_features(img)
                    if features is not None:
                        data['features'].append(features)

    def extract_features(self, image):
        """��적화된 특징 추출"""
        try:
            if image is None or image.size == 0:
                return None, None

            # 이미지 해시로 캐시 확인
            img_hash = hash(image.tobytes())
            if img_hash in self.feature_cache:
                return self.feature_cache[img_hash]

            # 1. 얼굴 특징
            face_features = None
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                face_img = image[y:y+h, x:x+w]
                if face_img.size > 0:
                    face_tensor = torch.from_numpy(cv2.resize(face_img, (64, 64))).float()
                    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
                    with torch.no_grad():
                        face_features = self.feature_extractor(face_tensor).squeeze()

            # 2. 옷 특징
            h, w = image.shape[:2]
            upper_body = image[h//4:3*h//4, :]
            
            if upper_body.size > 0:
                # HSV 색상 히스토그램
                hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                # 텍스처 특징
                gray_upper = cv2.cvtColor(upper_body, cv2.COLOR_BGR2GRAY)
                lbp = self.compute_lbp(gray_upper)  # LBP 특징
                
                cloth_features = torch.cat([
                    torch.from_numpy(hist).float().to(self.device),
                    torch.from_numpy(lbp).float().to(self.device)
                ])
            else:
                cloth_features = None

            # 결과 캐싱
            features = (face_features, cloth_features)
            self.feature_cache[img_hash] = features
            
            # 캐시 크기 제한
            if len(self.feature_cache) > 1000:
                self.feature_cache.clear()
                
            return features

        except Exception as e:
            print(f"특징 추출 오류: {e}")
            return None, None

    def compute_lbp(self, image):
        """LBP 텍스처 특징 계산"""
        lbp = np.zeros_like(image)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] > center) << 7
                code |= (image[i-1, j] > center) << 6
                code |= (image[i-1, j+1] > center) << 5
                code |= (image[i, j+1] > center) << 4
                code |= (image[i+1, j+1] > center) << 3
                code |= (image[i+1, j] > center) << 2
                code |= (image[i+1, j-1] > center) << 1
                code |= (image[i, j-1] > center) << 0
                lbp[i, j] = code
        
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        return cv2.normalize(hist, hist).flatten()

    def find_matching_id(self, frame, bbox, current_id):
        """최적화된 ID 매칭"""
        if not self.is_matching:
            return current_id

        try:
            self.frame_count += 1
            x1, y1, w, h = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            w = min(frame.shape[1] - x1, w)
            h = min(frame.shape[0] - y1, h)
            
            if w <= 0 or h <= 0:
                return current_id

            current_img = frame[y1:y1+h, x1:x1+w]
            if current_img is None or current_img.size == 0:
                return current_id

            current_face, current_cloth = self.extract_features(current_img)
            current_pos = np.array([x1 + w/2, y1 + h/2])

            # 매칭 점수 계산을 위한 텐서 준비
            scores = []
            candidates = []

            for db_id, data in self.db.items():
                if len(data['features']) > 0:
                    # 시간 기반 필터링
                    if data['last_seen'] is not None:
                        time_diff = self.frame_count - data['last_seen']
                        if time_diff > 300:  # 10초 이상 안 보인 ID는 제외 (30fps 기준)
                            continue

                    # 위치 기반 필터링
                    if data['last_pos'] is not None:
                        dist = np.linalg.norm(current_pos - data['last_pos'])
                        if dist > 300:  # 너무 멀리 있는 ID는 제외
                            continue

                    # 특징 비교
                    for face_feat, cloth_feat in data['features'][-5:]:  # 최근 5개만 사용
                        score = 0
                        count = 0

                        if current_face is not None and face_feat is not None:
                            face_sim = torch.nn.functional.cosine_similarity(
                                current_face.unsqueeze(0),
                                face_feat.unsqueeze(0)
                            ).item()
                            score += (1 - face_sim) * 0.6  # 얼굴 가중치 증가
                            count += 1

                        if current_cloth is not None and cloth_feat is not None:
                            cloth_sim = torch.nn.functional.cosine_similarity(
                                current_cloth.unsqueeze(0),
                                cloth_feat.unsqueeze(0)
                            ).item()
                            score += (1 - cloth_sim) * 0.4
                            count += 1

                        if count > 0:
                            scores.append(score / count)
                            candidates.append(db_id)

            if scores:
                best_idx = np.argmin(scores)
                best_score = scores[best_idx]
                
                if best_score < 0.3:  # 더 엄격한 임계값
                    best_match_id = candidates[best_idx]
                    print(f"ID 매칭: {current_id} -> {best_match_id} (점수: {best_score:.4f})")
                    
                    # 매칭된 ID 정보 업데이트
                    self.db[best_match_id]['last_pos'] = current_pos
                    self.db[best_match_id]['last_seen'] = self.frame_count
                    return best_match_id

            return current_id

        except Exception as e:
            print(f"ID 매칭 오류: {e}")
            return current_id

    def save_person_image(self, frame, bbox, track_id):
        """이미지 저장"""
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
                self.db[track_id] = {'images': [], 'features': [], 'last_pos': None, 'last_seen': None}

            current_images = len(self.db[track_id]['images'])
            if current_images < self.max_images:
                img_path = os.path.join(person_dir, f"{current_images + 1}.jpg")
                cv2.imwrite(img_path, cropped)
                self.db[track_id]['images'].append(img_path)
                
                # 특징 즉시 추출
                features = self.extract_features(cropped)
                if features is not None:
                    self.db[track_id]['features'].append(features)

                if len(self.db[track_id]['images']) >= self.max_images:
                    print(f"ID {track_id}의 이미지 수집 완료 (100장)")
                    self.is_collecting = False

            return cropped
        except Exception as e:
            print(f"이미지 저장 오류: {e}")
            return None

    def load_db(self):
        """기존 데이터베이스 로드"""
        for person_dir in os.listdir(PERSON_DB_DIR):
            if person_dir.startswith('ID_'):
                track_id = int(person_dir.split('_')[1])
                self.db[track_id] = {'images': [], 'features': [], 'last_pos': None, 'last_seen': None}
                person_path = os.path.join(PERSON_DB_DIR, person_dir)
                for img_file in os.listdir(person_path):
                    if img_file.endswith('.jpg'):
                        img_path = os.path.join(person_path, img_file)
                        self.db[track_id]['images'].append(img_path)

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
