import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import json
import torch
import torchreid
from torchreid.utils import FeatureExtractor
from pathlib import Path

DEVICE = 'cuda'

class PersonTracker:
    def __init__(self):
        # YOLO 모델 초기화
        self.model = YOLO("yolo11x.pt")
        self.model.to(DEVICE)
        
        # DeepSORT 초기화
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            embedder='mobilenet',
            embedder_gpu=True,
            half=True
        )
        
        # OSNet 초기화
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='',
            device=DEVICE
        )
        
        # 기본 설정
        self.conf_threshold = 0.6
        self.frame_count = 0
        self.reid_threshold = 0.4
        self.max_objects = 6
        
        # 데이터베이스 초기화
        self.embedding_db = {}      # 현재 세션의 임베딩 저장
        self.saved_ids = {}         # 저장된 ID와 임베딩
        self.current_matches = {}   # 현재 매칭된 ID 정보
        self.confirmed_ids = set()  # P 키로 저장된 고정 ID
        self.used_saved_ids = set() # 이미 매칭된 저장 ID 추적
        self.is_matching_mode = False
        self.perform_matching = False  # L 키를 눌렀을 때 한 번만 매칭 수행
        self.feature_cache = {}  # 특징 벡터 캐싱
        self.cache_max_size = 100  # 캐시 크기 제한
        
        # 성능 최적화를 위한 설정
        self.skip_frames = 1  # 프레임 스킵 수 조정
        self.process_counter = 0
        self.previous_tracks = []
        self.frame_buffer = None
        self.resize_factor = 0.5  # 프레임 크기 조정 비율
        
        # 배치 처리를 위한 설정
        self.batch_size = 4
        self.feature_batch = []
        
        # CUDA 최적화
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # 연산 제어를 위한 플래그 추가
        self.is_extracting = False  # P 버튼 눌렀을 때만 True
        self.is_matching = False    # L 버튼 눌렀을 때만 True

    def extract_features(self, image):
        """OSNet으로 특징 추출 - 캐싱 추가"""
        if image.size == 0:
            return None
            
        try:
            # 전처리 순서 최적화
            img = cv2.resize(image, (128, 256))  # 먼저 리사이즈
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            img = img.unsqueeze(0).div(255.0)
            img = img.to(DEVICE)
            
            with torch.no_grad():
                features = self.extractor(img)
                features = features.cpu().numpy().flatten()
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def calculate_similarity(self, feat1, feat2):
        """특징 벡터 간 유사도 계산"""
        if feat1 is None or feat2 is None:
            return 0.0
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

    def save_embeddings(self):
        """현재 추적 중인 사람들의 임베딩을 저장"""
        save_data = {
            'embeddings': []
        }
        
        # 현재 프레임의 모든 트랙에 대해 특징 추출 및 저장
        for track_id, features in self.embedding_db.items():
            save_data['embeddings'].append({
                'id': int(track_id),
                'features': features.tolist()
            })
            self.confirmed_ids.add(track_id)
        
        with open('embeddings.json', 'w') as f:
            json.dump(save_data, f)
        print(f"Embeddings saved for {len(save_data['embeddings'])} IDs")

    def load_embeddings(self):
        """저장된 임베딩을 로드하고 매칭 준비"""
        try:
            with open('embeddings.json', 'r') as f:
                data = json.load(f)
                
                self.saved_ids.clear()
                for item in data['embeddings']:
                    self.saved_ids[item['id']] = np.array(item['features'])
                
                print(f"Loaded {len(self.saved_ids)} saved IDs")
                self.perform_matching = True
                self.current_matches.clear()
                
        except FileNotFoundError:
            print("No saved embeddings found.")
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")

    def match_temporary_ids(self, tracks, frame):
        """임시 ID 매칭 최적화"""
        if not self.perform_matching:
            return

        print("Matching temporary IDs with most similar saved embeddings...")
        matched_count = 0

        # 1. 임시 ID만 수집 (초록색 제외)
        temp_features = {}
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            
            # 이미 매칭되었거나 고정된 ID는 완전히 제외
            if track_id in self.confirmed_ids or track_id in self.current_matches:
                continue

            features = self.get_cached_features(track_id, frame, track.to_tlbr())
            if features is not None:
                temp_features[track_id] = features

        if not temp_features:
            self.perform_matching = False
            return

        # 2. 각 임시 ID에 대해 가장 유사한 저장된 ID 찾기
        for temp_id, temp_feat in temp_features.items():
            best_match_id = None
            best_similarity = self.reid_threshold  # 0.4 미만은 매칭하지 않음

            # 저장된 모든 ID와 비교
            for saved_id, saved_feat in self.saved_ids.items():
                similarity = self.calculate_enhanced_similarity(temp_feat, saved_feat)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = saved_id

            # 매칭 결과가 있고 유사도가 0.4 이상인 경우만 적용
            if best_match_id is not None:
                self.current_matches[temp_id] = best_match_id
                matched_count += 1
                print(f"Matched: Temp ID {temp_id} -> Saved ID {best_match_id} (similarity: {best_similarity:.3f})")

        print(f"Matching completed: {matched_count} temporary IDs matched")
        self.perform_matching = False

    def process_frame(self, frame):
        """프레임 처리 최적화"""
        with torch.no_grad():
            results = self.model(frame, verbose=False)
        
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes.cpu().numpy()
            valid_boxes = boxes[(boxes.cls == 0) & (boxes.conf > self.conf_threshold)]
            
            # 최대 객체 수 제한 적용
            if len(valid_boxes) > self.max_objects:
                # 신뢰도가 높은 순으로 정렬하여 상위 6개만 선택
                conf_scores = valid_boxes.conf
                top_indices = np.argsort(conf_scores)[-self.max_objects:]
                valid_boxes = valid_boxes[top_indices]
            
            for box in valid_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], float(box.conf[0]), 0))

        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks

    def extract_features_batch(self, images):
        """배치 단위 특징 추출"""
        if not images:
            return []
            
        try:
            # 배치 전처리
            batch = []
            for img in images:
                if img.size == 0:
                    continue
                img = cv2.resize(img, (128, 256))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img.transpose(2, 0, 1)).float()
                img = img.div(255.0)
                batch.append(img)

            if not batch:
                return []

            # 배치 처리
            batch = torch.stack(batch).to(DEVICE)
            with torch.no_grad():
                features = self.extractor(batch)
                features = features.cpu().numpy()
            return features

        except Exception as e:
            print(f"Batch feature extraction error: {e}")
            return []

    def get_cached_features(self, track_id, frame, bbox):
        """캐시된 특징 벡터 반환 또는 새로 추출"""
        cache_key = f"{track_id}_{self.frame_count}"
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
            
        x1, y1, x2, y2 = map(int, bbox)
        person_img = frame[y1:y2, x1:x2]
        features = self.extract_features(person_img)
        
        # 캐시 크기 제한
        if len(self.feature_cache) > self.cache_max_size:
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]
            
        self.feature_cache[cache_key] = features
        return features

    def update_feature_buffer(self, track_id, features):
        """특징 벡터 버퍼 업데이트"""
        if track_id not in self.feature_buffer:
            self.feature_buffer[track_id] = []
        
        self.feature_buffer[track_id].append(features)
        if len(self.feature_buffer[track_id]) > self.buffer_size:
            self.feature_buffer[track_id].pop(0)

    def get_averaged_features(self, track_id):
        """버퍼에 있는 특징 벡터들의 평균 계산"""
        if track_id not in self.feature_buffer:
            return None
        
        features = self.feature_buffer[track_id]
        if not features:
            return None
            
        return np.mean(features, axis=0)

    def calculate_enhanced_similarity(self, feat1, feat2):
        """향상된 유사도 계산 - 벡터화 및 최적화"""
        if feat1 is None or feat2 is None:
            return 0.0
        
        # 벡터 정규화 미리 계산
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # 코사인 유사도
        cosine_sim = np.dot(feat1, feat2) / (norm1 * norm2)
        
        # 유클리디안 거리
        euclidean_sim = 1 / (1 + np.sum(np.square(feat1 - feat2)))
        
        return 0.7 * cosine_sim + 0.3 * euclidean_sim

    def find_best_match(self, features, current_track_id):
        """저장된 임베딩과 가장 유사한 ID 찾기"""
        best_match_id = None
        best_similarity = 0

        for saved_id, saved_features in self.saved_ids.items():
            # 이미 다른 트랙에 매칭된 ID는 건너뛰기
            if saved_id in self.used_saved_ids:
                continue

            similarity = self.calculate_enhanced_similarity(features, saved_features)
            if similarity > best_similarity and similarity > self.reid_threshold:
                best_similarity = similarity
                best_match_id = saved_id

        return best_match_id, best_similarity

    def display_frame(self, frame, tracks, fps):
        """화면 표시 로직"""
        for track in tracks:
            if not track.is_confirmed():
                continue

            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            track_id = track.track_id

            # P 버튼이 눌렸을 때만 특징 추출
            if self.is_extracting:
                person_img = frame[y1:y2, x1:x2]
                if person_img.size > 0:
                    features = self.extract_features(person_img)
                    if features is not None:
                        self.embedding_db[track_id] = features

            # ID 및 색상 결정
            color = (0, 0, 255)  # 기본 빨간색
            display_id = track_id

            if track_id in self.confirmed_ids:
                color = (0, 255, 0)  # 저장된 ID는 초록색
            elif track_id in self.current_matches:
                display_id = self.current_matches[track_id]
                color = (0, 255, 0)  # 매칭된 ID도 초록색

            # 바운딩 박스와 정보 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            info_text = f"ID: {display_id}"
            if track_id in self.confirmed_ids:
                info_text += " [Saved]"
            elif track_id in self.current_matches:
                info_text += " [Matched]"

            cv2.putText(frame, info_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 상태 표시
        status_text = [
            f"Tracks: {len(tracks)}",
            f"Fixed IDs: {len(self.confirmed_ids)}",
            f"Matched IDs: {len(self.current_matches)}",
            f"Extract Mode: {'On' if self.is_extracting else 'Off'}",
            f"Match Mode: {'On' if self.is_matching else 'Off'}",
            f"FPS: {fps}"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(frame, text, (10, 25 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Person Tracking", frame)

    def run(self, video_source=0):
        """실시간 추적 실행"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            return

        print("\n=== Control Guide ===")
        print("P: Save current embeddings (Start feature extraction)")
        print("L: Load saved embeddings (Start matching)")
        print("ESC: Exit\n")

        fps_start_time = time.time()
        fps_counter = 0
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            fps_counter += 1

            if time.time() - fps_start_time > 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()

            # 기본 추적 수행
            tracks = self.process_frame(frame)
            
            # L 버튼으로 매칭이 활성화된 경우에만 매칭 수행
            if self.is_matching and self.perform_matching:
                self.match_temporary_ids(tracks, frame)
            
            # 화면 표시
            self.display_frame(frame, tracks, fps)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('p'):  # P 키: 특징 추출 시작
                self.is_extracting = True
                self.save_embeddings()
                self.is_extracting = False
                print("Embeddings saved")
            elif key == ord('l'):  # L 키: 매칭 시작
                self.is_matching = True
                self.load_embeddings()
                print("Matching mode activated")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = PersonTracker()
    tracker.run()
