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
        self.reid_threshold = 0.3
        
        # 데이터베이스 초기화
        self.embedding_db = {}      # 현재 세션의 임베딩 저장
        self.saved_ids = {}         # 저장된 ID와 임베딩
        self.current_matches = {}   # 현재 매칭된 ID 정보
        self.confirmed_ids = set()  # P 키로 저장된 고정 ID
        self.used_saved_ids = set() # 이미 매칭된 저장 ID 추적
        self.is_matching_mode = False
        self.perform_matching = False  # L 키를 눌렀을 때 한 번만 매칭 수행

    def extract_features(self, image):
        """OSNet으로 특징 추출"""
        if image.size == 0:
            return None
            
        try:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 256))
            img = img.transpose(2, 0, 1)
            img = img / 255.0
            img = torch.from_numpy(img).float()
            img = img.unsqueeze(0)
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
        """저장된 임베딩을 로드하고 한 번의 매칭 수행"""
        try:
            with open('embeddings.json', 'r') as f:
                data = json.load(f)
                
                self.saved_ids.clear()
                for item in data['embeddings']:
                    self.saved_ids[item['id']] = np.array(item['features'])
                
                print(f"Loaded {len(self.saved_ids)} saved IDs")
                self.perform_matching = True  # 매칭 수행 플래그 설정
                self.current_matches.clear()  # 매칭 정보 초기화
                self.used_saved_ids.clear()  # 사용된 ID 초기화
                
        except FileNotFoundError:
            print("No saved embeddings found.")
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")

    def match_temporary_ids(self, tracks, frame):
        """임시 ID(빨간색)를 가진 객체들을 가장 유사도가 높은 저장된 ID로 매칭"""
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
                print(f"Skipping ID {track_id} (already matched/fixed)")
                continue

            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            person_img = frame[y1:y2, x1:x2]
            
            if person_img.size == 0:
                continue

            features = self.extract_features(person_img)
            if features is not None:
                temp_features[track_id] = features
                print(f"Processing temporary ID: {track_id}")

        if not temp_features:
            print("No temporary IDs found for matching")
            self.perform_matching = False
            return

        # 2. 모든 임시 ID와 저장된 ID 간의 유사도 계산
        all_similarities = []  # [(similarity, temp_id, saved_id)]
        for temp_id, temp_feat in temp_features.items():
            for saved_id, saved_feat in self.saved_ids.items():
                # 이미 매칭된 ID는 제외
                if saved_id in [v for v in self.current_matches.values()]:
                    continue
                similarity = self.calculate_enhanced_similarity(temp_feat, saved_feat)
                all_similarities.append((similarity, temp_id, saved_id))

        # 3. 유사도 기준으로 정렬 (높은 순)
        all_similarities.sort(reverse=True)

        # 4. 매칭 수행
        matched_temp_ids = set()
        used_saved_ids = set()

        for similarity, temp_id, saved_id in all_similarities:
            # 이미 매칭된 임시 ID나 저장 ID는 건너뛰기
            if temp_id in matched_temp_ids or saved_id in used_saved_ids:
                continue

            # 매칭 적용
            self.current_matches[temp_id] = saved_id
            matched_temp_ids.add(temp_id)
            used_saved_ids.add(saved_id)
            matched_count += 1
            print(f"Matched: Temp ID {temp_id} -> Saved ID {saved_id} (similarity: {similarity:.3f})")

        print(f"Matching completed: {matched_count} temporary IDs matched")
        self.perform_matching = False

    def process_frame(self, frame):
        """프레임 처리 및 객체 검출/추적"""
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if cls == 0 and conf > self.conf_threshold:
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(([x1, y1, w, h], conf, cls))

        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks

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
        """향상된 유사도 계산"""
        if feat1 is None or feat2 is None:
            return 0.0
        
        # 코사인 유사도
        cosine_sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        
        # 유클리디안 거리 기반 유사도
        euclidean_dist = np.linalg.norm(feat1 - feat2)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # 가중 평균 (코사인 유사도에 더 높은 가중치)
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

    def run(self, video_source=0):
        """실시간 추적 실행"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        print("\n=== Control Guide ===")
        print("P: Save current embeddings")
        print("L: Load saved embeddings")
        print("ESC: Exit\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            tracks = self.process_frame(frame)
            
            # L 키로 매칭이 요청된 경우에만 매칭 수행
            if self.perform_matching:
                self.match_temporary_ids(tracks, frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue

                bbox = track.to_tlbr()
                x1, y1, x2, y2 = map(int, bbox)
                track_id = track.track_id

                # 사람 영역 추출
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue

                # 특징 추출
                features = self.extract_features(person_img)
                if features is None:
                    continue

                # 현재 특징 저장 (P 키를 위한 저장)
                self.embedding_db[track_id] = features

                # ID 및 색상 결정
                color = (0, 0, 255)  # 기본 빨간색 (임시 ID)
                display_id = track_id

                # 확정된 ID(초록색) 확인
                if track_id in self.confirmed_ids:
                    color = (0, 255, 0)  # 저장된 고정 ID는 초록색
                    display_id = track_id  # 원래 ID 유지
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
                f"Matching Ready: {'Yes' if self.perform_matching else 'No'}"
            ]
            
            for i, text in enumerate(status_text):
                cv2.putText(frame, text, (10, 25 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Person Tracking", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('p'):  # P 키: 현재 임베딩 저장
                self.save_embeddings()
                print("Current embeddings saved")
            elif key == ord('l'):  # L 키: 저장된 임베딩 로드 및 매칭 시작
                self.load_embeddings()
                print("Matching mode activated")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = PersonTracker()
    tracker.run()
