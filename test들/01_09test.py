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
        self.conf_threshold = 0.5
        self.frame_count = 0
        self.similarity_threshold = 0.6
        self.reid_threshold = 0.7  # ReID 유사도 임계값
        
        # 데이터베이스 초기화
        self.person_database = {}  # 실시간 트래킹 DB
        self.embedding_database = self.load_embedding_db()  # 임베딩 DB
        
        # 히스토그램 설정
        self.hist_bins = [8, 8, 8]
        self.hist_ranges = [[0, 180], [0, 256], [0, 256]]
        
        # 영역별 가중치
        self.color_weights = {
            'face': 0.4,
            'upper': 0.35,
            'lower': 0.25
        }

    def load_embedding_db(self):
        """JSON 파일에서 임베딩 데이터베이스 로드"""
        db_path = Path('embedding_db.json')
        if db_path.exists():
            with open(db_path, 'r') as f:
                return json.load(f)
        return {}

    def save_embedding_db(self):
        """임베딩 데이터베이스를 JSON 파일로 저장"""
        with open('embedding_db.json', 'w') as f:
            json.dump(self.embedding_database, f)

    def extract_features(self, image):
        """OSNet을 사용하여 이미지에서 특징 추출"""
        if image.size == 0:
            return None
            
        try:
            # 이미지 전처리
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 256))  # OSNet 입력 크기
            
            # [H, W, C] -> [C, H, W] 변환
            img = img.transpose(2, 0, 1)
            
            # 정규화 (0-1 범위로)
            img = img / 255.0
            
            # numpy -> torch 변환 및 배치 차원 추가
            img = torch.from_numpy(img).float()
            img = img.unsqueeze(0)  # 배치 차원 추가
            img = img.to(DEVICE)
            
            # 특징 추출
            with torch.no_grad():
                features = self.extractor(img)
                features = features.cpu().numpy().flatten()  # 1차원 벡터로 변환
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def calculate_reid_similarity(self, feat1, feat2):
        """두 특징 벡터 간의 유사도 계산"""
        if feat1 is None or feat2 is None:
            return 0.0
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

    def find_best_match(self, current_hist, current_features):
        """히스토그램과 ReID 특징을 모두 사용하여 최적의 매칭 찾기"""
        best_match_id = None
        best_score = -1

        # 실시간 트래킹 DB 검사
        for track_id, data in self.person_database.items():
            # 히스토그램 유사도
            hist_similarity = self.compare_histograms(current_hist, data['hist'])
            
            # ReID 유사도
            reid_similarity = self.calculate_reid_similarity(
                current_features, 
                data.get('features', None)
            )
            
            # 종합 점수 (히스토그램 40%, ReID 60%)
            total_score = hist_similarity * 0.4 + reid_similarity * 0.6
            
            if total_score > best_score and total_score > self.similarity_threshold:
                best_score = total_score
                best_match_id = track_id

        # 임베딩 DB 검사 (실시간 DB에서 매칭이 없는 경우)
        if best_match_id is None:
            for person_id, embeddings in self.embedding_database.items():
                max_similarity = 0
                for embedding in embeddings:
                    similarity = self.calculate_reid_similarity(
                        current_features, 
                        np.array(embedding)
                    )
                    max_similarity = max(max_similarity, similarity)
                
                if max_similarity > self.reid_threshold:
                    best_match_id = person_id
                    best_score = max_similarity
                    break

        return best_match_id, best_score

    def update_database(self, track_id, hist, features, frame):
        """데이터베이스 업데이트"""
        # 실시간 트래킹 DB 업데이트
        if track_id not in self.person_database:
            self.person_database[track_id] = {
                'hist': hist,
                'features': features,
                'last_seen': self.frame_count
            }
        
        # 주기적으로 임베딩 DB 업데이트 (예: 100프레임마다)
        if self.frame_count % 100 == 0:
            if str(track_id) not in self.embedding_database:
                self.embedding_database[str(track_id)] = []
            self.embedding_database[str(track_id)].append(features.tolist())
            self.save_embedding_db()

    def calculate_histogram(self, image):
        """영역별 HSV 컬러 히스토그램 계산"""
        if image.size == 0:
            return None
            
        # HSV 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 영역 분할
        height = image.shape[0]
        face_region = hsv[:height//4]           # 상위 25%는 얼굴
        upper_region = hsv[height//4:height//2]  # 25~50%는 상체
        lower_region = hsv[height//2:]           # 50~100%는 하체
        
        # 각 영역별 히스토그램 계산
        hist_face = cv2.calcHist([face_region], [0, 1, 2], None, 
                                self.hist_bins, 
                                [0, 180, 0, 256, 0, 256])
        hist_upper = cv2.calcHist([upper_region], [0, 1, 2], None, 
                                 self.hist_bins, 
                                 [0, 180, 0, 256, 0, 256])
        hist_lower = cv2.calcHist([lower_region], [0, 1, 2], None, 
                                 self.hist_bins, 
                                 [0, 180, 0, 256, 0, 256])
        
        # 정규화
        cv2.normalize(hist_face, hist_face, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_upper, hist_upper, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_lower, hist_lower, 0, 1, cv2.NORM_MINMAX)
        
        return {
            'face': hist_face,
            'upper': hist_upper,
            'lower': hist_lower
        }
#ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ ㅇㅠㄴ준서바보 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 방문록 - 정수현
    def compare_histograms(self, hist1, hist2):
        """영역별 히스토그램 비교"""
        if hist1 is None or hist2 is None:
            return 0.0
            
        # 각 영역별 유사도 계산
        similarity_face = cv2.compareHist(hist1['face'], hist2['face'], cv2.HISTCMP_CORREL)
        similarity_upper = cv2.compareHist(hist1['upper'], hist2['upper'], cv2.HISTCMP_CORREL)
        similarity_lower = cv2.compareHist(hist1['lower'], hist2['lower'], cv2.HISTCMP_CORREL)
        
        # 가중치 적용한 종합 점수
        weighted_similarity = (
            similarity_face * self.color_weights['face'] +
            similarity_upper * self.color_weights['upper'] +
            similarity_lower * self.color_weights['lower']
        )
        
        return max(0, weighted_similarity)

    def process_frame(self, frame):
        """프레임 처리 및 객체 검출/추적"""
        # YOLO로 사람 검출
        results = self.model(frame, verbose=False)
        
        # DeepSORT 형식으로 변환
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # 사람 클래스(0)만 처리
                if cls == 0 and conf > self.conf_threshold:
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(([x1, y1, w, h], conf, cls))

        # DeepSORT로 추적
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks

    def run(self, video_source=0):
        """실시간 추적 실행"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        print("\n=== Control Guide ===")
        print("ESC: Exit\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            tracks = self.process_frame(frame)
            
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

                # 히스토그램과 ReID 특징 추출
                hist = self.calculate_histogram(person_img)
                features = self.extract_features(person_img)
                
                if hist is None or features is None:
                    continue

                # 데이터베이스 업데이트
                self.update_database(track_id, hist, features, frame)

                # 매칭 수행
                best_match_id, similarity_score = self.find_best_match(hist, features)

                # 바운딩 박스 색상 결정
                if similarity_score > 0.8:
                    box_color = (0, 255, 0)  # 높은 유사도: 녹색
                elif similarity_score > 0.6:
                    box_color = (0, 255, 255)  # 중간 유사도: 노란색
                else:
                    box_color = (0, 0, 255)  # 낮은 유사도: 빨간색

                # 바운딩 박스와 영역 구분선 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                face_y = y1 + (y2-y1)//4
                upper_y = y1 + (y2-y1)//2
                cv2.line(frame, (x1, face_y), (x2, face_y), (0, 255, 255), 1)
                cv2.line(frame, (x1, upper_y), (x2, upper_y), (0, 255, 255), 1)

                # 정보 표시
                info_lines = [
                    f"ID:{track_id} Match:{best_match_id} [{similarity_score:.2f}]",
                    f"ReID: {self.calculate_reid_similarity(features, self.person_database.get(track_id, {}).get('features', None)):.2f}",
                    f"Hist: {self.compare_histograms(hist, self.person_database.get(track_id, {}).get('hist', None)):.2f}"
                ]

                # 텍스트 표시
                for i, text in enumerate(info_lines):
                    y = y1 - 5 - (len(info_lines) - 1 - i) * 15
                    cv2.putText(frame, text, (x1, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, text, (x1, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 전체 상태 표시
            status_text = [
                f"Tracks: {len([t for t in tracks if t.is_confirmed()])}",
                f"Frame: {self.frame_count}",
                f"DB Size: {len(self.embedding_database)}"
            ]
            
            for i, text in enumerate(status_text):
                cv2.putText(frame, text, (10, 25 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Person Tracking with ReID", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = PersonTracker()
    tracker.run()
