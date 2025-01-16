# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchreid.utils import FeatureExtractor
import mediapipe as mp
import time
import sys

# CUDA 사용 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

class PersonReID:
    def __init__(self):
        # YOLO 모델 초기화 (CUDA 사용)
        self.model = YOLO("yolo11x.pt")
        self.model.to(DEVICE)
        
        # DeepSORT 초기화
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            embedder='mobilenet',  # 기본 embedder 사용
            embedder_gpu=True if DEVICE == 'cuda' else False,
            half=True if DEVICE == 'cuda' else False
        )
        
        # ReID 모델 초기화
        self.reid_model = FeatureExtractor(
            model_name='osnet_x1_0',
            device=DEVICE
        )
        
        # 설정값
        self.conf_threshold = 0.5
        self.reid_threshold = 0.30  # 70% 유사도로 완화 (1 - 0.70)
        self.embeddings_file = "embeddings.json"
        self.embeddings_data = self.load_embeddings()
        
        # 가중치 조정
        self.weights = {
            'reid': 0.3,      # ReID 임베딩 (비중 감소)
            'position': 0.2,   # 위치 (비중 증가)
            'velocity': 0.2,   # 속도 (비중 증가)
            'color': 0.15,    # 컬러 히스토그램
            'pose': 0.15      # 관절 위치
        }
        
        # 위치 및 속도 추적을 위한 히스토리
        self.track_history = {}  # {track_id: {'positions': [], 'timestamps': [], 'colors': [], 'poses': []}}
        self.history_size = 30   # 프레임 히스토리 크기
        
        # MediaPipe pose 초기화
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def load_embeddings(self):
        """저장된 임베딩 로드"""
        try:
            with open(self.embeddings_file, "r") as f:
                data = json.load(f)
            print(f"임베딩 데이터 로드 완료: {len(data)} 개의 ID")
            return data
        except FileNotFoundError:
            print("새로운 임베딩 데이터베이스 시작")
            return {}

    def save_embeddings(self):
        """현재 임베딩 저장"""
        with open(self.embeddings_file, "w") as f:
            json.dump(self.embeddings_data, f, indent=4)
        print("임베딩 데이터 저장 완료")

    def process_frame(self, frame):
        """프레임 처리 및 객체 검출/추적"""
        # YOLO 검출
        results = self.model(frame, verbose=False)
        
        # DeepSORT 형식으로 변환
        detections = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if cls == 0 and conf > self.conf_threshold:  # 사람 클래스만
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, cls))

        # DeepSORT 추적
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks

    def extract_reid_features(self, image):
        """ReID 모델을 사용한 특징 추출"""
        try:
            if image is None or image.size == 0:
                return None

            # 이미지 전처리
            img = cv2.resize(image, (128, 256))  # ReID 모델 입력 크기
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 특징 추출
            features = self.reid_model(img)
            features = features.cpu().numpy().flatten()
            
            return features

        except Exception as e:
            print(f"ReID 특징 추출 오류: {e}")
            return None

    def save_person_embeddings(self, tracks, frame):
        """ReID 모델을 사용하여 현재 트랙들의 임베딩 저장"""
        try:
            saved_count = 0
            for track in tracks:
                if not track.is_confirmed():
                    continue

                try:
                    # 바운딩 박스 얻기
                    bbox = track.to_tlbr()
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # 유효한 좌표인지 확인
                    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                        continue
                    
                    # 이미지 크기가 너무 작은지 확인
                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue

                    # 사람 이미지 추출
                    person_img = frame[y1:y2, x1:x2]
                    if person_img is None or person_img.size == 0:
                        continue

                    # ReID 특징 추출
                    reid_features = self.extract_reid_features(person_img)
                    if reid_features is not None:
                        track_id = track.track_id
                        self.embeddings_data[f"person_{track_id}"] = {
                            "id": track_id,
                            "embedding": reid_features.tolist()
                        }
                        saved_count += 1
                        print(f"Features saved for ID {track_id}")

                except Exception as e:
                    print(f"Error processing track: {e}")
                    continue

            if saved_count > 0:
                self.save_embeddings()
                print(f"Total {saved_count} ReID features saved")
            else:
                print("No features available to save")

        except Exception as e:
            print(f"Error saving features: {e}")

    def update_track_history(self, track, frame):
        """트랙의 히스토리 정보 업데이트"""
        try:
            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            
            # 유효한 바운딩 박스 확인
            if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                return
            if x2 <= x1 or y2 <= y1:
                return
                
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            timestamp = time.time()

            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'positions': [],
                    'timestamps': [],
                    'colors': [],
                    'poses': [],
                    'velocities': []
                }

            history = self.track_history[track_id]
            
            # 위치 업데이트
            history['positions'].append(center)
            history['timestamps'].append(timestamp)
            
            # 속도 계산
            if len(history['positions']) >= 2:
                dt = history['timestamps'][-1] - history['timestamps'][-2]
                if dt > 0:
                    velocity = (history['positions'][-1] - history['positions'][-2]) / dt
                    history['velocities'].append(velocity)
            
            # 컬러 히스토그램
            person_img = frame[y1:y2, x1:x2]
            if person_img.size > 0:
                hist = cv2.calcHist([person_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                history['colors'].append(hist)
            
            # 관절 위치
            if person_img.size > 0 and person_img.shape[0] > 0 and person_img.shape[1] > 0:
                try:
                    person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                    pose_results = self.mp_pose.process(person_img_rgb)
                    if pose_results.pose_landmarks:
                        pose_points = np.array([[lm.x, lm.y] for lm in pose_results.pose_landmarks.landmark])
                        history['poses'].append(pose_points)
                except Exception as e:
                    pass  # 포즈 추출 실패는 무시

            # 히스토리 크기 제한
            for key in history:
                if len(history[key]) > self.history_size:
                    history[key] = history[key][-self.history_size:]

        except Exception as e:
            print(f"Track history update error: {e}")

    def calculate_similarity(self, track_id1, track_id2):
        """두 트랙 간의 종합 유사도 계산"""
        if track_id1 not in self.track_history or track_id2 not in self.track_history:
            return {'total': 0.0}

        hist1 = self.track_history[track_id1]
        hist2 = self.track_history[track_id2]
        similarities = {}

        try:
            # 1. 위치 유사도
            if hist1['positions'] and hist2['positions']:
                dist = np.linalg.norm(hist1['positions'][-1] - hist2['positions'][-1])
                similarities['position'] = float(np.exp(-dist / 100))

            # 2. 속도 유사도
            if hist1['velocities'] and hist2['velocities']:
                vel_diff = np.linalg.norm(hist1['velocities'][-1] - hist2['velocities'][-1])
                similarities['velocity'] = float(np.exp(-vel_diff / 10))

            # 3. 컬러 히스토그램 유사도
            if hist1['colors'] and hist2['colors']:
                color_sim = cv2.compareHist(
                    hist1['colors'][-1],
                    hist2['colors'][-1],
                    cv2.HISTCMP_CORREL
                )
                similarities['color'] = float((color_sim + 1) / 2)

            # 4. 포즈 유사도
            if hist1['poses'] and hist2['poses']:
                pose_dist = np.mean(np.linalg.norm(hist1['poses'][-1] - hist2['poses'][-1], axis=1))
                similarities['pose'] = float(np.exp(-pose_dist / 0.5))

            # 가중치 적용 및 종합 점수 계산
            total_score = 0
            total_weight = 0
            for key, score in similarities.items():
                if key in self.weights:
                    total_score += score * self.weights[key]
                    total_weight += self.weights[key]

            similarities['total'] = float(total_score / total_weight) if total_weight > 0 else 0.0
            return similarities

        except Exception as e:
            print(f"유사도 계산 오류: {e}")
            return {'total': 0.0}

    def find_matching_id(self, image, current_track):
        """ReID와 추가 특징을 결합한 ID 매칭"""
        try:
            # 1. ReID 특징 추출
            reid_features = self.extract_reid_features(image)
            if reid_features is None:
                return None, float('inf')

            best_match = None
            best_score = 0
            
            for person_id, data in self.embeddings_data.items():
                # ReID 유사도 계산
                saved_features = np.array(data['embedding'])
                reid_sim = np.dot(reid_features, saved_features) / (
                    np.linalg.norm(reid_features) * np.linalg.norm(saved_features)
                )
                reid_sim = (reid_sim + 1) / 2

                # 추가 특징 유사도 계산
                track_sim = self.calculate_similarity(
                    current_track.track_id,
                    data['id']
                )

                # 종합 점수 계산
                total_score = (
                    reid_sim * self.weights['reid'] +
                    track_sim * (1 - self.weights['reid'])
                )

                if total_score > best_score:
                    best_score = total_score
                    best_match = data['id']

            # 80% 이상의 종합 유사도를 가질 때만 매칭으로 인정
            if best_score >= 0.80:
                print(f"매칭 유사도: {best_score*100:.2f}%")
                return best_match, 1 - best_score
            return None, 1 - best_score

        except Exception as e:
            print(f"ID 매칭 오류: {e}")
            return None, float('inf')

    def run(self, video_source=0):
        """메인 실행 함수"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        print("\n=== Control Guide ===")
        print("p: Save ReID features for all current people")
        print("l: Toggle ReID matching mode")
        print("ESC: Exit\n")

        matching_mode = False
        last_key = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracks = self.process_frame(frame)
            active_tracks = [track for track in tracks if track.is_confirmed()]

            # 모든 트랙의 히스토리 업데이트
            for track in active_tracks:
                self.update_track_history(track, frame)

            # 트랙 표시 및 처리
            for track in active_tracks:
                bbox = track.to_tlbr()
                x1, y1, x2, y2 = map(int, bbox)
                track_id = track.track_id

                # 박스 색상 설정
                color = (0, 255, 0)
                label = f"ID: {track_id}"

                if matching_mode:
                    person_img = frame[y1:y2, x1:x2]
                    if person_img.size > 0:  # 이미지가 유효한 경우에만 처리
                        matched_id, distance = self.find_matching_id(person_img, track)
                        similarity = (1 - distance) * 100

                        if matched_id is not None:
                            color = (0, 255, 255)
                            label = f"ID: {track_id} -> {matched_id} ({similarity:.1f}%)"
                            
                            # 매칭 정보 출력
                            print(f"\n매칭 상세 정보 (ID {track_id} -> {matched_id}):")
                            similarities = self.calculate_similarity(track_id, matched_id)
                            for key, value in similarities.items():
                                if key != 'total':
                                    print(f"- {key}: {value*100:.1f}%")
                        else:
                            color = (0, 0, 255)
                            label = f"ID: {track_id} (New) ({similarity:.1f}%)"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('p') and key != last_key and active_tracks:
                print("P key detected: Starting feature save")
                self.save_person_embeddings(active_tracks, frame)
                
            elif key == ord('l') and key != last_key:
                matching_mode = not matching_mode
                status = "started" if matching_mode else "stopped"
                print(f"L key detected: ReID matching mode {status}")
                
            elif key == 27:  # ESC
                break

            last_key = key  # 현재 키 입력 저장

            # 상태 표시
            mode_text = "ReID Matching Mode" if matching_mode else "Normal Mode"
            cv2.putText(frame, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracks: {len(active_tracks)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Person ReID", frame)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    reid_system = PersonReID()
    reid_system.run()
