import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# CUDA 사용 설정
DEVICE = 'cuda'

class PersonTracker:
    def __init__(self):
        # YOLO 모델 초기화
        self.model = YOLO("yolo11x.pt")
        self.model.to(DEVICE)
        
        # DeepSORT 초기화
        self.tracker = DeepSort(
            max_age=30,        # 트랙 유지 최대 프레임
            n_init=3,          # 트랙 초기화에 필요한 프레임
            nn_budget=100,     # 트랙 히스토리 크기
            embedder='mobilenet',
            embedder_gpu=True,
            half=True
        )
        
        self.conf_threshold = 0.5  # 검출 신뢰도 임계값
        self.person_database = {}  # {track_id: {'hist': histogram, 'last_seen': frame_count}}
        self.frame_count = 0
        self.similarity_threshold = 0.6  # 히스토그램 유사도 임계값
        
        # 히스토그램 설정 수정
        self.hist_bins = [8, 8, 8]  # H, S, V 각각 8개 bins
        self.hist_ranges = [
            [0, 180],  # H range
            [0, 256],  # S range
            [0, 256]   # V range
        ]
        
        # 영역별 가중치 수정
        self.color_weights = {
            'face': 0.4,     # 얼굴 영역
            'upper': 0.35,   # 상체 영역
            'lower': 0.25    # 하체 영역
        }

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

    def find_best_match(self, current_hist):
        """개선된 매칭 로직"""
        best_match_id = None
        best_similarity = -1

        for track_id, data in self.person_database.items():
            similarity = self.compare_histograms(current_hist, data['hist'])
            
            # 디버깅 정보 출력
            if similarity > 0.4:  # 의미있는 유사도만 출력
                print(f"ID:{track_id} Similarity:{similarity:.3f}")
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = track_id

        return best_match_id, best_similarity

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

                # 사람 영역 추출 및 히스토그램 계산
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue
                hist = self.calculate_histogram(person_img)
                if hist is None:
                    continue

                # 데이터베이스 업데이트
                if track_id not in self.person_database:
                    self.person_database[track_id] = {
                        'hist': hist,
                        'last_seen': self.frame_count
                    }

                # 바운딩 박스 그리기
                similarity = 0
                if track_id in self.person_database:
                    similarity = self.compare_histograms(hist, self.person_database[track_id]['hist'])
                    # 유사도에 따른 박스 색상 변경
                    if similarity > 0.8:
                        box_color = (0, 255, 0)  # 높은 유사도: 녹색
                    elif similarity > 0.6:
                        box_color = (0, 255, 255)  # 중간 유사도: 노란색
                    else:
                        box_color = (0, 0, 255)  # 낮은 유사도: 빨간색
                else:
                    box_color = (255, 255, 255)  # 새로운 객체: 흰색

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # 영역 구분선 표시
                face_y = y1 + (y2-y1)//4
                upper_y = y1 + (y2-y1)//2
                cv2.line(frame, (x1, face_y), (x2, face_y), (0, 255, 255), 1)
                cv2.line(frame, (x1, upper_y), (x2, upper_y), (0, 255, 255), 1)

                # HSV 정보 계산
                height = person_img.shape[0]
                face_hsv = cv2.cvtColor(person_img[:height//4], cv2.COLOR_BGR2HSV)
                upper_hsv = cv2.cvtColor(person_img[height//4:height//2], cv2.COLOR_BGR2HSV)
                lower_hsv = cv2.cvtColor(person_img[height//2:], cv2.COLOR_BGR2HSV)

                # 각 영역의 평균 HSV 값과 표준편차 계산
                face_mean = np.mean(face_hsv, axis=(0,1)).astype(int)
                upper_mean = np.mean(upper_hsv, axis=(0,1)).astype(int)
                lower_mean = np.mean(lower_hsv, axis=(0,1)).astype(int)

                face_std = np.std(face_hsv, axis=(0,1)).astype(int)
                upper_std = np.std(upper_hsv, axis=(0,1)).astype(int)
                lower_std = np.std(lower_hsv, axis=(0,1)).astype(int)

                # 정보 표시
                info_lines = [
                    f"ID:{track_id} [{similarity:.2f}]",
                    f"Face HSV: ({face_mean[0]},{face_mean[1]},{face_mean[2]}) σ({face_std[0]},{face_std[1]},{face_std[2]})",
                    f"Upper HSV: ({upper_mean[0]},{upper_mean[1]},{upper_mean[2]}) σ({upper_std[0]},{upper_std[1]},{upper_std[2]})",
                    f"Lower HSV: ({lower_mean[0]},{lower_mean[1]},{lower_mean[2]}) σ({lower_std[0]},{lower_std[1]},{lower_std[2]})",
                    f"Weights: F({self.color_weights['face']:.1f}) U({self.color_weights['upper']:.1f}) L({self.color_weights['lower']:.1f})"
                ]

                # 텍스트 표시
                for i, text in enumerate(info_lines):
                    y = y1 - 5 - (len(info_lines) - 1 - i) * 15
                    # 외곽선 효과로 가독성 확보
                    cv2.putText(frame, text, (x1, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, text, (x1, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 전체 상태 표시
            status_text = [
                f"Tracks: {len([t for t in tracks if t.is_confirmed()])}",
                f"Frame: {self.frame_count}",
                f"Threshold: {self.similarity_threshold:.2f}"
            ]
            
            for i, text in enumerate(status_text):
                cv2.putText(frame, text, (10, 25 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Person Tracking with Color Analysis", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = PersonTracker()
    tracker.run()
