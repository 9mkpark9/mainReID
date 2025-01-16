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

            # 객체 검출 및 추적
            tracks = self.process_frame(frame)
            
            # 추적 결과 시각화
            for track in tracks:
                if not track.is_confirmed():
                    continue

                # 바운딩 박스 그리기
                bbox = track.to_tlbr()
                x1, y1, x2, y2 = map(int, bbox)
                track_id = track.track_id
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 추적 중인 객체 수 표시
            active_tracks = len([t for t in tracks if t.is_confirmed()])
            cv2.putText(frame, f"Tracks: {active_tracks}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Person Tracking", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = PersonTracker()
    tracker.run()
