import cv2
import numpy as np
from ultralytics import YOLO

DEVICE = 'cuda'

class PoseDetector:
    def __init__(self):
        # YOLO-Pose 모델 초기화
        self.model = YOLO("yolo11l-pose.pt")
        self.model.to(DEVICE)
        
        # 키포인트 연결 정의 (관절 연결)
        self.skeleton = [
            [0, 1], [1, 3], [0, 2], [2, 4],  # 얼굴-귀-눈
            [5, 6],        # 어깨
            [5, 7], [7, 9], [6, 8], [8, 10],  # 팔
            [5, 11], [6, 12],  # 몸통
            [11, 13], [13, 15], [12, 14], [14, 16]  # 다리
        ]
        
        # 키포인트 색상 정의
        self.colors = {
            'joint': (0, 255, 0),    # 녹색: 관절
            'bone': (255, 255, 0),   # 청록색: 뼈대
            'text': (255, 255, 255)  # 흰색: 텍스트
        }

    def draw_pose(self, frame, keypoints, confidence):
        """포즈 시각화"""
        # keypoints data 가져오기
        kpts = keypoints.data.cpu().numpy()
        
        # 키포인트 그리기
        for idx in range(kpts.shape[1]):  # 17개의 키포인트
            x, y, conf = kpts[0, idx]  # 첫 번째 차원은 배치 차원
            if conf > confidence:  # 신뢰도가 높은 키포인트만 표시
                cv2.circle(frame, (int(x), int(y)), 4, self.colors['joint'], -1)
                # 키포인트 번호 표시
                cv2.putText(frame, str(idx), (int(x), int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        # 스켈레톤(뼈대) 그리기
        for start_idx, end_idx in self.skeleton:
            start_point = kpts[0, start_idx]
            end_point = kpts[0, end_idx]
            
            if start_point[2] > confidence and end_point[2] > confidence:
                cv2.line(frame,
                        (int(start_point[0]), int(start_point[1])),
                        (int(end_point[0]), int(end_point[1])),
                        self.colors['bone'], 2)

    def run(self, video_source=0):
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

            # YOLO-Pose로 포즈 검출
            results = self.model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                keypoints = result.keypoints
                
                # 각 검출된 사람에 대해
                for i in range(len(boxes)):
                    box = boxes[i]
                    kpts = keypoints[i]
                    
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    if conf > 0.5:  # 검출 신뢰도가 높은 경우만
                        # 바운딩 박스 그리기
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 포즈 그리기
                        self.draw_pose(frame, kpts, confidence=0.5)
                        
                        # 신뢰도가 높은 키포인트 수 계산
                        valid_kpts = sum(1 for conf in kpts.conf[0].cpu().numpy() if conf > 0.5)
                        
                        # 정보 표시
                        info_text = [
                            f"Person {i}",
                            f"Confidence: {conf:.2f}",
                            f"Valid Keypoints: {valid_kpts}/17"
                        ]
                        
                        for j, text in enumerate(info_text):
                            cv2.putText(frame, text, (x1, y1 - 10 - j*20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                      self.colors['text'], 2)

            # 안내 텍스트 표시
            cv2.putText(frame, f"Press ESC to exit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)

            cv2.imshow("YOLO Pose Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = PoseDetector()
    detector.run()
