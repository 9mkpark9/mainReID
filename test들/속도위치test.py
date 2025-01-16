import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

DEVICE = 'cuda'

class PersonTracker:
    def __init__(self):
        self.model = YOLO("yolo11x.pt")
        self.model.to(DEVICE)
        
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            embedder='mobilenet',
            embedder_gpu=True,
            half=True
        )
        
        self.conf_threshold = 0.5
        self.person_database = {}  # {track_id: {'position': [], 'velocity': [], 'last_seen': frame_count}}
        self.frame_count = 0
        self.last_time = time.time()
        
        # 매칭 임계값
        self.position_threshold = 150  # 픽셀 단위
        self.velocity_threshold = 50   # 픽셀/초 단위
        self.time_window = 30         # 프레임 수
        
        # 움직임 감지를 위한 임계값 추가
        self.min_movement_threshold = 3.0    # 최소 이동 거리 (픽셀)
        self.min_speed_threshold = 5.0       # 최소 속도 (픽셀/초)
        self.movement_smoothing = 3          # 이동 평균을 위한 프레임 수
        
        # 매칭 가중치 및 임계값 수정
        self.position_weight = 0.4    # 위치 가중치
        self.velocity_weight = 0.3    # 속도 가중치
        self.direction_weight = 0.3   # 이동 방향 가중치
        
        self.direction_threshold = 45.0  # 방향 차이 허용 각도 (도)
        self.velocity_similarity_threshold = 0.7  # 속도 유사도 임계값

    def calculate_velocity(self, positions, timestamps):
        """속도 계산 (픽셀/초) - 노이즈 필터링 추가"""
        if len(positions) < 2:
            return np.array([0, 0])
        
        dt = timestamps[-1] - timestamps[-2]
        if dt == 0:
            return np.array([0, 0])
        
        # 최근 N개 프레임의 평균 속도 계산
        n = min(self.movement_smoothing, len(positions))
        if n >= 2:
            recent_positions = positions[-n:]
            recent_timestamps = timestamps[-n:]
            total_displacement = recent_positions[-1] - recent_positions[0]
            total_time = recent_timestamps[-1] - recent_timestamps[0]
            velocity = total_displacement / total_time if total_time > 0 else np.array([0, 0])
        else:
            velocity = (positions[-1] - positions[-2]) / dt
            
        # 노이즈 필터링
        speed = np.linalg.norm(velocity)
        if speed < self.min_speed_threshold:
            velocity = np.array([0, 0])
            
        return velocity

    def is_actually_moving(self, positions, velocities):
        """실제 움직임 여부 판단"""
        if len(positions) < 2:
            return False
            
        # 최근 위치 변화 확인
        recent_movement = np.linalg.norm(positions[-1] - positions[-2])
        
        # 최근 속도 확인
        recent_speed = np.linalg.norm(velocities[-1]) if len(velocities) > 0 else 0
        
        # 둘 다 임계값을 넘을 때만 움직임으로 판단
        return (recent_movement > self.min_movement_threshold and 
                recent_speed > self.min_speed_threshold)

    def update_track_info(self, track_id, position, current_time):
        """트랙 정보 업데이트"""
        if track_id not in self.person_database:
            self.person_database[track_id] = {
                'positions': [],
                'timestamps': [],
                'velocities': [],
                'last_seen': self.frame_count
            }
        
        data = self.person_database[track_id]
        data['positions'].append(position)
        data['timestamps'].append(current_time)
        
        # 속도 계산 및 저장
        velocity = self.calculate_velocity(data['positions'], data['timestamps'])
        data['velocities'].append(velocity)
        data['last_seen'] = self.frame_count
        
        # 윈도우 크기 제한
        if len(data['positions']) > self.time_window:
            data['positions'] = data['positions'][-self.time_window:]
            data['timestamps'] = data['timestamps'][-self.time_window:]
            data['velocities'] = data['velocities'][-self.time_window:]

    def calculate_direction_similarity(self, direction1, direction2):
        """두 방향 간의 유사도 계산 (0~1)"""
        angle_diff = abs(direction1 - direction2)
        # 각도 차이를 0-180도 범위로 정규화
        angle_diff = min(angle_diff, 360 - angle_diff)
        # 각도 차이를 유사도로 변환 (180도 차이 -> 0, 0도 차이 -> 1)
        return max(0, 1 - angle_diff / 180.0)

    def calculate_velocity_similarity(self, vel1, vel2):
        """두 속도 벡터 간의 유사도 계산"""
        speed1 = np.linalg.norm(vel1)
        speed2 = np.linalg.norm(vel2)
        
        # 속도가 너무 작으면 비교하지 않음
        if speed1 < self.min_speed_threshold or speed2 < self.min_speed_threshold:
            return 0.0
            
        # 속도 크기의 유사도
        speed_ratio = min(speed1, speed2) / max(speed1, speed2)
        
        # 방향 유사도
        direction1 = np.arctan2(vel1[1], vel1[0]) * 180 / np.pi
        direction2 = np.arctan2(vel2[1], vel2[0]) * 180 / np.pi
        direction_similarity = self.calculate_direction_similarity(direction1, direction2)
        
        return speed_ratio * 0.5 + direction_similarity * 0.5

    def find_best_match(self, position, velocity, current_time):
        """향상된 매칭 로직"""
        best_match_id = None
        best_score = float('-inf')
        
        for track_id, data in self.person_database.items():
            # 최근에 본 트랙만 고려
            if self.frame_count - data['last_seen'] > self.time_window:
                continue
                
            if not data['positions'] or not data['velocities']:
                continue
            
            # 1. 위치 유사도
            last_position = data['positions'][-1]
            position_diff = np.linalg.norm(position - last_position)
            position_score = max(0, 1 - position_diff / self.position_threshold)
            
            # 2. 속도 유사도
            last_velocity = data['velocities'][-1]
            velocity_similarity = self.calculate_velocity_similarity(velocity, last_velocity)
            
            # 3. 방향 유사도
            if np.linalg.norm(velocity) > self.min_speed_threshold and np.linalg.norm(last_velocity) > self.min_speed_threshold:
                current_direction = np.arctan2(velocity[1], velocity[0]) * 180 / np.pi
                last_direction = np.arctan2(last_velocity[1], last_velocity[0]) * 180 / np.pi
                direction_similarity = self.calculate_direction_similarity(current_direction, last_direction)
            else:
                direction_similarity = 0.0
            
            # 종합 점수 계산
            total_score = (
                position_score * self.position_weight +
                velocity_similarity * self.velocity_weight +
                direction_similarity * self.direction_weight
            )
            
            # 디버깅 정보 출력
            if total_score > 0.5:  # 의미있는 매칭만 출력
                print(f"Match candidate - ID:{track_id} "
                      f"Pos:{position_score:.2f} "
                      f"Vel:{velocity_similarity:.2f} "
                      f"Dir:{direction_similarity:.2f} "
                      f"Total:{total_score:.2f}")
            
            if total_score > best_score and total_score > 0.6:  # 최소 임계값 추가
                best_score = total_score
                best_match_id = track_id
        
        return best_match_id, best_score

    def process_frame(self, frame):
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

            self.frame_count += 1
            current_time = time.time()
            tracks = self.process_frame(frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue

                bbox = track.to_tlbr()
                x1, y1, x2, y2 = map(int, bbox)
                track_id = track.track_id
                
                # 중심점 계산
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                
                # 현재 위치 업데이트
                self.update_track_info(track_id, center, current_time)
                
                # 바운딩 박스 그리기
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 정보 텍스트 준비
                info_text = [
                    f"ID:{track_id} ({x2-x1}x{y2-y1})",  # ID와 크기
                ]
                
                if track_id in self.person_database:
                    data = self.person_database[track_id]
                    
                    # 실제 움직임 여부 확인
                    is_moving = self.is_actually_moving(data['positions'], data['velocities'])
                    
                    # 위치와 속도 정보 (항상 표시)
                    pos_diff = 0
                    speed = 0
                    direction = 0
                    if len(data['positions']) >= 2:
                        pos_diff = np.linalg.norm(data['positions'][-1] - data['positions'][-2])
                        if len(data['velocities']) > 0:
                            velocity = data['velocities'][-1]
                            speed = np.linalg.norm(velocity)
                            direction = np.arctan2(velocity[1], velocity[0]) * 180 / np.pi
                    
                    # 상세 정보 표시
                    status = "Moving" if is_moving else "Static"
                    info_text.extend([
                        f"Status: {status}",
                        f"Move: {pos_diff:.1f}px (min:{self.min_movement_threshold})",
                        f"Speed: {speed:.1f}px/s (min:{self.min_speed_threshold})",
                        f"Dir: {direction:.1f}°",
                        f"Center: ({int(center[0])}, {int(center[1])})"
                    ])
                    
                    # 시간 정보
                    tracking_time = current_time - data['timestamps'][0]
                    info_text.append(f"Time: {tracking_time:.1f}s")
                    
                    # 이동 경로와 속도 벡터는 움직일 때만 표시
                    if is_moving:
                        if len(data['positions']) >= 2:
                            positions = data['positions'][-30:]
                            overlay = frame.copy()
                            for i in range(1, len(positions)):
                                pt1 = tuple(map(int, positions[i-1]))
                                pt2 = tuple(map(int, positions[i]))
                                cv2.line(overlay, pt1, pt2, (255, 0, 0), 1)
                            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                        
                        if len(data['velocities']) > 0:
                            velocity = data['velocities'][-1]
                            if np.linalg.norm(velocity) > 0.1:
                                end_point = (
                                    int(center[0] + velocity[0]),
                                    int(center[1] + velocity[1])
                                )
                                cv2.arrowedLine(frame, 
                                              (int(center[0]), int(center[1])),
                                              end_point,
                                              (0, 0, 255), 1)

                # 텍스트 그리기 (가독성 향상)
                text_y_start = y1 - 5
                for i, text in enumerate(info_text):
                    text_y = text_y_start - i * 15
                    text_x = x1 + 2
                    
                    # 검은색 외곽선으로 텍스트 가독성 향상
                    cv2.putText(frame, text,
                              (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.4, (0, 0, 0), 2)
                    cv2.putText(frame, text,
                              (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.4, (255, 255, 255), 1)

            # 전체 상태 표시
            status_text = [
                f"Active Tracks: {len([t for t in tracks if t.is_confirmed()])}",
                f"Frame: {self.frame_count}",
                f"Move Threshold: {self.min_movement_threshold}px",
                f"Speed Threshold: {self.min_speed_threshold}px/s"
            ]
            
            for i, text in enumerate(status_text):
                cv2.putText(frame, text, (10, 25 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Person Tracking with Motion", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = PersonTracker()
    tracker.run()
