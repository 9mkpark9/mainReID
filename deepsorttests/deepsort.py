import cv2
import torch
import os
import pickle
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

############################################
# GPU 기반 중앙 카메라 추적 시스템
# YOLOv8를 통한 사람 검출, 향상된 DeepSort로 추적, 안정적인 ID 부여
############################################

# GPU 사용 가능 여부 확인 및 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# YOLOv8 모델 로드 - GPU 설정
yolo_model = YOLO('yolo11x.pt')
yolo_model.to(DEVICE)

# DeepSort 추적기 초기화 - 안정성 향상을 위한 파라미터 조정
tracker = DeepSort(
    max_age=15,                # 트랙 수명 감소
    n_init=8,                 # 트랙 초기화에 필요한 연속 탐지 수 증가
    max_iou_distance=0.5,      # IOU 매칭 임계값 더 엄격하게
    max_cosine_distance=0.3,   # 특징 유사도 임계값 더 엄격하게
    nn_budget=200,             # 더 많은 특징 저장
    nms_max_overlap=1.0,
    embedder='mobilenet',
    embedder_gpu=True if DEVICE == 'cuda' else False,
    half=True if DEVICE == 'cuda' else False
)

def process_detections(results, min_confidence=0.5):  # 신뢰도 임계값 설정
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0 and conf > min_confidence:
                w = x2 - x1
                h = y2 - y1
                # 크기 필터링 강화
                if w * h > 200 and w/h < 2 and h/w < 3:  # 사람의 일반적인 비율 고려
                    bbox = [x1, y1, w, h]
                    detections.append((bbox, conf, 0))
    return detections

def filter_tracks(tracks):
    """트랙 필터링을 통한 오탐지 제거"""
    filtered_tracks = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        # 트랙 안정성 검사
        if track.time_since_update > 1:
            continue
            
        # 속도 기반 필터링
        if hasattr(track, 'velocity') and track.velocity is not None:
            velocity = (track.velocity[0]**2 + track.velocity[1]**2)**0.5
            if velocity > 100:  # 급격한 움직임 필터링
                continue
        
        filtered_tracks.append(track)
    return filtered_tracks

def load_track_history():
    id_file = 'track_ids.pkl'
    if os.path.exists(id_file):
        with open(id_file, 'rb') as f:
            return pickle.load(f)
    return {}

def save_track_history(track_ids):
    with open('track_ids.pkl', 'wb') as f:
        pickle.dump(track_ids, f)

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    track_ids = load_track_history()
    next_id = 1 if not track_ids else max(track_ids.values()) + 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("중앙 카메라를 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"카메라가 시작되었습니다. 디바이스: {DEVICE}")
    print("종료하려면 'esc '를 누르세요.")

    # 이전 프레임의 트랙 정보 저장
    prev_tracks = {}

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다. 카메라를 다시 시작합니다...")
                cap.release()
                cap = cv2.VideoCapture(0)
                continue

            frame_count += 1
            if frame_count % 3 != 0:
                continue

            # numpy 배열을 직접 사용
            with torch.no_grad():
                preds = yolo_model(frame, verbose=False)
            
            detections = process_detections(preds)
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # 트랙 필터링
            filtered_tracks = filter_tracks(tracks)

            # 트랙 시각화
            for track in filtered_tracks:
                track_id = track.track_id
                
                # ID 안정성 향상을 위한 이전 프레임 정보 활용
                if track_id not in track_ids:
                    # 이전 위치와 비교하여 가까운 트랙 찾기
                    assigned_id = None
                    if track_id in prev_tracks:
                        prev_pos = prev_tracks[track_id]
                        curr_pos = track.to_tlbr()
                        dist = ((prev_pos[0] - curr_pos[0])**2 + (prev_pos[1] - curr_pos[1])**2)**0.5
                        if dist < 100:  # 이동 거리가 적으면 이전 ID 유지
                            assigned_id = track_ids.get(track_id)
                    
                    if assigned_id is None:
                        track_ids[track_id] = next_id
                        next_id += 1
                
                final_id = track_ids[track_id]
                
                l, t, r, b = track.to_tlbr()
                x1, y1, x2, y2 = map(int, [l, t, r, b])
                
                # 현재 위치 저장
                prev_tracks[track_id] = (x1, y1, x2, y2)
                
                color = ((final_id * 123) % 255, (final_id * 85) % 255, (final_id * 147) % 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID: {final_id}"
                if hasattr(track, 'det_conf') and track.det_conf is not None:
                    label += f" Conf: {track.det_conf:.2f}"
                
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            active_tracks = len(filtered_tracks)
            cv2.putText(frame, f"People: {active_tracks}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                cv2.putText(frame, f"GPU Memory: {gpu_memory:.1f}MB", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            status = f"Monitoring on {DEVICE}... Press 'esc' to quit"
            cv2.putText(frame, status, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Central Camera (Multi-person Tracking)", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                print("프로그램을 종료합니다...")
                break

    finally:
        save_track_history(track_ids)
        cap.release()
        cv2.destroyAllWindows()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()