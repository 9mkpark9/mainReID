import cv2
import numpy as np
import json
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("yolov8n.pt")  # YOLOv8 경량 모델 사용

# 1. JSON 파일에서 임베딩 불러오기
def load_embeddings_from_file(filename="embeddings.json"):
    with open(filename, "r") as f:
        embeddings_data = json.load(f)
    print(f"'{filename}'에서 임베딩 데이터를 불러왔습니다.")
    return embeddings_data

# 2. 임베딩 비교 함수
def find_closest_person_id(new_embedding, embeddings_data, threshold=0.5):
    """
    새로운 임베딩이 들어왔을 때, 저장된 임베딩과 비교하여 가장 가까운 ID를 반환.
    """
    best_id = None
    best_distance = float('inf')

    for person_key, person_data in embeddings_data.items():
        saved_embedding = np.array(person_data["embedding"])
        # 유클리디안 거리 계산
        distance = np.linalg.norm(new_embedding - saved_embedding)
        if distance < best_distance:
            best_distance = distance
            best_id = person_data["id"]

    # Threshold 이하일 경우 동일인으로 간주
    if best_distance < threshold:
        return best_id
    return None

# 3. YOLO로 객체 추적 및 비교
def run_yolo_and_compare(video_path, embeddings_file="embeddings.json", threshold=0.5):
    embeddings_data = load_embeddings_from_file(embeddings_file)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO로 객체 감지
        results = model(frame)

        for result in results:
            boxes = result.boxes.cpu().numpy()  # Bounding boxes
            for box in boxes:
                cls = int(box.cls[0])  # 클래스 ID
                conf = box.conf[0]    # Confidence score

                # "사람" 클래스만 필터링
                if cls == 0 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 박스 좌표
                    person_image = frame[y1:y2, x1:x2]  # 박스 안의 영역

                    # 가짜 임베딩 생성 (여기서 ReID 모델로 교체 가능)
                    new_embedding = np.random.rand(128)

                    # 기존 임베딩과 비교해 ID 할당
                    person_id = find_closest_person_id(new_embedding, embeddings_data, threshold)

                    if person_id is not None:
                        print(f"기존 사람 발견! ID: {person_id}")
                    else:
                        print("새로운 사람 발견!")

                    # Bounding box와 ID 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {person_id if person_id else 'New'}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 결과 화면 출력
        cv2.imshow("YOLOv8 + ReID Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    video_path = 0  # 0은 웹캠, 또는 "path_to_video.mp4" 입력
    run_yolo_and_compare(video_path)
