import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # YOLO11s-Pose 모델 로드
    model = YOLO("yolo11s-pose.pt")

    # 카메라 연결
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 사람 추적을 위한 이전 프레임 정보 저장
    prev_people = {}  # {id: {'center': (x, y), 'posture': str}}
    next_id = 0

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                continue  # 다음 프레임 시도

            results = model(frame)
            current_people = []  # 현재 프레임의 사람들 정보

            # 결과가 유효한지 확인
            if results is None or len(results) == 0:
                cv2.imshow("YOLO11s-Pose", frame)  # 빈 프레임이라도 표시
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    try:
                        keypoints = result.keypoints.xy.cpu().numpy()
                        
                        for person_keypoints in keypoints:
                            try:
                                # 스켈레톤 그리기 - 항상 수행
                                draw_skeleton(frame, [person_keypoints])
                                
                                # 사람의 중심점 계산 (어깨 중심)
                                shoulder_l = person_keypoints[5]
                                shoulder_r = person_keypoints[6]
                                center_x = int((shoulder_l[0] + shoulder_r[0]) / 2)
                                center_y = int((shoulder_l[1] + shoulder_r[1]) / 2)
                                
                                # 자세 감지
                                posture = detect_posture(person_keypoints)
                                
                                # standing 자세일 때만 정보 추가 및 표시
                                if posture == 'standing':
                                    current_people.append({
                                        'center': (center_x, center_y),
                                        'keypoints': person_keypoints,
                                        'posture': posture
                                    })
                            except Exception as e:
                                continue  # 한 사람의 처리가 실패해도 다음 사람 처리
                    except Exception as e:
                        continue  # 키포인트 변환 실패시 다음 결과 처리

            # ID 할당 및 추적 부분은 현재 사람이 감지된 경우에만 실행
            if current_people:
                matched_people = {}
                
                for person in current_people:
                    min_dist = float('inf')
                    matched_id = None
                    
                    for prev_id, prev_info in prev_people.items():
                        dist = np.sqrt((person['center'][0] - prev_info['center'][0])**2 +
                                     (person['center'][1] - prev_info['center'][1])**2)
                        if dist < min_dist and dist < 100:
                            min_dist = dist
                            matched_id = prev_id
                    
                    if matched_id is None:
                        matched_id = next_id
                        next_id += 1
                    
                    matched_people[matched_id] = {
                        'center': person['center'],
                        'posture': person['posture']
                    }
                    
                    x, y = person['center']
                    cv2.putText(frame, f"ID: {matched_id}, {person['posture']}", 
                               (x - 50, y - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                prev_people = matched_people

                # 전체 상태 표시
                y_offset = 30
                for person_id, info in matched_people.items():
                    status_text = f"Person {person_id}: {info['posture']}"
                    cv2.putText(frame, status_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_offset += 30

            cv2.imshow("YOLO11s-Pose", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {e}")
            continue  # 오류가 발생해도 다음 프레임 처리

    cap.release()
    cv2.destroyAllWindows()

def detect_posture(keypoints, confidence_threshold=0.5):
    """자세 판단 함수"""
    try:
        # 필요한 키포인트 추출
        hip_l = keypoints[11]  # 왼쪽 골반
        hip_r = keypoints[12]  # 오른쪽 골반
        knee_l = keypoints[13]  # 왼쪽 무릎
        knee_r = keypoints[14]  # 오른쪽 무릎

        # 각도 계산 함수
        def calculate_angle(p1, p2):
            try:
                # 수직 벡터 (위쪽 방향)
                vertical = np.array([0, -1])
                # 골반-무릎 벡터
                leg_vector = np.array(p2) - np.array(p1)
                # 각도 계산
                cos_angle = np.dot(leg_vector, vertical) / (np.linalg.norm(leg_vector) * np.linalg.norm(vertical))
                angle = np.degrees(np.arccos(cos_angle))
                return angle
            except:
                return None

        # 왼쪽 다리 각도 계산 (골반-무릎)
        left_valid = all(hip_l[:2]) and all(knee_l[:2])  # 왼쪽 키포인트가 유효한지 확인
        left_angle = calculate_angle(hip_l[:2], knee_l[:2]) if left_valid else None
        
        # 오른쪽 다리 각도 계산 (골반-무릎)
        right_valid = all(hip_r[:2]) and all(knee_r[:2])  # 오른쪽 키포인트가 유효한지 확인
        right_angle = calculate_angle(hip_r[:2], knee_r[:2]) if right_valid else None

        # 디버깅을 위한 각도 출력
        if left_angle is not None:
            print(f"왼쪽 다리 각도: {left_angle:.1f}")
        if right_angle is not None:
            print(f"오른쪽 다리 각도: {right_angle:.1f}")

        # 서있는 상태 판단 (한쪽이라도 조건을 만족하면 standing)
        if (left_angle is not None and 165 <= left_angle <= 195) or \
           (right_angle is not None and 165 <= right_angle <= 195):
            return 'standing'

        return None

    except Exception as e:
        print(f"자세 감지 중 오류 발생: {e}")
        return None


def draw_skeleton(frame, keypoints):
    """
    관절 키포인트를 연결하여 스켈레톤을 그립니다.
    """
    h, w, _ = frame.shape

    # YOLO pose 키포인트 연결 관계 정의
    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12],  # 팔
        [11, 12],  # 어깨
        [5, 11], [6, 12],   # 몸통
        [5, 6],    # 골반
        [5, 7], [7, 9], [6, 8], [8, 10],  # 다리
        [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # 얼굴
        [3, 5], [4, 6]  # 귀-어깨 연결
    ]

    # 각 사람에 대해 처리
    for person in keypoints:
        # 키포인트 그리기
        for point in person:
            if len(point) >= 2:
                x, y = point[:2]
                if x > 0 and y > 0:
                    # 상대 좌표를 절대 좌표로 변환
                    if 0 < x <= 1 and 0 < y <= 1:
                        x, y = int(x * w), int(y * h)
                    else:
                        x, y = int(x), int(y)
                    
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # 스켈레톤 선 그리기
        for pair in skeleton:
            if len(person[pair[0]]) >= 2 and len(person[pair[1]]) >= 2:
                pt1 = person[pair[0]][:2]
                pt2 = person[pair[1]][:2]

                # 두 점이 모두 유효한 경우에만 선 그리기
                if all(pt1) and all(pt2):
                    # 상대 좌표를 절대 좌표로 변환
                    if 0 < pt1[0] <= 1 and 0 < pt1[1] <= 1:
                        pt1 = (int(pt1[0] * w), int(pt1[1] * h))
                    else:
                        pt1 = (int(pt1[0]), int(pt1[1]))

                    if 0 < pt2[0] <= 1 and 0 < pt2[1] <= 1:
                        pt2 = (int(pt2[0] * w), int(pt2[1] * h))
                    else:
                        pt2 = (int(pt2[0]), int(pt2[1]))

                    # 선 그리기
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

if __name__ == "__main__":
    main()