import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # YOLO11s-Pose 모델 로드
    model = YOLO("yolo11l-pose.pt")
    model.conf = 0.5
    model.verbose = False

    # 카메라 연결
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # ROI 설정을 위한 변수들
    roi_points = []
    is_setting_roi = True
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and is_setting_roi and len(roi_points) < 5:
            roi_points.append((x, y))
            
    # ROI 설정 윈도우 생성
    cv2.namedWindow("Set ROI")
    cv2.setMouseCallback("Set ROI", mouse_callback)
    
    # ROI 설정
    print("ROI 영역의 5개 점을 순서대로 클릭하세요.")
    while is_setting_roi:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # 현재까지 찍은 점들 표시
        for i, point in enumerate(roi_points):
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(frame, roi_points[i-1], point, (0, 255, 255), 2)
        
        # 마지막 점과 첫 점을 연결 (영역이 완성된 경우)
        if len(roi_points) == 5:
            cv2.line(frame, roi_points[-1], roi_points[0], (0, 255, 255), 2)
            is_setting_roi = False
            
        cv2.imshow("Set ROI", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
            cap.release()
            cv2.destroyAllWindows()
            return
            
    cv2.destroyWindow("Set ROI")
    
    # 사람 추적을 위한 변수들
    prev_people = {}
    next_id = 0

    # 메인 루프
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                continue

            # ROI 영역 표시
            pts = np.array(roi_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

            results = model(frame, verbose=False)
            current_people = []
            standing_count = 0  # standing 인원 카운트 변수
            total_count = 0     # 전체 인원 카운트 변수 추가

            if results is None or len(results) == 0:
                # 결과가 없어도 카운트는 표시
                cv2.putText(frame, f"Total detected: {total_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Standing: {standing_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("YOLO11s-Pose", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    try:
                        keypoints = result.keypoints.xy.cpu().numpy()
                        boxes = result.boxes.xyxy.cpu().numpy()
                        
                        for idx, person_keypoints in enumerate(keypoints):
                            try:
                                # 사람의 중심점 계산
                                shoulder_l = person_keypoints[5]
                                shoulder_r = person_keypoints[6]
                                center_x = int((shoulder_l[0] + shoulder_r[0]) / 2)
                                center_y = int((shoulder_l[1] + shoulder_r[1]) / 2)

                                # ROI 영역 내부에 있는 사람만 처리
                                if point_in_polygon((center_x, center_y), roi_points):
                                    total_count += 1  # ROI 내부의 전체 인원 카운트
                                    
                                    # 바운딩 박스 그리기
                                    box = boxes[idx]
                                    x1, y1, x2, y2 = map(int, box)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                    # 스켈레톤 그리기
                                    draw_skeleton(frame, [person_keypoints])
                                    
                                    # 자세 감지
                                    posture, info = detect_posture(person_keypoints)
                                    
                                    # standing인 경우 카운트 증가
                                    if posture == 'standing':
                                        standing_count += 1
                                        
                                        # 정보 표시 시작 위치 계산 (위에서부터 시작)
                                        y_offset = y1 - 125  # 바운딩 박스 상단에서 시작

                                        # 모든 텍스트의 간격을 25로 통일
                                        texts = [
                                            # 발목 미감지시 사용하는 각도
                                            (f"LKnee: {info['left_vertical_angle']:6.1f}" if info['left_vertical_angle'] is not None else "LKnee: None   ", y_offset),
                                            (f"RKnee: {info['right_vertical_angle']:6.1f}" if info['right_vertical_angle'] is not None else "RKnee: None   ", y_offset + 25),
                                            # 발목 감지시 사용하는 각도
                                            (f"LAnkle: {info['left_hip_knee_ankle_angle']:6.1f}" if info['left_hip_knee_ankle_angle'] is not None else "LAnkle: None   ", y_offset + 50),
                                            (f"RAnkle: {info['right_hip_knee_ankle_angle']:6.1f}" if info['right_hip_knee_ankle_angle'] is not None else "RAnkle: None   ", y_offset + 75),
                                            # ID와 상태 정보
                                            (f"ID: {next_id:2d}, {posture:<8}", y_offset + 100)
                                        ]

                                        # 텍스트 표시
                                        for text, y in texts:
                                            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                            cv2.rectangle(frame, 
                                                         (center_x - 50, y - text_height), 
                                                         (center_x - 50 + text_width, y + 5), 
                                                         (192, 192, 192),  # 밝은 회색 배경
                                                         -1)
                                            cv2.putText(frame, text, (center_x - 50, y), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # 검정색 텍스트

                                        # standing 상태 추적을 위해 정보 저장
                                        current_people.append({
                                            'center': (center_x, center_y),
                                            'keypoints': person_keypoints,
                                            'posture': posture,
                                            'info': info
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

                prev_people = matched_people

            # 프레임의 왼쪽 상단에 인원 수와 상태 표시
            cv2.putText(frame, f"Total detected: {total_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Standing: {standing_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 수업/쉬는시간 판단 (전체 인원의 50% 이상이 standing이면 쉬는시간)
            if total_count > 0:  # 0으로 나누기 방지
                standing_ratio = standing_count / total_count
                status = "Break Time" if standing_ratio >= 0.5 else "Class Time"
                status_color = (0, 255, 255) if standing_ratio >= 0.5 else (0, 0, 255)  # 쉬는시간은 노란색, 수업시간은 빨간색
                
                # 상태 텍스트 표시
                cv2.putText(frame, status, (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

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
        hip_l = keypoints[11]    # 왼쪽 골반
        hip_r = keypoints[12]    # 오른쪽 골반
        knee_l = keypoints[13]   # 왼쪽 무릎
        knee_r = keypoints[14]   # 오른쪽 무릎
        ankle_l = keypoints[15]  # 왼쪽 발목
        ankle_r = keypoints[16]  # 오른쪽 발목

        def calculate_angle_three_points(p1, p2, p3):
            """세 점 사이의 각도 계산"""
            try:
                v1 = np.array(p1) - np.array(p2)
                v2 = np.array(p3) - np.array(p2)
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                return angle
            except:
                return None

        def calculate_angle_vertical(p1, p2):
            """수직선과의 각도 계산"""
            try:
                vertical = np.array([0, -1])
                leg_vector = np.array(p2) - np.array(p1)
                cos_angle = np.dot(leg_vector, vertical) / (np.linalg.norm(leg_vector) * np.linalg.norm(vertical))
                angle = np.degrees(np.arccos(cos_angle))
                return angle
            except:
                return None

        # 발목 감지 여부 확인
        left_ankle_detected = all(ankle_l[:2])
        right_ankle_detected = all(ankle_r[:2])
        any_ankle_detected = left_ankle_detected or right_ankle_detected

        # 골반-무릎 수직각도 계산
        left_vertical_angle = calculate_angle_vertical(hip_l[:2], knee_l[:2]) if all(hip_l[:2]) and all(knee_l[:2]) else None
        right_vertical_angle = calculate_angle_vertical(hip_r[:2], knee_r[:2]) if all(hip_r[:2]) and all(knee_r[:2]) else None

        # 골반-무릎-발목 각도 계산
        left_hip_knee_ankle_angle = None
        right_hip_knee_ankle_angle = None
        if all(hip_l[:2]) and all(knee_l[:2]) and all(ankle_l[:2]):
            left_hip_knee_ankle_angle = calculate_angle_three_points(hip_l[:2], knee_l[:2], ankle_l[:2])
        if all(hip_r[:2]) and all(knee_r[:2]) and all(ankle_r[:2]):
            right_hip_knee_ankle_angle = calculate_angle_three_points(hip_r[:2], knee_r[:2], ankle_r[:2])

        # 발목이 감지되지 않았을 때의 조건
        if not any_ankle_detected:
            standing_condition = (
                (left_vertical_angle is not None and 170 <= left_vertical_angle <= 190) or
                (right_vertical_angle is not None and 170 <= right_vertical_angle <= 190)
            )
        # 발목이 하나라도 감지되었을 때의 조건
        else:
            # 골반-무릎-발목 각도 조건 (양쪽 모두 만족해야 함)
            standing_condition = (
                left_hip_knee_ankle_angle is not None and 170 <= left_hip_knee_ankle_angle <= 190 and
                right_hip_knee_ankle_angle is not None and 170 <= right_hip_knee_ankle_angle <= 190
            )

        # 최종 판단
        if standing_condition:
            return 'standing', {
                'left_hip_knee_ankle_angle': left_hip_knee_ankle_angle,
                'right_hip_knee_ankle_angle': right_hip_knee_ankle_angle,
                'left_vertical_angle': left_vertical_angle,
                'right_vertical_angle': right_vertical_angle
            }

        return None, {
            'left_hip_knee_ankle_angle': left_hip_knee_ankle_angle,
            'right_hip_knee_ankle_angle': right_hip_knee_ankle_angle,
            'left_vertical_angle': left_vertical_angle,
            'right_vertical_angle': right_vertical_angle
        }

    except Exception as e:
        print(f"자세 감지 중 오류 발생: {e}")
        return None, None


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

def point_in_polygon(point, polygon):
    """점이 다각형 내부에 있는지 확인하는 함수"""
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > y) != (polygon[j][1] > y) and
            x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
            (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            inside = not inside
        j = i

    return inside

if __name__ == "__main__":
    main()