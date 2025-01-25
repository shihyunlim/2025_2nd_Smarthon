from ultralytics import YOLO
import cv2
import time
import mediapipe as mp
import numpy as np
from color_utils import extract_color
from audio_utils import WarningPlayer

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import os
import datetime
import asyncio

# Mediapipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands  # 손 감지 추가

def init_models():
        face_cigar_model = YOLO("models/face_cigar_1.pt")
        smoke_vapepod_model = YOLO("models/smoke_vapepod_1.pt")
        clothing_model = YOLO("models/clothing.pt")
        return face_cigar_model, smoke_vapepod_model, clothing_model
    
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# 담배와 손 거리 확인
def check_cigarette_in_hand(cigar_box, hand_landmarks, frame_shape): 
    if hand_landmarks is None:
        return False
    
    # 담배 박스의 중심점 계산
    cigar_center_x = (cigar_box[0] + cigar_box[2]) / 2
    cigar_center_y = (cigar_box[1] + cigar_box[3]) / 2
    
    # 손가락 끝점들의 위치 확인
    height, width = frame_shape[:2]
    for hand_lm in hand_landmarks.landmark:
        hand_x = hand_lm.x * width
        hand_y = hand_lm.y * height
        
        # 담배와 손 사이의 거리 계산
        distance = np.sqrt((cigar_center_x - hand_x)**2 + (cigar_center_y - hand_y)**2)
        
        # 일정 거리 이내에 있으면 손에 쥐고 있다고 판단
        if distance < 50:  # 픽셀 단위, 필요에 따라 조정
            return True
            
    return False

# FastAPI 앱 생성
app = FastAPI()

# Static 폴더 생성 (이미지 저장용)
if not os.path.exists("static"):
    os.mkdir("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates 폴더 생성
if not os.path.exists("templates"):
    os.mkdir("templates")

# HTML 파일 경로
html_file_path = "templates/main.html"

# HTML 반환
@app.get("/")
async def get():
    with open(html_file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

# WebSocket 경로
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    face_cigar_model, smoke_vapepod_model, clothing_model = init_models()
    warning_player = WarningPlayer()

    # Mediapipe Pose와 Hands 초기화
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    left_counter = 0
    right_counter = 0
    left_stage = None
    right_stage = None

    arm_connections = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
    ]

    capture = cv2.VideoCapture(0)
    time.sleep(2)
    if not capture.isOpened():
        print("웹캠 연결 오류")
        return
    
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("영상 캡처 실패")
                break

            # YOLO 객체 탐지
            results_face_cigar = face_cigar_model(frame)
            results_smoke_vapepod = smoke_vapepod_model(frame)
            results_clothing = clothing_model(frame)
            
            cigar_detected = False
            person_detected = False
            detected_class_names = []
            cigar_boxes = []

            for result in results_face_cigar:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if class_name == "cigarette":
                        cigar_boxes.append([x1, y1, x2, y2])
                        detected_class_names.append(class_name)
                    elif class_name == "face":
                        person_detected = True
                        detected_class_names.append(class_name)
                        
            # 전자담배 탐지
            for result in results_smoke_vapepod:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)]
                    if class_name == "vapepod":
                        cigar_detected = True
                        detected_class_names.append(class_name)
                    elif class_name == "smoke":
                        detected_class_names.append(class_name)

            # Mediapipe Hands 처리
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_results = hands.process(image_rgb)

            # 담배가 손에 있는지 확인
            if cigar_boxes and hands_results.multi_hand_landmarks:
                for cigar_box in cigar_boxes:
                    cigar_detected = False  # 초기화
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        if check_cigarette_in_hand(cigar_box, hand_landmarks, frame.shape):
                            cigar_detected = True
                            break
                    if not cigar_detected:
                        print("손과 담배 거리가 멉니다.")  # 손과 담배의 거리가 멀 경우 출력
                    if cigar_detected:
                        break

            # Mediapipe Pose 처리
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                try:
                    # 왼쪽 팔 랜드마크 추출
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    if left_angle > 130:
                        left_stage = "down"
                    if left_angle < 50 and left_stage == "down":
                        left_stage = "up"
                        left_counter += 1

                    if right_angle > 130:
                        right_stage = "down"
                    if right_angle < 50 and right_stage == "down":
                        right_stage = "up"
                        right_counter += 1

                    cv2.putText(image, f"Left Angle: {int(left_angle)}",
                              tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Right Angle: {int(right_angle)}",
                              tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Left Reps: {left_counter}",
                              (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Right Reps: {right_counter}",
                              (10, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                except:
                    pass

                # 팔 랜드마크 시각화
                for connection in arm_connections:
                    start_idx = connection[0].value
                    end_idx = connection[1].value
                    start_landmark = landmarks[start_idx]
                    end_landmark = landmarks[end_idx]
                    start_point = tuple(np.multiply([start_landmark.x, start_landmark.y], [640, 480]).astype(int))
                    end_point = tuple(np.multiply([end_landmark.x, end_landmark.y], [640, 480]).astype(int))
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)
                    cv2.circle(image, start_point, 4, (0, 0, 255), -1)
                    cv2.circle(image, end_point, 4, (0, 0, 255), -1)


            # 담배를 손에 들고 있을 때만 경고
            # 담배와 사람이 모두 탐지되고 팔 동작 조건 만족 시 경고
            if cigar_detected and person_detected and (left_counter >= 2 or right_counter >= 2):
                print("담배를 피우고 있음!")
                # 옷 탐지 및 색상 추출
                clothing_detected = False
                clothing_color = None
                clothing_class_name = None
                for result in results_clothing:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = result.names[class_id]

                        if class_name in [
                            "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear", 
                            "long_sleeved_outwear", "vest", "sling", "shorts", "trousers", 
                            "skirt", "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"
                        ]:
                            clothing_color = extract_color(frame, box.xyxy[0])  # 옷 색상 추출
                            clothing_class_name = class_name
                            clothing_detected = True
                            break
                    if clothing_detected:
                        break
                
                # 경고 음성 출력
                if clothing_detected and clothing_color:
                    warning_player.play_warning(detected_class_names, clothing_color, clothing_class_name)
                else:
                    warning_player.play_warning(detected_class_names)
                left_counter = 0
                right_counter = 0

            # 시각화
            rendered_frame = results_face_cigar[0].plot()
            combined_frame = cv2.addWeighted(rendered_frame, 0.6, image, 0.4, 0)
            
            # 손 랜드마크 시각화
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(combined_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        
            # 현재 시간과 날짜
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 이미지 저장
            capture_path = f"static/capture_{current_time.replace(':', '-').replace(' ', '_')}.jpg"
            cv2.imwrite(capture_path, frame)

            # WebSocket으로 결과 전송
            await websocket.send_json({
                "time": current_time,
                "objects": detected_class_names,
                "image_url": f"/static/{os.path.basename(capture_path)}"
            })
            
            # 화면 출력
            cv2.imshow("Frame", combined_frame)

            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(1)
         
    except Exception as e:
        print(f"에러 발생: {e}")
        
    finally:
        capture.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)