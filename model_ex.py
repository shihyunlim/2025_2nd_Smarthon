from ultralytics import YOLO
import cv2
import sys
import time

# 캠 기본 설정
live_cam = None

# 모델 초기화
def init():
    ciga_model = YOLO("./models/cigarette/ciga_best.pt")
    person_model = YOLO("./models/person/person_best.pt")
    return ciga_model, person_model

# 입력 소스 설정
def get_models(stream=True):
    ciga_model, person_model = init()
    if stream: # 웹캠 연결
        capture = cv2.VideoCapture(0)
        global live_cam
        live_cam = capture
        time.sleep(2)
        if not capture.isOpened():
            sys.exit("카메라 연결 오류")
    else:
        sys.exit("동영상 연결 기능이 주석 처리됨")

    return capture, ciga_model, person_model

# 웹캠 메모리 해제
def cam_close():
    if live_cam.isOpened():
        live_cam.release()

# 탐지 및 시각화
def run_detection():
    capture, ciga_model, person_model = get_models(stream=True)

    while True:
        ret, frame = capture.read()
        if not ret:
            print("캡처 오류 또는 스트림 종료")
            break

        # YOLO 모델로 사람 탐지 수행
        person_results = person_model.predict(frame, conf=0.5, iou=0.5)

        # 사람 박스를 기준으로 담배 탐지
        for person_result in person_results:
            for person_box in person_result.boxes:
                px1, py1, px2, py2 = map(int, person_box.xyxy[0])  # 사람 좌표
                conf_person = person_box.conf[0]  # 사람 탐지 신뢰도
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                cv2.putText(frame, f"Person {conf_person:.2f}", (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # 사람 영역 잘라내기
                person_frame = frame[py1:py2, px1:px2]

                # 사람 영역 내에서 담배 탐지 수행
                ciga_results = ciga_model.predict(person_frame, conf=0.5, iou=0.3)
                for ciga_result in ciga_results:
                    for ciga_box in ciga_result.boxes:
                        cx1, cy1, cx2, cy2 = map(int, ciga_box.xyxy[0])  # 담배 좌표
                        conf_ciga = ciga_box.conf[0]  # 담배 탐지 신뢰도

                        # 원본 프레임 좌표로 변환
                        abs_cx1, abs_cy1 = px1 + cx1, py1 + cy1
                        abs_cx2, abs_cy2 = px1 + cx2, py1 + cy2

                        # 담배 박스 및 레이블 표시
                        cv2.rectangle(frame, (abs_cx1, abs_cy1), (abs_cx2, abs_cy2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Cigarette {conf_ciga:.2f}", (abs_cx1, abs_cy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 결과 프레임 화면에 출력
        cv2.imshow("YOLO Detection", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cam_close()
    cv2.destroyAllWindows()

# 탐지 실행
run_detection()
