from ultralytics import YOLO
from gtts import gTTS
import cv2
import os
import time

# YOLO 모델 초기화
def init_models():
    face_cigar_model = YOLO("C:/Users/shihy/SmokeVision/models/face_cigar_1.pt")  # 얼굴과 담배 모델
    smoke_vapepod_model = YOLO("C:/Users/shihy/SmokeVision/models/smoke_vapepod_1.pt")  # 연기와 전자담배 모델
    return face_cigar_model, smoke_vapepod_model

# 경고 음성 출력 함수
def play_warning():
    warning_text = "이곳에서 담배를 피우시면 안됩니다."
    tts = gTTS(text=warning_text, lang='ko')
    warning_file = "warning.mp3"
    tts.save(warning_file)
    os.system(f"start {warning_file}")

# 메인 함수
def main():
    # 모델 초기화
    face_cigar_model, smoke_vapepod_model = init_models()

    # 웹캠 열기
    capture = cv2.VideoCapture(0)
    time.sleep(2)  # 웹캠 안정화 대기
    if not capture.isOpened():
        print("웹캠 연결 오류")
        return

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("영상 캡처 실패")
                break

            # 모델로 객체 탐지
            results_face_cigar = face_cigar_model(frame)
            results_smoke_vapepod = smoke_vapepod_model(frame)

            # 탐지 결과에서 필요한 객체 확인
            cigar_detected = False
            for result in results_face_cigar:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)]
                    if class_name == "cigarette":  # 담배 객체 확인
                        cigar_detected = True
                        break

            # 연기나 전자담배는 추가적으로 필요시 확인
            # for result in results_smoke_vapepod:
            #     for box in result.boxes:
            #         class_name = result.names[int(box.cls)]
            #         if class_name in ["smoke", "vapepod"]:
            #             cigar_detected = True
            #             break

            # 담배 객체 감지 시 경고 음성 출력
            if cigar_detected:
                play_warning()

            # 탐지 결과 화면 출력
            rendered_frame = results_face_cigar[0].plot()  # 탐지 결과 시각화
            cv2.imshow("Cigarette Detection", rendered_frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 웹캠 자원 해제
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
