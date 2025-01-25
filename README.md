# Smoke Vision

**Smoke Vision** is a service designed to detect smoking behavior in non-smoking areas using YOLO and MediaPipe technologies. It utilizes TTS to deliver warning messages. The admin web application is built with FastAPI and WebSocket, designed to operate in a local environment.


## Highlights

1. **Object Detection with YOLO**
   - Detects objects such as faces, cigarettes, and vape pods.

2. **Cigarette Detection and Hand Proximity**
   - Measures the distance between detected cigarette objects and hands to determine smoking behavior.

3. **Motion Analysis with [MediaPipe](https://youtu.be/06TE_U21FK4?si=tHIS09G9UaemtRo5)**
   - Tracks and counts the bending movements of both arms.
   - Triggers detection when a single arm bends more than twice.

4. **Clothing Color Detection**
   - Identifies clothing objects using YOLO and extracts their color.
   - Mentions clothing color to pinpoint the smoking individual.

5. **Warning Notification**
   - Issues a warning message using TTS technology when all the following conditions are met:
     - Face and cigarette(including vape pod) is detected.
     - Cigarette detected near the hand.
     - Arm bending movements confirm smoking gestures.

## 실행 방법
1. 클론/다운로드 후 `requirements.txt` 안에 적힌 라이브러리를 모두 설치해주세요. 생략된 라이브러리가 있을 수 있으니 터미널에서 발생하는 메시지를 따라 추가 설치해주세요.
2. FastAPI 서버 실행
   ```
   uvicorn app:app --reload
   ```
3. 브라우저에서 접속
   ```
   <http://127.0.0.1:8000>
   ```

## 주의
웹캠과 실제 속도 간의 차이가 발생할 수 있음
