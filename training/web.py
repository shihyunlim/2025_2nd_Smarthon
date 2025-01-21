from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import torch
import os
import datetime
import asyncio

# YOLO 캐시 경로 설정
os.environ['TORCH_HOME'] = 'C:/Users/kimsi/yolo_cache'

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# FastAPI 앱 생성
app = FastAPI()

# Static 폴더 생성 (이미지 저장용)
if not os.path.exists("static"):
    os.mkdir("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML 코드
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>흡연자 탐지 시스템</title>
        <style>
            body {
                background-color: #2c2c54; 
                color: white;
                font-family: Arial, sans-serif;
            }
            .container {
                display: flex;
                flex-direction: row;
                justify-content: space-between;
                padding: 20px;
            }
            .results {
                width: 30%;
                padding: 10px;
            }
            .results ul {
                list-style: none;
                padding: 0;
            }
            .results li {
                margin: 10px 0;
                padding: 10px;
                background-color: #40407a;
                border-radius: 5px;
                text-align: center;
            }
            .image-container {
                width: 65%;
                text-align: center;
            }
            img {
                max-width: 100%;
                border: 2px solid #575fcf;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <h1 style="text-align:center;">흡연자 탐지 시스템</h1>
        <div class="container">
            <div class="results">
                <h3>탐지 결과</h3>
                <ul id="results"></ul>
            </div>
            <div class="image-container">
                <h3>탐지된 이미지</h3>
                <img id="capture" src="" alt="탐지된 이미지">
            </div>
        </div>
        <script>
            const ws = new WebSocket("ws://127.0.0.1:8000/ws");
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);

                // 탐지된 객체가 "cell phone"일 경우만 업데이트
                if (data.objects.includes("cell phone")) {
                    const listItem = document.createElement("li");
                    listItem.textContent = `[${data.time}] Detected: cell phone`;
                    document.getElementById("results").innerHTML = ""; // 기존 결과 지우기
                    document.getElementById("results").appendChild(listItem);

                    // 이미지 업데이트
                    if (data.image_url) {
                        document.getElementById("capture").src = data.image_url;
                    }
                }
            };
        </script>
    </body>
</html>
"""

# HTML 반환
@app.get("/")
async def get():
    return HTMLResponse(content=html)

# WebSocket 경로
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)  # 웹캠

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 탐지 수행
        results = model(frame)
        detected_objects = results.pandas().xyxy[0]["name"].tolist()

        if "cell phone" in detected_objects:
            # 탐지 결과 박스 표시
            for _, row in results.pandas().xyxy[0].iterrows():
                if row["name"] == "cell phone":
                    cv2.rectangle(
                        frame,
                        (int(row["xmin"]), int(row["ymin"])),
                        (int(row["xmax"]), int(row["ymax"])),
                        (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        row["name"],
                        (int(row["xmin"]), int(row["ymin"]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            # 현재 시간과 날짜
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 이미지 저장
            capture_path = f"static/capture_{current_time.replace(':', '-').replace(' ', '_')}.jpg"
            cv2.imwrite(capture_path, frame)

            # WebSocket으로 결과 전송
            await websocket.send_json({
                "time": current_time,
                "objects": detected_objects,
                "image_url": f"/static/{os.path.basename(capture_path)}"
            })

        await asyncio.sleep(1)

    cap.release()
