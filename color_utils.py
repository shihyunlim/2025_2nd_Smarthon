import numpy as np

# 색상 기준 정의
COLOR_RANGES = {
    "파란색": lambda b, g, r: b > 150 and g < 100 and r < 100,
    "빨간색": lambda b, g, r: b < 100 and g < 100 and r > 150,
    "초록색": lambda b, g, r: b < 100 and g > 150 and r < 100,
    "노란색": lambda b, g, r: b < 150 and g > 150 and r > 150,
    "회색": lambda b, g, r: 80 < b < 120 and 80 < g < 120 and 80 < r < 120,
    "검은색": lambda b, g, r: b < 50 and g < 50 and r < 50,
    "흰색": lambda b, g, r: b > 200 and g > 200 and r > 200,
    "핑크색": lambda b, g, r: b < 100 and g < 100 and 150 < r < 200,
    "보라색": lambda b, g, r: b > 100 and g < 100 and r > 100,
    "갈색": lambda b, g, r: 60 < b < 120 and 40 < g < 80 and 20 < r < 60,
    "주황색": lambda b, g, r: b < 100 and 80 < g < 150 and r > 150,
    "베이지색": lambda b, g, r: 150 < b < 200 and 200 < g < 250 and 170 < r < 220,
}

# 색 추출 함수
def extract_color(image, box):
    x1, y1, x2, y2 = map(int, box)  # 좌표로 변환
    cropped_image = image[y1:y2, x1:x2]
    average_color = np.mean(cropped_image, axis=(0, 1))  # 평균 BGR 값
    b, g, r = average_color  # 평균 색상 분리

    # 색상 분류
    for color_name, condition in COLOR_RANGES.items():
        if condition(b, g, r):
            return color_name

    return "알 수 없는 색상"  # 정의된 조건에 맞지 않으면 기본값 반환