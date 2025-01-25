from gtts import gTTS
import time
import subprocess

class WarningPlayer:
    def __init__(self):
        self.last_warning_time = None  # 마지막 경고 시간을 기록

    def play_warning(self, class_names, color=None, clothing_class_name=None):
        current_time = time.time()

        # 마지막 경고로부터 30초가 지나지 않았다면 새로운 경고를 무시
        if self.last_warning_time and current_time - self.last_warning_time < 30:
            return

        self.last_warning_time = current_time  # 경고 시간 갱신

        # 경고 텍스트 생성
        if color and clothing_class_name:
            warning_text = (
                f"{', '.join(class_names)} 객체가 탐지되었습니다. "
                f"{color} {clothing_class_name}을(를) 입은 분! 이곳에서 담배를 피시면 안됩니다."
            )
        else:
            warning_text = (
                f"{', '.join(class_names)} 객체가 탐지되었습니다. "
                "이곳에서 담배를 피시면 안됩니다."
            )

        # 음성 파일 생성
        try:
            tts = gTTS(text=warning_text, lang='ko')
            warning_file = "warning.mp3"
            tts.save(warning_file)

            # 음성 파일 재생
            subprocess.run(["start", warning_file], shell=True)
        except Exception as e:
            print(f"경고 음성 생성/재생 중 오류 발생: {e}")
