from gtts import gTTS
import os

# 변환하려는 문장들을 리스트에 저장합니다.
sentences = ["담배가 감지되었습니다. 이곳에서 담배를 피시면 안됩니다!"]

# 파일을 저장할 디렉토리를 설정합니다.
directory = "C:/Users/shihy/SmokeVision/voices"

# 각 문장을 음성으로 변환하고 출력합니다.
for i, sentence in enumerate(sentences):
    tts = gTTS(text=sentence, lang='ko')
    filename = os.path.join(directory, f"voice{i}.mp3")
    tts.save(filename)

    # 저장된 음성 파일을 재생합니다. 아래 코드는 Windows에서 동작합니다.
    os.system(f"start {filename}")