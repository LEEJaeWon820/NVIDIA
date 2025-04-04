##### https://www.youtube.com/watch?v=7LrWGGJFEJo

```bash
# 1. 필요한 라이브러리 설치
!pip install ultralytics

# 2. 필요한 모듈 임포트
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from google.colab import files

# 3. 동영상 업로드
uploaded = files.upload()

# 4. 업로드된 동영상 파일 이름 가져오기
video_file = list(uploaded.keys())[0]

# 5. YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# 6. 동영상 읽기
cap = cv2.VideoCapture(video_file)

# 7. 프레임별로 처리 (모든 프레임 출력)
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 8. 객체 탐지 수행 (신뢰도 0.5 이상)
    results = model(frame_rgb, conf=0.5)

    # 9. 결과 시각화
    annotated_frame = results[0].plot()
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_frame)
    plt.axis('off')
    plt.title(f"Frame {frame_count}")
    plt.show()

    # 10. 자동차 수 계산 (클래스 ID 2 = car)
    car_count = sum(1 for result in results[0].boxes if int(result.cls) == 2)
    print(f"Frame {frame_count} - 감지된 자동차 수: {car_count}")

# 11. 동영상 객체 해제
cap.release()
print("동영상 처리 완료")
```
