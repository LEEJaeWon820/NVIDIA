##### https://www.jjn.co.kr/news/photo/201706/717452_99561_4332.jpg

```bash
# 1. 필요한 라이브러리 설치
!pip install ultralytics

# 2. 필요한 모듈 임포트
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from google.colab import files

# 3. 이미지 업로드
uploaded = files.upload()  # '1.jpg' 파일을 업로드하세요

# 4. YOLOv8 모델 로드 (사전 학습된 모델 사용)
model = YOLO('yolov8n.pt')  # 'yolov8n.pt'는 가장 가벼운 모델, 필요 시 'yolov8s.pt', 'yolov8m.pt' 등 사용 가능

# 5. 이미지 읽기
image = cv2.imread('1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환

# 6. 객체 탐지 수행
results = model(image_rgb)

# 7. 결과 시각화
# 탐지된 객체를 이미지에 그리기
annotated_image = results[0].plot()  # YOLOv8의 plot() 메서드로 결과 이미지 생성

# 8. 결과 출력
plt.figure(figsize=(10, 10))
plt.imshow(annotated_image)
plt.axis('off')
plt.show()

# 9. 탐지된 사람 수 계산 및 출력
person_count = 0
for result in results[0].boxes:
    if result.cls == 0:  # YOLO에서 '0'은 'person' 클래스
        person_count += 1
print(f"감지된 사람 수: {person_count}")
```

### 1.jpg로 인식 -> 문제점: 컨피던스 문제, 파일이름이 1.jpg일 때만 인식 가능

---

```bash
# 1. 필요한 라이브러리 설치
!pip install ultralytics

# 2. 필요한 모듈 임포트
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from google.colab import files

# 3. 이미지 업로드
uploaded = files.upload()  

# 4. 업로드된 파일 이름 동적으로 가져오기
file_name = list(uploaded.keys())[0]  # 업로드된 첫 번째 파일 이름 사용

# 5. YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # 자동으로 다운로드됨

# 6. 이미지 읽기
image = cv2.imread(file_name)  # 동적으로 파일 이름 사용
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환

# 7. 객체 탐지 수행 (신뢰도 임계값 0.5 설정)
results = model(image_rgb, conf=0.5)  # 신뢰도 0.5 미만인 객체는 제외

# 8. 결과 시각화
annotated_image = results[0].plot()  # 탐지된 객체 표시

# 9. 결과 출력
plt.figure(figsize=(10, 10))
plt.imshow(annotated_image)
plt.axis('off')
plt.show()

# 10. 탐지된 사람 수 계산 및 출력
person_count = sum(1 for result in results[0].boxes if int(result.cls) == 0)  # 'person' 클래스는 0
print(f"감지된 사람 수: {person_count}")
```
![image](https://github.com/user-attachments/assets/c4bd776a-d64c-4204-8950-42784f184645)
