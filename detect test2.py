import cv2
import torch
import numpy as np
import time

# YOLOv5 모델 불러오기 (CPU)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.3
model.to('cpu')  # CPU 강제 지정 (GPU 없으면 안 하면 자동 CPU)

# ESP32-CAM MJPEG 스트림 URL
stream_url = 'http://192.168.162.130:81/stream'

def get_signal(area):
    if 10000 <= area < 30000:
        return 1
    elif 30000 <= area < 50000:
        return 2
    elif 50000 <= area < 70000:
        return 3
    elif 70000 <= area < 100000:
        return 4
    return None

current_ref_area = 0
current_signal = None

# VideoCapture로 스트림 열기
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("스트림을 열 수 없습니다.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        # 프레임을 읽지 못했을 때 재시도
        print("프레임을 읽는 중 오류 발생, 잠시 후 재연결 시도...")
        time.sleep(1)
        cap.release()
        cap = cv2.VideoCapture(stream_url)
        continue

    # BGR -> RGB 변환
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLOv5 객체 감지
    results = model(img_rgb)
    dets = results.xyxy[0].cpu().numpy()  # tensor -> numpy

    areas, boxes = [], []
    for det in dets:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # 사람(class=0)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
            boxes.append((x1, y1, x2, y2))

    if areas:
        max_idx = np.argmax(areas)
        max_area = areas[max_idx]
        max_box = boxes[max_idx]

        # 이전 기준 면적보다 크면 갱신, 아니면 영역 리스트에 없으면 갱신
        if max_area > current_ref_area:
            current_ref_area = max_area
        else:
            if current_ref_area not in areas:
                current_ref_area = max_area

        signal = get_signal(current_ref_area)
        if signal != current_signal:
            current_signal = signal
            print(f"신호 변경: {current_signal} (면적: {current_ref_area})")

        # 바운딩 박스 그리기 (가장 큰 영역에만)
        x1, y1, x2, y2 = max_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Area: {current_ref_area}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        current_ref_area = 0
        current_signal = None

    cv2.imshow("Person Detection (ESP32-CAM)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
