import cv2
import torch
import numpy as np
import requests
import time

# YOLOv5 모델 불러오기 (cpu)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.3
model.to('cpu')  # CPU 강제 지정 (GPU 없으면 자동 CPU)

stream_url = 'http://192.168.236.130:81/stream'

while True:
    try:
        r = requests.get(stream_url, stream=True, timeout=5)
        if r.status_code != 200:
            print(f"스트림 연결 실패 (상태 코드: {r.status_code})")
            time.sleep(1)
            continue

        bytes_data = b''
        for chunk in r.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')  # JPEG 시작
            b = bytes_data.find(b'\xff\xd9')  # JPEG 끝

            if a != -1 and b != -1 and b > a:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # BGR -> RGB 변환
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # YOLOv5 객체 감지
                results = model(img_rgb)
                dets = results.xyxy[0].cpu().numpy()

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

                    # 바운딩 박스 그리기 (최대 면적에 대해서만)
                    x1, y1, x2, y2 = max_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Area: {max_area}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Person Detection (ESP32-CAM)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except requests.exceptions.RequestException as e:
        print(f"스트림 연결 오류: {e}")
        time.sleep(1)
        continue

cv2.destroyAllWindows()
