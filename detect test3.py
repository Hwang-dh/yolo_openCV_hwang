import cv2
import torch
import numpy as np
import requests
import time

# YOLOv5 모델 불러오기 (CPU)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.3
model.to('cpu')  # CPU 강제 지정

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
frame_idx = 0  # 프레임 카운터

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

                frame_idx += 1

                # 감지는 3프레임마다 한 번씩 실행
                if frame_idx % 3 == 0:
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

                        if max_area > current_ref_area:
                            current_ref_area = max_area
                        else:
                            if current_ref_area not in areas:
                                current_ref_area = max_area

                        signal = get_signal(current_ref_area)
                        if signal != current_signal:
                            current_signal = signal
                            print(f"신호 변경: {current_signal} (면적: {current_ref_area})")

                        # 바운딩 박스 그리기 (가장 큰 영역만)
                        x1, y1, x2, y2 = max_box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f'Area: {current_ref_area}',
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )
                    else:
                        current_ref_area = 0
                        current_signal = None

                # 감지하지 않는 프레임에서는 이전 바운딩 박스 없이 그대로 표시
                cv2.imshow("Person Detection (ESP32-CAM)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except requests.exceptions.RequestException as e:
        print(f"스트림 연결 오류: {e}")
        time.sleep(1)
        continue

cv2.destroyAllWindows()
