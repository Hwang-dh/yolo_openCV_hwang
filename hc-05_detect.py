import cv2
import torch
import numpy as np
import requests
import time
import serial  # HC-05 시리얼 통신용

# YOLOv5 모델 불러오기 (cpu)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5
model.to('cpu')

stream_url = 'http://192.168.162.130:81/stream'

# HC-05 시리얼 통신 설정
try:
    ser = serial.Serial(port='COM5', baudrate=9600, timeout=0.1)  # 반드시 COM 포트 확인해서 수정
    time.sleep(1)
    print("✅ HC-05 시리얼 포트 열림")
except Exception as e:
    print("❌ 시리얼 포트 열기 실패:", e)
    ser = None

def get_signal(area):
    if 10000 <= area < 30000:
        return 1
    elif 30000 <= area < 50000:
        return 2
    elif 50000 <= area < 70000:
        return 3
    elif 70000 <= area < 100000:
        return 4
    return 0  # 사람이 없을 때는 0 전송

current_ref_area = 0
current_signal = None

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
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')

            if a != -1 and b != -1 and b > a:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(img_rgb)
                dets = results.xyxy[0].cpu().numpy()

                areas, boxes = [], []
                for det in dets:
                    x1, y1, x2, y2, conf, cls = det
                    if int(cls) == 0:
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
                        print(f"📤 신호 전송: {current_signal} (면적: {current_ref_area})")
                        if ser and ser.is_open:
                            try:
                                ser.write(f"{current_signal}\n".encode('utf-8'))
                            except Exception as e:
                                print("❌ 시리얼 전송 실패:", e)

                    # 박스 그리기
                    for i, area in enumerate(areas):
                        if area == current_ref_area:
                            x1, y1, x2, y2 = boxes[i]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'Area: {area}', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            break
                else:
                    if current_signal != 0:
                        current_signal = 0
                        if ser and ser.is_open:
                            try:
                                ser.write("0\n".encode('utf-8'))
                                print("📤 신호 전송: 0 (사람 없음)")
                            except Exception as e:
                                print("❌ 시리얼 전송 실패:", e)

                cv2.imshow("Person Detection (ESP32-CAM)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except requests.exceptions.RequestException as e:
        print(f"스트림 연결 오류: {e}")
        time.sleep(1)
        continue

cv2.destroyAllWindows()
if ser and ser.is_open:
    ser.close()
    print("🔌 시리얼 포트 닫힘")
