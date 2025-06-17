import cv2
import torch
import numpy as np
import requests
import time
import serial    # HC-10 통신을 위해 추가

# YOLOv5 모델 불러오기 (cpu)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5
model.to('cpu')  # CPU 강제 지정 (GPU 없으면 안 하면 자동 CPU)

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

# --- HC-10 시리얼 설정 (추가) ---
# Windows의 경우 'COM3', Linux/Mac의 경우 '/dev/ttyUSB0' 등으로 수정
try:
    ser = serial.Serial(port='COM3', baudrate=9600, timeout=0.1)
    time.sleep(1)  # 시리얼 포트 안정화 대기
    print("Serial port opened:", ser.portstr)
except Exception as e:
    print("시리얼 포트 열기 실패:", e)
    ser = None

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

                    # 이전보다 더 큰 사람이 검출되면 업데이트, 아니면 리스트에서 여전히 있는지 확인
                    if max_area > current_ref_area:
                        current_ref_area = max_area
                    else:
                        if current_ref_area not in areas:
                            current_ref_area = max_area

                    signal = get_signal(current_ref_area)

                    # signal 값이 바뀌었을 때만 시리얼로 전송
                    if signal != current_signal:
                        current_signal = signal
                        print(f"신호 변경: {current_signal} (면적: {current_ref_area})")
                        if ser and ser.is_open:
                            try:
                                msg = f"{current_signal}\n"  # "1\n", "2\n" 등
                                ser.write(msg.encode('utf-8'))
                                print(f"Sent signal to Arduino: {current_signal}")
                            except Exception as e:
                                print("시리얼 전송 에러:", e)
                        else:
                            print("시리얼 포트가 열려 있지 않습니다.")

                    # 바운딩 박스 그리기
                    for i, area in enumerate(areas):
                        if area == current_ref_area:
                            x1, y1, x2, y2 = boxes[i]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'Area: {area}', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            break
                else:
                    # 사람이 화면에 없으면 ref_area 및 signal 초기화
                    if current_signal is not None:
                        current_ref_area = 0
                        current_signal = None
                        print("사람 미검출: signal 초기화")
                        if ser and ser.is_open:
                            try:
                                ser.write("0\n".encode('utf-8'))  # 0을 보내 부저 끄기
                                print("Sent signal to Arduino: 0 (no person)")
                            except Exception as e:
                                print("시리얼 전송 에러:", e)

                cv2.imshow("Person Detection (ESP32-CAM)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except requests.exceptions.RequestException as e:
        print(f"스트림 연결 오류: {e}")
        time.sleep(1)
        continue

# 루프 종료 후 자원 정리
cv2.destroyAllWindows()
if ser and ser.is_open:
    ser.close()
    print("Serial port closed.")

