import cv2
import torch
import numpy as np
import requests
import time
import serial
# 파이썬 3.13.3버전 이용할 것
# ✅ 시리얼 연결
try:
    ser = serial.Serial('COM4', 9600, timeout=0.5)  # COM 포트 확인 필수
    time.sleep(1)
    print("✅ 아두이노 시리얼 연결 성공")
except Exception as e:
    print("❌ 시리얼 연결 실패:", e)
    ser = None

# ✅ YOLO 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.3
model.to('cpu')

# ✅ ESP32-CAM 스트림 주소
stream_url = 'http://192.168.236.130:81/stream'

# ✅ 면적에 따른 신호 판단 함수
def get_signal(area):
    if 3000 <= area < 10000:
        return 1
    elif 10000 <= area < 20000:
        return 2
    elif 20000 <= area < 40000:
        return 3
    elif 40000 <= area < 100000:
        return 4
    return None

# ✅ 상태 변수들
current_ref_area = 0
current_signal = None

# ✅ 감지 실패 시간 체크
no_person_start_time = None
no_person_delay_sec = 2  # 2초 연속 실패 시만 0 전송

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
                    if int(cls) == 0:  # 사람 class
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        area = (x2 - x1) * (y2 - y1)
                        areas.append(area)
                        boxes.append((x1, y1, x2, y2))

                if areas:
                    # ✅ 사람 감지됨 → 타이머 리셋
                    no_person_start_time = None

                    max_idx = np.argmax(areas)
                    max_area = areas[max_idx]
                    max_box = boxes[max_idx]

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

                    # ✅ 바운딩 박스 표시
                    x1, y1, x2, y2 = boxes[max_idx]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Area: {current_ref_area}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # ✅ 사람 없음 → 2초간 대기 후 신호 전송
                    if no_person_start_time is None:
                        no_person_start_time = time.time()
                    elif time.time() - no_person_start_time >= no_person_delay_sec:
                        if current_signal != 0:
                            current_signal = 0
                            print("📤 신호 전송: 0 (2초 연속 감지 실패)")
                            if ser and ser.is_open:
                                try:
                                    ser.write(b"0\n")
                                except Exception as e:
                                    print("❌ 시리얼 전송 실패:", e)

                # ✅ 영상 출력
                cv2.imshow("YOLO Person Detection", frame)
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

