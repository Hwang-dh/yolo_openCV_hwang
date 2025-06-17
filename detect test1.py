import cv2
import torch
import numpy as np
import time

# 1. YOLOv5 모델 불러오기 (CPU 전용)
#    - 리사이즈 최적화는 아래에서 직접 처리
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5
model.to('cpu')

# 스트림 URL (ESP32-CAM)
stream_url = 'http://192.168.162.130:81/stream'

# 면적별 신호를 반환하는 함수 (기존과 동일)
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

# 2. VideoCapture로 스트림 열기
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("스트림 열기 실패")
    exit(1)

frame_count = 0            # 총 프레임 카운터
skip_frames = 2            # 0,1 프레임 스킵 → 2프레임마다 탐지
resize_width = 640         # 리사이즈할 너비
resize_height = 480        # 리사이즈할 높이

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        # 스트림에서 프레임을 못 읽으면 잠시 대기
        time.sleep(0.1)
        continue

    frame_count += 1

    # 3. 윈도우 크기가 크면 매번 탐지하면 느려지므로, N프레임마다 한 번만 탐지
    if frame_count % (skip_frames + 1) == 0:
        # (1) 원본 프레임을 리사이즈
        #     -> YOLOv5의 연산량 감소
        frame_resized = cv2.resize(frame, (resize_width, resize_height))

        # (2) BGR → RGB 변환
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # (3) YOLOv5 객체 감지
        results = model(img_rgb)
        dets = results.xyxy[0].cpu().numpy()  # tensor → numpy

        # (4) 사람(class=0)만 걸러내서 면적, 박스 정보 확보
        areas, boxes = [], []
        for det in dets:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0:  # 사람
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                area = (x2 - x1) * (y2 - y1)
                areas.append(area)
                boxes.append((x1, y1, x2, y2))

        # (5) 영역 비교 로직 (기존 로직 유지)
        if areas:
            max_idx = np.argmax(areas)
            max_area = areas[max_idx]
            max_box = boxes[max_idx]

            # 이전 기준 면적보다 더 크면 갱신, 그렇지 않으면 영역 안에 없으면 갱신
            if max_area > current_ref_area:
                current_ref_area = max_area
            else:
                if current_ref_area not in areas:
                    current_ref_area = max_area

            signal = get_signal(current_ref_area)
            if signal != current_signal:
                current_signal = signal
                print(f"신호 변경: {current_signal} (면적: {current_ref_area})")
        else:
            # 사람 감지가 없으면 면적 초기화
            current_ref_area = 0
            current_signal = None

        # (6) 최종적으로 기준 면적 박스만 원본 크기로 다시 그리기
        #     → ROI 좌표를 원본 프레임 크기에 맞게 스케일링
        if current_ref_area > 0 and areas:
            # 기준이 되는 박스 좌표 (리사이즈 기준)
            sx1, sy1, sx2, sy2 = boxes[np.argmax(areas)]
            # 스케일링 비율 계산
            h_ratio = frame.shape[0] / resize_height
            w_ratio = frame.shape[1] / resize_width
            ox1 = int(sx1 * w_ratio)
            oy1 = int(sy1 * h_ratio)
            ox2 = int(sx2 * w_ratio)
            oy2 = int(sy2 * h_ratio)
            # 원본 프레임에 박스 그리기
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
            cv2.putText(frame, f'Area: {current_ref_area}', (ox1, oy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 4. 화면 출력 (원본 해상도로 보여줌)
    cv2.imshow("Person Detection (ESP32-CAM)", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
