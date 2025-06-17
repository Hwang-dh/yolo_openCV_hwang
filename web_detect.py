import cv2
import torch
import numpy as np
import time

# ✅ YOLO 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.3
model.to('cpu')

# ✅ 면적에 따른 신호 판단 함수 (전송은 안 하지만 출력용으로 유지)
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
no_person_start_time = None
no_person_delay_sec = 2  # 2초 연속 실패 시만 신호 0 출력

# ✅ 웹캠 사용
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

print("✅ 웹캠 감지 시작")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

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
        no_person_start_time = None
        max_idx = np.argmax(areas)
        max_area = areas[max_idx]
        max_box = boxes[max_idx]

        current_ref_area = max_area
        signal = get_signal(current_ref_area)

        if signal != current_signal:
            current_signal = signal
            print(f"🟢 감지 신호: {current_signal} (면적: {current_ref_area})")

        # ✅ 바운딩 박스 표시
        x1, y1, x2, y2 = boxes[max_idx]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Area: {current_ref_area}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        if no_person_start_time is None:
            no_person_start_time = time.time()
        elif time.time() - no_person_start_time >= no_person_delay_sec:
            if current_signal != 0:
                current_signal = 0
                print("🟡 감지 신호: 0 (2초 연속 감지 실패)")

    # ✅ 영상 출력
    cv2.imshow("YOLO Person Detection (Webcam)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
