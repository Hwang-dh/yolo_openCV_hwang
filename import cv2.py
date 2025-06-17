import cv2
import torch

# YOLOv5 모델 로딩 (YOLOv8일 경우 ultralytics 패키지 사용)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 처음 실행 시 다운로드됨

cap = cv2.VideoCapture(0)  # 웹캠 사용

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0]  # x1, y1, x2, y2, conf, class

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)

        # 넓이가 2000 이상인 경우만 처리
        if area > 2000:
            # 사각형과 텍스트 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Area: {area}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()