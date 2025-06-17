import serial
import time

ser = serial.Serial('COM4', 9600, timeout=0.5)  # 포트 확인 필수!
time.sleep(1)
ser.write(b"1\n")
print("✅ 1번 신호 전송 완료")