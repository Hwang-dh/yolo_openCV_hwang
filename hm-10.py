import serial

# 예: COM6으로 연결 (COM 포트는 장치 관리자에서 확인)
ser = serial.Serial('COM6', 9600)  # HM-10 기본 속도는 9600 또는 115200
print("Connected to HM-10")

# 데이터 송수신 예시
ser.write(b'Hello\n')
data = ser.readline()
print(data)
ser.close()
