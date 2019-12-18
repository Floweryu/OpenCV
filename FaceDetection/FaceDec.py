import cv2

cap = cv2.VideoCapture(0)  # 使用内置摄像头
face_cascade = cv2.CascadeClassifier(
    'FaceDetection/haarcascade_frontalface_default.xml')  # 加载人脸特征库

if cap.isOpened() == 0:
    print("error")

while(True):
    ret, frame = cap.read()  # 读取一帧的图像

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转灰

    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5))  # 检测人脸

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 用矩形圈出人脸

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 释放摄像头
cv2.destroyAllWindows()
