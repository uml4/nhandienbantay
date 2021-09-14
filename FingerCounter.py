import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# tên folder chứa hình 
folderPath = "finger"
myList = os.listdir(folderPath)
print(myList)

overlayList = []

# lấy thư mục chứa hình
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))


pTime = 0
detector = htm.handDetector(detectionCon=0.75)
# ID của 4 đầu ngón tay
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)

    # Danh sách ID và tọa độ của ID
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    # nếu phát hiện có bàn tay mới thực hiện
    if len(lmList) != 0:
        fingers = []
        #  Khuyết điểm chỉ đúng với tay phải do chỉ đang xử lý ngón cái ở tay phải
        # Thumb  ngón cái chỉ có 1 khớp ID =3 nên chỉ tipIds[0] - 1 ,
        #  nếu cX của ID4 mà nhỏ hơn ID 3 có nghĩa là ngón cái đang gập xuống 0 (ko dùng y vì có khả năng gập ngón cái ngang)
        # bình thường ngón cái luôn chỉa ra ngoài nên x của ID4 > Id3
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1) 
        else:
            fingers.append(0)
        # 4 Fingers
        # so sánh chiều cao(tọa độ y ) của đầu ngón với các khớp (-2)
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        # đếm tổng ngón tay - chỉ đém các ngón có gia trị = 1
        totalFingers = fingers.count(1)
        print(totalFingers)

        # lấy hình theo thứ từ trong mảng overlayList        
        h, w, c = overlayList[totalFingers].shape
        # chèn hình lấy được từ mảng overlayList vào khung hình frame hiện tại
        img[0:h, 0:w] = overlayList[totalFingers]
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)
                    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)