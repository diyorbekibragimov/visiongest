import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import numpy as np
import pyautogui

################
wCam, hCam = 640, 480
frameR  = 100 # Frame Reduction
smoothening = 5
################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.handDetector(detectionCon=0.7)
fingers_apart = False

screen_width, screen_height = pyautogui.size()

while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lmList, bbox = detector.findPosition(frame, draw=True)

    if len(lmList) != 0:
        thumbX, thumbY = lmList[4][1], lmList[4][2] # Thumb
        indexX, indexY = lmList[8][1], lmList[8][2] # Index
        middleX, middleY = lmList[12][1], lmList[12][2] # Middle

        # cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        fingers = detector.fingersUp()

        click_distance = ((thumbX - indexX) ** 2 + (thumbY - indexY) ** 2) ** 0.5 # Distance between thumb and middle finger

        # Move
        if (fingers[1] == 1 and fingers[2] == 0 
            and fingers[3] == 0 and fingers[4] == 0
            and fingers[0] == 0):

            moveX = np.interp(indexX, (frameR, wCam-frameR), (0, screen_width))
            moveY = np.interp(indexY, (frameR, hCam-frameR), (0, screen_height))

            # Smoothen the values
            clocX = plocX + (moveX - plocX) / smoothening
            clocY = plocY + (moveY - plocY) / smoothening

            pyautogui.moveTo(screen_width-clocX, clocY)
            cv2.circle(frame, (indexX, indexY), 10, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Click
        if (fingers[0] == 1 and fingers[1] == 1
            and fingers[2] == 0 and fingers[3] == 0
            and fingers[4] == 0):

            if click_distance < 50:
                if fingers_apart:
                    cx, cy = (thumbX + indexX) // 2, (thumbY + indexY) // 2
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    pyautogui.click()
                fingers_apart = False
            else:
                fingers_apart = True

        cv2.rectangle(frame, (frameR, frameR), (wCam - frameR, hCam - frameR), 
                        (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    
cap.release()
cv2.destroyAllWindows()