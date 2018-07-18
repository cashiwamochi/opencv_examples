import cv2
import numpy as np

if __name__ == "__main__":
    count = 0
    cap = cv2.VideoCapture(1)
    while 1:
        status, frame = cap.read()
        k = cv2.waitKey(10)
        cv2.imshow('camera',frame)
        if k == ord('s'):
            cv2.imwrite(str(count)+'.jpg',frame)
            count = count + 1
            print('wrote!')
        elif k == 27:
            break
