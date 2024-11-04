import cv2
import numpy as np


video_path = "../video/fluid/dynamic.mp4"

cap = cv2.VideoCapture(video_path)

ret, old_frame = cap.read()
old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

ret, mid_frame = cap.read()

mid_frame = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    color_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    diff1 = cv2.absdiff(mid_frame, old_frame)
    diff2 = cv2.absdiff(frame, mid_frame)

    diff = diff1 + diff2

    diff[diff < 60] = 0

    ret, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel_ero = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel_ero, iterations=1)


    kernel_dil = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dil, iterations=1)




    old_frame = mid_frame
    mid_frame = frame
    thresh = cv2.resize(thresh, (640, 480), fx=0.5, fy=0.5)
    color_frame = cv2.resize(color_frame, (640, 480), fx=0.5, fy=0.5)
    cv2.imshow("color_frame", color_frame)
    cv2.imshow("diff", thresh)
    cv2.waitKey(1)

cv2.destroyAllWindows()