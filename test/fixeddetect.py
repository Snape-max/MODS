import cv2
import numpy as np
video_path = "../video/target_center.mp4"

background_path = "../video/background.mp4"

def calculate_background(background_video_path: str):
    cap = cv2.VideoCapture(background_video_path)
    background_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background_frames.append(frame_gray)
    return np.mean(background_frames, axis=0).astype(np.uint8)


background = calculate_background(background_path)

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame_original = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(frame, background)

    diff[diff < 60] = 0

    # 大津法自适应二值化
    ret, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel_ero = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel_ero, iterations=2)
    kernel_dil = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dil, iterations=2)

    # 离散区域合并
    # 连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)


    threshold_distance = 200
    for i in range(1, num_labels):
        for j in range(i + 1, num_labels):
            # 计算两个组件中心点之间的距离
            distance = np.linalg.norm(centroids[i] - centroids[j])
            if distance < threshold_distance:
                # 合并组件
                x1, y1 = centroids[i]
                x2, y2 = centroids[j]

                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)

                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                cv2.rectangle(thresh, (x1, y1), (x2, y2), 255, -1)

    # 提取轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(frame_original, [contour], -1, (0, 255, 0), 2)

        # 最大外界矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(frame_original, [box], 0, (0, 0, 255), 2)


    cv2.imshow("diff", frame_original)
    cv2.waitKey(33)


cv2.destroyAllWindows()
