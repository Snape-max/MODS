import cv2
import numpy as np
from typing import Callable
from numpy import ndarray
from utils import add_mask, calculate_background




def fixedbackground_detect(show_callback:Callable[[ndarray],None],
                           log_callback:Callable[[str],None],
                           video_path:str) -> None:
    """
        检测固定背景下的运动物体。

    参数:
        show_callback (Callable[[ndarray], None]): 显示处理后的图像帧的回调函数, 接受处理后numpy图像矩阵
        log_callback (Callable[[str], None]): 用于记录日志信息的回调函数
        video_path (str): 视频文件的路径

    返回:
        None
    """
    background_path = "./video/background.mp4"
    log_callback("Background calculating")
    background = calculate_background(background_path)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
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
            # 最大外界矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(frame_original, [box], 0, (0, 0, 255), 2)

        thresh[thresh == 255] = 1
        mask_image = add_mask(frame_original[:, :, ::-1], thresh.astype(int), (61, 132, 168))
        show_callback(mask_image)
    log_callback("processed")


def fluidbackground_detect(show_callback:Callable[[ndarray],None],
                           log_callback:Callable[[str],None],
                           video_path:str) -> None:
    """
    检测动态背景下的运动物体。

    参数:
        show_callback (Callable[[ndarray], None]): 用于显示处理后的图像帧的回调函数
        log_callback (Callable[[str], None]): 用于记录日志信息的回调函数
        video_path (str): 视频文件的路径

    返回:
        None
    """
    camera = cv2.VideoCapture(video_path)
    # 获取视频的宽度和高度
    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))

    # 初始化当前帧的前帧
    lastFrame = None

    # kernel of erode and dilate
    kernel_ero = np.ones((3, 3), np.uint8)
    kernel_dil = np.ones((10, 10), np.uint8)
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        if lastFrame is None:
            lastFrame = frame
            continue

        frameDelta = cv2.absdiff(lastFrame, frame)
        lastFrame = frame.copy()
        gray = cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY)

        thresh2 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]  # 另一个阈值

        thresh2 = cv2.erode(thresh2, kernel_ero, iterations=1)
        thresh2 = cv2.dilate(thresh2, kernel_dil, iterations=2)

        cnts2, hierarchy2 = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_detect2 = frame.copy()

        # for c in cnts2:
        #     if cv2.contourArea(c) < 300:
        #         continue
        #     (x, y, w, h) = cv2.boundingRect(c)
        #     cv2.rectangle(frame_detect2, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色检测框

        if cnts2:
            all_contours = np.vstack(cnts2)  # 将所有轮廓合并
            x, y, w, h = cv2.boundingRect(all_contours)  # 计算外接矩形
            cv2.rectangle(frame_detect2, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色检测框

        show_callback(frame_detect2[:, :, ::-1])
    log_callback("processed")