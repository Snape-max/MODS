import cv2
import numpy as np
import time
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
    s_time = None
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
            cv2.drawContours(frame_original, [box], 0, (99, 46, 255), 2)


        thresh[thresh == 255] = 1
        mask_image = add_mask(frame_original[:, :, ::-1], thresh.astype(int), (16, 221, 194)).astype(np.uint8)
        e_time = time.time()
        if s_time is not None:
            fps = round(1 / (e_time - s_time), 2)
            cv2.putText(mask_image, str(fps), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        s_time = time.time()
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
    cap = cv2.VideoCapture(video_path)  # 替换为你的录像文件路径

    # 创建背景减除对象
    backSub = cv2.createBackgroundSubtractorMOG2(240, 16, True)

    while True:
        # 逐帧捕捉
        ret, frame = cap.read()
        if not ret:
            break

        # 应用背景减除
        fg_mask = backSub.apply(frame)
        fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)[1]
        # 进行形态学操作以去除噪声
        kernel_ero = np.ones((3, 3), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel_ero, iterations=2)

        kernel_dil = np.ones((3, 3), np.uint8)
        fg_mask = cv2.dilate(fg_mask, kernel_dil, iterations=3)

        fg_mask = cv2.erode(fg_mask, kernel_ero, iterations=1)
        fg_mask = cv2.dilate(fg_mask, kernel_dil, iterations=2)
        # fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        frame_detect = frame.copy()

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
        threshold_distance = 70
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

                    cv2.rectangle(fg_mask, (x1, y1), (x2, y2), 255, -1)

        # 查找轮廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cv2.drawContours(frame_detect, [contour], -1, (0, 255, 0), 2)

        for contour in contours:
            if cv2.contourArea(contour) > 200:  # 过滤小轮廓
                (x, y, w, h) = cv2.boundingRect(contour)

                cv2.rectangle(frame_detect, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 绘制边界框
        show_callback(frame_detect[:, :, ::-1])
    log_callback("processed")