import cv2
import numpy as np
from typing import Callable
from numpy import ndarray
from utils import add_mask



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
    ...
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


        if cnts2:
            all_contours = np.vstack(cnts2)  # 将所有轮廓合并
            x, y, w, h = cv2.boundingRect(all_contours)  # 计算外接矩形
            cv2.rectangle(frame_detect2, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色检测框
        thresh2[thresh2 == 255] = 1
        mask_image = add_mask(frame_detect2[:, :, ::-1], thresh2.astype(int), (61, 132, 168))
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