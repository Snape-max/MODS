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
    ...