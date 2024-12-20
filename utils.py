import numpy as np
import torch
import cv2

def set_devices():
    """
    select the device for computation
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    return device


def add_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """
    在图像上添加遮罩。

    参数:
    image: np.ndarray - 输入的图像数组。
    mask: np.ndarray - 遮罩数组，用于指定遮罩区域。
    color: tuple[int, int, int] - 遮罩的颜色，以RGB格式表示。

    返回:
    np.ndarray - 添加遮罩后的图像数组。
    """
    mask_image = np.zeros_like(image)
    # 遮罩
    mask_image[mask==1] = color
    # 提取原图遮罩区域
    image_mask_area = image[mask==1]
    # 添加遮罩
    if len(image_mask_area) != 0:
        masked_image = cv2.addWeighted(image_mask_area, 0.6, mask_image[mask == 1], 0.4, 0,
                                       dtype=cv2.CV_32F).astype(np.uint8)
        # 替换区域
        image[mask == 1] = masked_image
    return image


def calculate_background(background_video_path: str) -> np.ndarray:
    """
    计算视频的平均背景帧。
    参数:
    background_video_path: str - 视频文件的路径。

    返回:
    np.ndarray - 计算得到的灰度平均背景帧。
    """
    cap = cv2.VideoCapture(background_video_path)
    background_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background_frames.append(frame_gray)
    return np.mean(background_frames, axis=0).astype(np.uint8)

