import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import streamlit as st

def set_devices():
    # select the device for computation
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
    mask_image = np.zeros_like(image)
    # 遮罩
    mask_image[mask == 1] = color
    # 提取原图遮罩区域
    image_mask_area = image[mask == 1]
    # 添加遮罩
    masked_image = cv2.addWeighted(image_mask_area, 0.7, mask_image[mask == 1], 0.3, 0,
                                   dtype=cv2.CV_32F).astype(np.uint8)
    # 替换区域
    image[mask == 1] = masked_image
    return image




