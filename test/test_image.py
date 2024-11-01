import os
import numpy as np
import torch
from PIL import Image
from utils import set_devices, add_mask
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2

device = set_devices()
image = Image.open('image/me.jpg')
image = np.array(image.convert("RGB"))

# ! resize
original_height, original_width = image.shape[:2]
new_width = 500  # 设置为500像素
ratio = new_width / float(original_width)
new_height = int(original_height * ratio)
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
cv2.imshow("image", cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
predictor.set_image(image)


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("click at ({}, {})".format(x, y))
        realx, realy = x / ratio, y / ratio
        input_point = np.array([[realx, realy]])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        # 原图大小输出掩码
        masks = masks[sorted_ind]
        # 置信度
        scores = scores[sorted_ind]
        # 256*256 缩放后掩码
        logits = logits[sorted_ind]

        mask_image = add_mask(image.copy(), masks[0], (30, 144, 255))

        resized_mask = cv2.resize(mask_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", resized_mask)


cv2.setMouseCallback("image", mouse_callback)
cv2.waitKey(0)
if cv2.waitKey(1) & 0xFF == ord("q"):
    cv2.destroyAllWindows()
