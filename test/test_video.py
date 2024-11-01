import os
import numpy as np
import torch
from PIL import Image
from utils import set_devices, add_mask, show_mask
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
import cv2
import matplotlib.pyplot as plt

device = set_devices()
sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


video_dir = "video/drones"
# video_path = "./video/input.mp4"

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".drones", ".jpeg", ".JPG", ".JPEG"]
]

frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=video_dir)

ann_frame_idx = 10  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (210, 350) to get started
points = np.array([[947, 24]], dtype=np.float32)
box = np.array([892, 11,971,119], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
    box=box,
)

# mask = (out_mask_logits[0] > 0.0).cpu().numpy()
#
#
# image = np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])).convert("RGB"))
# mask_image = add_mask(image, mask[0], (30, 144, 255))
# cv2.imshow("image", mask_image)

# run propagation throughout the video and collect the results in a dict
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    image = np.array(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))[...,::-1]
    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
    mask_image = add_mask(image, mask[0], (30, 144, 255))
    cv2.imwrite("./output/{}.drones".format(out_frame_idx), mask_image)







