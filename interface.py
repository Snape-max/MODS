import os
import numpy as np
import torch
from PIL import Image
from utils import set_devices, add_mask
from sam2.build_sam import build_sam2_video_predictor
from typing import Callable
from numpy import ndarray
from hansome_static_detect import *
from preprocessing import preprocessing

class Sam2Interface:
    def __init__(self, show_callback:Callable[[ndarray],None], log_callback:Callable[[str],None], video_dir:str):
        device = set_devices()
        # sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        log_callback("Loading sam2 model")
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        self.show_callback = show_callback
        self.log_callback = log_callback
        self.video_dir = video_dir


    def handle(self) -> None:
        if not os.path.isdir(self.video_dir):
            self.log_callback("Not a video directory")
            return
        self.log_callback("Loading video...")
        frame_names = [
            p for p in os.listdir(self.video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]



        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        inference_state = self.predictor.init_state(video_path=self.video_dir)
        self.log_callback("Video loading success...")
        ann_frame_idx = 10  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        # Let's add a positive click at (x, y) = (210, 350) to get started
        frame0 = cv2.imread(os.path.join(self.video_dir, frame_names[9]))
        frame1 = cv2.imread(os.path.join(self.video_dir, frame_names[10]))
        self.log_callback("Preprocessing frames...")
        move_rect = preprocessing(frame0, frame1)
        box = np.array(move_rect[0], dtype=np.float32)
        point_x = (move_rect[0][0] + move_rect[0][2])/2
        point_y = (move_rect[0][1] + move_rect[0][3]) / 2
        points = np.array([[point_x, point_y]], dtype=np.float32)
        # points = np.array([[947, 24]], dtype=np.float32)
        # box = np.array([892, 11, 971, 119], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
            box=box,
        )
        self.log_callback("Start propagate in video")
        image = np.array(Image.open(os.path.join(self.video_dir, frame_names[ann_frame_idx])).convert("RGB"))
        self.show_callback(image)

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            image = np.array(Image.open(os.path.join(self.video_dir, frame_names[out_frame_idx])))
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            mask_image = add_mask(image, mask[0], (30, 144, 255))
            self.show_callback(mask_image)

        self.log_callback("Propagate Finish")



class Cv2Interface:
    def __init__(self, show_callback:Callable[[ndarray],None], log_callback:Callable[[str],None],
                 video_path:str, task:str):
        self.show_callback = show_callback
        self.log_callback = log_callback
        self.video_path = video_path
        self.task = task


    def handle(self):
        if not os.path.isfile(self.video_path):
            self.log_callback("Not a video file")
            return
        if self.task == "背景不变":
            fixedbackground_detect(self.show_callback, self.log_callback, self.video_path)
        elif self.task == "背景变化":
            fluidbackground_detect(self.show_callback, self.log_callback, self.video_path)
        else:
            self.log_callback("Value Error")





