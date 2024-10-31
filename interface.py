from time import sleep
import os
import numpy as np
import torch
from PIL import Image
from numpy import ndarray

from utils import set_devices, add_mask
from sam2.build_sam import build_sam2_video_predictor
import cv2
from typing import Callable
from numpy import ndarray

class Sam2Interface:
    def __init__(self, show_callback:Callable[[ndarray],None], log_callback:Callable[[str],None], video_dir:str):
        device = set_devices()
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        log_callback("Loading sam2 model")
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        self.show_callback = show_callback
        self.log_callback = log_callback
        self.video_dir = video_dir


    def handle(self) -> None:
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
        points = np.array([[947, 24]], dtype=np.float32)
        box = np.array([892, 11, 971, 119], dtype=np.float32)
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
    def __init__(self, show_callback:Callable[[ndarray],None], log_callback:Callable[[str],None], video_dir:str):
        self.show_callback = show_callback
        self.log_callback = log_callback
        self.video_dir = video_dir


    def handle(self): 
        capture = cv2.VideoCapture(self.video_dir)
        fps = 30
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                self.show_callback(np.array(frame)[..., ::-1])
            else:
                break
            sleep(1/fps-0.01)


        capture.release()
        self.log_callback("Process Finish...")




