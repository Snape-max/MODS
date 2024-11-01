<h1 align="center">
  <br>
  <img src="./asset/logo.png" alt="MODS" width="150">
  <br>
  <sub><sup><b>Moving Object Detection System</b></sup></sub>
</h1>


<p align="center">
    <a href="https://github.com/Snape-max/MODS" target="_blank" style="margin-right: 20px; font-style: normal; text-decoration: none;">
        <img src="https://img.shields.io/github/stars/Snape-max/MODS" alt="Github Stars" />
    </a>
    <a href="https://github.com/Snape-max/MODS" target="_blank" style="margin-right: 20px; font-style: normal; text-decoration: none;">
        <img src="https://img.shields.io/github/forks/Snape-max/MODS" alt="Github Forks" />
    </a>
    <a href="https://github.com/Snape-max/MODS" target="_blank" style="margin-right: 20px; font-style: normal; text-decoration: none;">
        <img src="https://img.shields.io/github/languages/code-size/Snape-max/MODS" alt="Code-size" />
    </a>
    <a href="https://github.com/Snape-max/MODS">
        <img src="https://img.shields.io/github/v/release/Snape-max/MODS"
            alt="Latest Release">
    </a>
</p>

## Requirements

- `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1` for `sam2`
- `opencv-python` and `opencv-contrib-python` for image processing
- `pyside6` for GUI
- `scikit-learn` for cluster

## How to run

1. Install the requirements using `pip install -r requirements.txt`, recommend to use `conda`
2. Install `sam2` following [this](https://github.com/facebookresearch/segment-anything)
3. Run `python windows.py`

## Interface display

<img src="./asset/gui.png" width="800" alt="GUI">
