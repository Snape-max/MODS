import streamlit as st
import os
from interface import *

st.set_page_config(
    page_title="运动目标检测",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )
video_dir = "./video"
sidebar = st.sidebar
task = sidebar.selectbox("任务场景", ["背景不变","背景变化","动平台"])
video = sidebar.selectbox("选择视频文件", os.listdir(video_dir))
sidebar.html("<br>")
start_button = sidebar.button("开始运动目标检测")
st_frame = st.empty()


if task == "背景不变":
    while start_button:
        video_path = os.path.join(video_dir, video)
        cv2_test = Cv2Interface(st_frame, "./output/output.mp4")
        cv2_test.handle()
        start_button = False

elif task == "背景变化":
    st.write("动平台")

elif task == "动平台":
    while start_button:
        video_path = os.path.join(video_dir, video)
        sam_test = Sam2Interface(st_frame, video_path)
        sam_test.handle()
        start_button = False