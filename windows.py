import os
import sys
import cv2
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QThread
from PySide6.QtGui import QIcon, QImage, QPixmap, Qt
from PySide6.QtWidgets import QGridLayout, QWidget, QLabel, QPushButton, QComboBox, QSpacerItem, QSizePolicy
from PySide6.QtCore import Signal, Slot
from interface import *

class WorkerThread(QThread):
    changePixmap = Signal(QImage)
    changeInfo = Signal(str)

    def __init__(self, video_path, task):
        super().__init__()
        self.video_path = video_path
        self.task = task

    def log_callback(self, string):
        self.changeInfo.emit(string)

    def show_callback(self, image):
        showframe = cv2.resize(image, (16 * 60, 9 * 60))
        h, w, ch = showframe.shape
        bytes_per_line = ch * w
        qImg = QImage(showframe.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.changePixmap.emit(qImg)

    def run(self) -> None:
        if self.task == "动平台":
            sam_test = Sam2Interface(self.show_callback, self.log_callback, self.video_path)
            sam_test.handle()

        elif self.task == "背景不变":
            cv2_test = Cv2Interface(self.show_callback, self.log_callback, self.video_path)
            cv2_test.handle()






class MoveWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.task = 0
        self.video_path = 0
        self.video_dir = "./video"
        self.task_box = QComboBox()
        self.video_box = QComboBox()
        self.start_button = QPushButton("开始运动目标检测")
        self.start_button.clicked.connect(self.start)
        self.frame_label = QLabel(self)
        self.set_ui()


    def set_ui(self):
        self.task_box.addItems(["背景不变","背景变化","动平台"])
        videos = os.listdir(self.video_dir)
        self.video_box.addItems(videos)
        central_widget = QWidget(self)
        grid = QGridLayout()
        central_widget.setLayout(grid)
        self.setCentralWidget(central_widget)
        self.setWindowTitle("运动目标识别")
        self.setGeometry(200, 100, 1150-30, 620-50)



        task_label = QLabel(self)
        task_label.setText("选择任务场景")
        video_label = QLabel(self)
        video_label.setText("选择视频文件")
        spacer = QSpacerItem(20, 20)

        # style
        task_label.setMaximumHeight(10)
        video_label.setMaximumHeight(10)
        self.video_box.setStyleSheet("QComboBox  { min-width: 100px; min-height: 30px; }")
        self.task_box.setStyleSheet("QComboBox { min-width: 100px; min-height: 30px; }")
        self.start_button.setStyleSheet("QPushButton { max-width: 120px; min-height: 30px; }")
        self.frame_label.setAlignment(Qt.AlignCenter)

        grid.addWidget(task_label, 0, 0)
        grid.addWidget(self.task_box, 1, 0)
        grid.addWidget(video_label, 2, 0)
        grid.addWidget(self.video_box, 3, 0)
        grid.addItem(spacer, 4, 0)
        grid.addWidget(self.start_button, 5, 0)
        grid.addWidget(self.frame_label, 0, 1,10,10)

    def start(self):
        task = self.task_box.currentText()
        video = self.video_box.currentText()
        video_path = os.path.join(self.video_dir, video)
        self.worker = WorkerThread(video_path, task)
        self.worker.changePixmap.connect(self.set_frame_label_pixmap)
        self.worker.changeInfo.connect(self.set_info)
        self.worker.start()

    @Slot(QImage)
    def set_frame_label_pixmap(self, image):
        self.frame_label.setPixmap(QPixmap.fromImage(image))

    @Slot(str)
    def set_info(self, info):
        self.frame_label.setText(info)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MoveWindow()
    window.show()
    sys.exit(app.exec())