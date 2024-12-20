import os
import sys
from PySide6 import QtWidgets
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, Qt, QFont, QIcon, QFontDatabase
from PySide6.QtWidgets import QGridLayout, QWidget, QLabel, QPushButton, QComboBox, QSpacerItem
from numpy import ndarray

from interface import *

class WorkerThread(QThread):
    changePixmap = Signal(QImage)
    changeInfo = Signal(str)
    finished = Signal(bool)

    def __init__(self, video_path, task):
        super().__init__()
        self.video_path = video_path
        self.task = task
        self.is_running = True

    def log_callback(self, string: str) -> None:
        self.changeInfo.emit(string)

    def show_callback(self, image: ndarray) -> None:
        showframe = cv2.resize(image, (16 * 60, 9 * 60))
        h, w, ch = showframe.shape
        bytes_per_line = ch * w
        qImg = QImage(showframe.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.changePixmap.emit(qImg)

    def run(self) -> None:
        if self.task == "动平台":
            sam_test = Sam2Interface(self.show_callback, self.log_callback, self.video_path)
            sam_test.handle()
        elif self.task == "背景不变" or self.task == "背景变化":
            cv2_test = Cv2Interface(self.show_callback, self.log_callback, self.video_path, self.task)
            cv2_test.handle()
        else:
            self.log_callback("Task Error")
        self.finished.emit(True)

    def stop(self):
        self.is_running = False
        self.finished.emit(True)
        self.terminate()
        self.wait()

class MoveWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.task = 0
        self.video_path = 0
        self.task_dir = {
            "背景不变": "./video/fixed",
            "背景变化": "./video/fluid",
            "动平台": "./video/move"
        }
        self.task_box = QComboBox()
        self.task_box.currentIndexChanged.connect(self.set_video_box)
        self.video_box = QComboBox()
        self.start_button = QPushButton("开始运动目标检测")
        self.stop_button = QPushButton("终止任务")
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.frame_label = QLabel(self)
        self.task_finish = True
        self.worker = None
        self.set_ui()

    def set_ui(self):
        icon = QIcon("asset/logo.png")
        self.setWindowIcon(icon)
        self.task_box.addItems(["背景不变", "背景变化", "动平台"])
        # 当前task
        self.set_video_box()
        central_widget = QWidget(self)
        grid = QGridLayout()
        central_widget.setLayout(grid)
        self.setCentralWidget(central_widget)
        self.setWindowTitle("运动目标识别")
        self.setGeometry(200, 100, 1150 - 30, 620 - 50)

        task_label = QLabel(self)
        task_label.setText("选择任务场景")
        video_label = QLabel(self)
        video_label.setText("选择视频文件")
        spacer = QSpacerItem(20, 20)

        # style
        task_label.setMaximumHeight(15)
        video_label.setMaximumHeight(15)
        self.video_box.setStyleSheet("QComboBox  { min-width: 100px; min-height: 30px; }")
        self.task_box.setStyleSheet("QComboBox { min-width: 100px; min-height: 30px; }")
        self.start_button.setStyleSheet("QPushButton { max-width: 120px; min-height: 30px; }")
        self.stop_button.setStyleSheet("QPushButton { max-width: 120px; min-height: 30px; }")
        self.frame_label.setAlignment(Qt.AlignCenter)

        grid.addWidget(task_label, 0, 0)
        grid.addWidget(self.task_box, 1, 0)
        grid.addWidget(video_label, 2, 0)
        grid.addWidget(self.video_box, 3, 0)
        grid.addItem(spacer, 4, 0)
        grid.addWidget(self.start_button, 5, 0)
        grid.addWidget(self.stop_button, 6, 0)
        grid.addWidget(self.frame_label, 0, 1, 10, 10)
        self.frame_label.setText("Wait operation")

    def start(self):
        if self.task_finish:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            task = self.task_box.currentText()
            video = self.video_box.currentText()
            video_path = os.path.join(self.task_dir[task], video)
            self.worker = WorkerThread(video_path, task)
            self.worker.changePixmap.connect(self.set_frame_label_pixmap)
            self.worker.changeInfo.connect(self.set_info)
            self.worker.finished.connect(self.worker_finish)
            self.worker.start()

    def stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.task_finish = True

    @Slot(QImage)
    def set_frame_label_pixmap(self, image):
        self.frame_label.setPixmap(QPixmap.fromImage(image))

    @Slot(str)
    def set_info(self, info):
        self.frame_label.setText(info)

    @Slot(bool)
    def worker_finish(self, finish):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.task_finish = True

    def set_video_box(self):
        task = self.task_box.currentText()
        self.video_box.clear()
        videos = os.listdir(self.task_dir[task])
        # 找到以.mp4结尾的文件或文件夹
        videos = [video for video in videos 
                  if video.endswith(".mp4") or os.path.isdir(os.path.join(self.task_dir[task], video))]
        self.video_box.addItems(videos)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    font_id = QFontDatabase.addApplicationFont("asset/font.ttf")
    font_families = QFontDatabase.applicationFontFamilies(font_id)
    font = QFont(font_families[0], 10)
    app.setFont(font)
    window = MoveWindow()
    window.show()
    sys.exit(app.exec())