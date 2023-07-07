import sys
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6 import uic
from PyQt6.QtGui import QPixmap, QImage, QTransform
import cv2
import os

from model.correct_rotation import *
import os
from keras.models import load_model
from PyQt6.QtCore import Qt, QCoreApplication, QThread, pyqtSignal, QThreadPool
from PyQt6.QtGui import QIcon, QMovie
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QProgressDialog,
    QCheckBox
)
class Dialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Schwarze Ränder entfernen")
        self.layout = QtWidgets.QVBoxLayout(self)

        self.checkbox = QCheckBox("Schwarze Ränder entfernen?")
        self.layout.addWidget(self.checkbox)

        button_box = QtWidgets.QDialogButtonBox()
        button_box.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.layout.addWidget(button_box)

    def is_remove_black_borders_checked(self):
        return self.checkbox.isChecked()
class LoadModelThread(QThread):
    model_loaded = pyqtSignal(object)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        model_location = load_model(self.model_path, custom_objects={'angle_error': angle_error})
        self.model_loaded.emit(model_location)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("form.ui", self)
        
        # Connect signals and slots
        self.open_button.triggered.connect(self.load_image)
        self.open_button.triggered.connect(self.load_model_thread)
        self.correct_rotation_button.clicked.connect(self.correct_rotation)
        self.save_button.clicked.connect(self.save_rotated_image)
        self.horizontalSlider_2.valueChanged.connect(self.update_rotation)
        self.horizontalSlider.valueChanged.connect(self.update_rotation_orig)
        #self.correct_rotation_button.clicked.connect(self.start_rotation_correction)
        # Initialize variables
        self.original_image = None
        self.rotated_image = None
        self.rotation_angle = 0
        self.rotation_angle_orig = 0
        self.filename = None
        self.remove_black_border = False
        self.model = None
    def load_image(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
        if file_dialog.exec():
            filenames = file_dialog.selectedFiles()
            if filenames:
                filename = filenames[0]
                self.filename = filenames
                image = cv2.imread(filename)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                self.original_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                original_pixmap = QPixmap.fromImage(self.original_image)
                self.original_label.setPixmap(original_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))

    def display_images(self):
        if self.original_image is None:
            original_pixmap = QPixmap.fromImage(self.original_image)
            self.original_label.setPixmap(original_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
        if self.rotated_image is not None:
            rotated_pixmap = QPixmap.fromImage(self.rotated_image)
            self.rotated_label.setPixmap(rotated_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
    def convertQImageToMat(incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = incomingImage.convertToFormat(4)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        return arr
    
    def correct_rotation(self):
        if self.original_image is not None:
            if self.model == None:
                self.model = load_model('./model/efficientnetv2_sv_open_images.hdf5', custom_objects={'angle_error': angle_error})
            transform = QTransform()
            predictions = self.model.predict(
            RotNetDataGenerator(
                    self.filename,
                    input_shape=(224, 224, 3),
                    batch_size=64,
                    one_hot=True,
                    preprocess_func=preprocess_input,
                    rotate=True,
                    crop_largest_rect=True,
                    crop_center=True,
                    angle = self.rotation_angle_orig
                )
            )
            self.rotation_angle = np.argmax(predictions, axis=1) - self.rotation_angle_orig
            transform.rotate(self.rotation_angle)
            self.rotated_image = self.original_image.transformed(transform)
            self.display_images()
    def load_model_thread(self):
        model_path = './model/efficientnetv2_sv_open_images.hdf5'  
        #model_path = './rotnet_open_images_resnet50_TCML_2.hdf5'
        # Create a thread to load the model
        self.model_thread = LoadModelThread(model_path)
        self.model_thread.model_loaded.connect(self.model_loaded)
        self.model_thread.start()
    def model_loaded(self, model_location):
        self.model = model_location     
            

    def rotate_image(self):
        if self.original_image is not None:
            transform = QTransform()
            transform.rotate(self.rotation_angle)
            self.rotated_image = self.original_image.transformed(transform)
            self.display_images()
    def update_rotation(self, value):
        self.rotation_angle = value
        self.rotate_image()

    def update_rotation_orig(self, value):
        self.rotation_angle_orig = value
        self.rotate_image_orig()

    def rotate_image_orig(self):
        if self.original_image is not None:
            transform = QTransform()
            transform.rotate(self.rotation_angle_orig)
            original_pixmap = QPixmap.fromImage(self.original_image.transformed(transform))
            self.original_label.setPixmap(original_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))

    def save_rotated_image(self):
        if self.rotated_image is not None:
            dialog = Dialog(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self.remove_black_border = dialog.is_remove_black_borders_checked()

                file_dialog = QFileDialog()
                file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
                if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                    filenames = file_dialog.selectedFiles()
                    if filenames:
                        filename = filenames[0]
                        if self.remove_black_border:
                            image = cv2.imread(self.filename[0])
                            a = self.rotation_angle[0]
                            height, width = image.shape[:2]
                            if width < height:
                                height = width
                            else:
                                width = height
                            image = rotate(image, -a)  
                            image = crop_largest_rectangle(image, a, height, width)
                            cv2.imwrite(filename,image)
                           
                        else:
                            self.rotated_image.save(filename)
    
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()