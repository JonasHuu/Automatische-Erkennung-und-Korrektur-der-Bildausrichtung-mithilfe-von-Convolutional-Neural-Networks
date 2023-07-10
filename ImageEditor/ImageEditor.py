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
import qdarktheme
dark_theme = """
    QWidget {
        background-color: #333333;
        color: #ffffff;
        border: none;
    }
    QPushButton {
        background-color: #4d4d4d;
        border: 1px solid #4d4d4d;
        border-radius: 4px;
        color: #ffffff;
        padding: 5px;
    }
    QPushButton:hover {
        background-color: #5a5a5a;
        border: 1px solid #5a5a5a;
    }
    QCheckBox {
        color: #ffffff;
    }
    QLineEdit {
        background-color: #4d4d4d;
        border: 1px solid #4d4d4d;
        color: #ffffff;
        padding: 5px;
    }
    QTextEdit {
        background-color: #4d4d4d;
        border: 1px solid #4d4d4d;
        color: #ffffff;
        padding: 5px;
    }
    QProgressBar {
        border: 1px solid #444444;
        border-radius: 7px;
        background-color: #2e2e2e;
        text-align: center;
        font-size: 10pt;
        color: white;
    }
    QProgressBar::chunk {
        background-color: #3a3a3a;
        width: 5px;
    }
    QScrollBar:vertical {
        border: none;
        background-color: #3a3a3a;
        width: 10px;
        margin: 16px 0 16px 0;
    }
    QScrollBar::handle:vertical {
        background-color: #444444;
        border-radius: 5px;
    }
    QScrollBar:horizontal {
        border: none;
        background-color: #3a3a3a;
        height: 10px;
        margin: 0px 16px 0 16px;
    }
    QScrollBar::handle:horizontal {
        background-color: #444444;
        border-radius: 5px;
    }
    QTabWidget {
        background-color: #2e2e2e;
        border: none;
    }
    QTabBar::tab {
        background-color: #2e2e2e;
        color: #b1b1b1;
        padding: 8px 20px;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
        border: none;
    }

    QTabBar::tab:selected, QTabBar::tab:hover {
        background-color: #3a3a3a;
        color: white;
    }
"""
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
class CorrectRotationThread(QThread):
    rotation_completed = pyqtSignal()

    def __init__(self, selected_files, output_folder, model, parent=None):
        super().__init__(parent)
        self.selected_files = selected_files
        self.output_folder = output_folder
        self.model = model
        self.running = True

    def quit(self):
        self.running = False
        super().quit()

    def run(self):
        output_path = self.output_folder
        file_path = self.selected_files

        if self.model is None:
            self.model = load_model('./model/efficientnetv2_sv_open_images.hdf5', custom_objects={'angle_error': angle_error})

        process_images(self.model, self.selected_files, output_path, batch_size=64, crop=True)

        if self.running:
            self.rotation_completed.emit()

class CorrectRotationThread_main(QThread):
    rotation_completed = pyqtSignal()

    def __init__(self, selected_files, original_angle,model, parent=None):
        super().__init__(parent)
        self.filename = selected_files
        self.model = model
        self.running = True
        self.rotation_angle_orig = original_angle
        self.rotation_angle = 0

    def quit(self):
        self.running = False
        super().quit()

    def run(self):
        if self.model is None:
                self.model = load_model('./model/efficientnetv2_sv_open_images.hdf5', custom_objects={'angle_error': angle_error})

        #transform = QTransform()
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
                angle=self.rotation_angle_orig
            )
        )
        self.rotation_angle = np.argmax(predictions, axis=1) - self.rotation_angle_orig

        if self.running:
            self.rotation_completed.emit()
# load previous app so it can be reused as a dialog
class Autocorrect(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        # load the .ui file
        uic.loadUi("autocorrect_dialog.ui", self)
        
        
        # connect buttons
        self.file_button.clicked.connect(self.select_files)
        self.folder_button.clicked.connect(self.select_folder)
        self.output_folder_button.clicked.connect(self.select_output_folder)
        self.folder_button.clicked.connect(self.load_model_thread)
        self.file_button.clicked.connect(self.load_model_thread)
        self.correct_rotation_button.clicked.connect(self.start_rotation_correction)
        self.correct_rotation_button.clicked.connect(self.button_click)
        # Initialize variables
        self.selected_files = []
        self.output_folder = ""
    def button_click(self):
        pass
    def center_window(self):
        frame_geometry = self.frameGeometry()
        center_point = QApplication.primaryScreen().availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())
    
    def select_files(self):
        options = QFileDialog()
        options = options.options()
        options |= QFileDialog.Option.DontUseNativeDialog
        file_dialog = QFileDialog()

        # Open the file dialog and allow the user to select multiple files or a folder
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        file_dialog.setOption(QFileDialog.Option.ReadOnly, True)
        file_names, _ = file_dialog.getOpenFileNames(
            self,
            "Select Files",
            "",
            "Images (*.png *.bmp *.jpg *.jpeg)",
            options=options,
        )

        # Check if any files or folder were selected
        if file_names:
            self.selected_files = file_names
            self.file_label.setText("Erfolgreich {} Datei/en ausgewählt".format(len(file_names)))
        self.model = None

    def select_folder(self):
        options = QFileDialog()
        options = options.options()
        options |= QFileDialog.Option.DontUseNativeDialog
        file_dialog = QFileDialog()

        # Open the file dialog and allow the user to select a folder
        file_dialog.setFileMode(QFileDialog.FileMode.Directory) 
        file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        file_dialog.setOption(QFileDialog.Option.ReadOnly, True)
        folder = file_dialog.getExistingDirectory(
            self,
            "Select Folder",
            options=options
        )

        # Check if any files or folder were selected
        if folder:
            self.selected_files = folder  # Store the selected folder as a list
            self.file_label.setText("Erfolgreich einen Ordner ausgewählt")
    def select_output_folder(self):
        options = QFileDialog()
        options = options.options()
        options |= QFileDialog.Option.DontUseNativeDialog
        file_dialog = QFileDialog()

        # Open the file dialog and allow the user to select the output folder
        output_folder = file_dialog.getExistingDirectory(
            self, "Ordner auswählen", options=options
        )

        # Check if an output folder was selected
        if output_folder:
            self.output_folder = output_folder
            self.output_folder_label.setText("Ausgewählter Ausgabeordner: {}".format(output_folder[:10]))
    def load_model_thread(self):
        model_path = './model/efficientnetv2_sv_open_images.hdf5'  
        #model_path = './rotnet_open_images_resnet50_TCML_2.hdf5'
        # Create a thread to load the model
        self.model_thread = LoadModelThread(model_path)
        self.model_thread.model_loaded.connect(self.model_loaded)
        self.model_thread.start()
    def model_loaded(self, model_location):
        self.model = model_location
    def start_rotation_correction(self):
        # Disable the button during the process
        self.correct_rotation_button.setEnabled(False)

        # Create and configure the progress dialog
        progress_dialog = QProgressDialog(self)
        progress_dialog.setWindowTitle("Bildorientierung korrigieren")
        progress_dialog.setLabelText("Bild/er werden verarbeitet...")
        progress_dialog.setCancelButtonText("Abbrechen")
        progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress_dialog.setRange(0, 0)  # Indeterminate progress
        progress_dialog.show()

        # Start the rotation correction in a separate thread
        self.rotation_thread = CorrectRotationThread(self.selected_files, self.output_folder, self.model)
        self.rotation_thread.started.connect(progress_dialog.show)
        self.rotation_thread.finished.connect(progress_dialog.close)
        self.rotation_thread.rotation_completed.connect(self.rotation_completed)
        self.rotation_thread.finished.connect(lambda: self.correct_rotation_button.setEnabled(True))
        self.rotation_thread.start()

    def rotation_completed(self):
        self.output_label.setText("Bildorientierung erfolgreich korrigiert!")

        # Reset the selected files and output folder
        self.selected_files = []
        self.output_folder = ""
        self.file_label.setText("Keine Dateien ausgewählt")
        self.output_folder_label.setText("Kein Ausgabeordner ausgewählt")

    def closeEvent(self, event):
        if hasattr(self, "rotation_thread") and self.rotation_thread.isRunning():
            self.rotation_thread.stop()
            self.rotation_thread.wait()
        event.accept()

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
        self.autocorrectButton.clicked.connect(self.autocorrect)
        self.open_button.triggered.connect(self.load_image)
        self.open_button.triggered.connect(self.load_model_thread)
        self.correct_rotation_button.clicked.connect(self.correct_rotation)
        self.save_button.clicked.connect(self.save_rotated_image)
        self.horizontalSlider_2.valueChanged.connect(self.update_rotation)
        self.horizontalSlider.valueChanged.connect(self.update_rotation_orig)
        self.actionDark_mode.triggered.connect(self.change_dark_mode)
        self.actionLight_mode.triggered.connect(self.change_light_mode)
        self.actionAuto.triggered.connect(self.change_auto_mode)
        #self.correct_rotation_button.clicked.connect(self.start_rotation_correction)
        # Initialize variables
        self.original_image = None
        self.rotated_image = None
        self.rotation_angle = 0
        self.rotation_angle_orig = 0
        self.filename = None
        self.remove_black_border = False
        self.model = None

    def change_dark_mode(self):
        qdarktheme.setup_theme("dark")
    def change_light_mode(self):
        qdarktheme.setup_theme("light")
    def change_auto_mode(self):
        qdarktheme.setup_theme("auto")
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
            progress_dialog = QProgressDialog(self)
            progress_dialog.setWindowTitle("Bildorientierung korrigieren")
            progress_dialog.setLabelText("Bild verarbeiten...")
            progress_dialog.setCancelButtonText("Abbrechen")
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setRange(0, 0)  # Indeterminate progress
            progress_dialog.show()

            #QCoreApplication.processEvents()  # Allow the progress dialog to be displayed
            # Start the rotation correction in a separate thread
            self.rotation_thread = CorrectRotationThread_main(self.filename, self.rotation_angle_orig,self.model)
            self.rotation_thread.started.connect(progress_dialog.show)
            self.rotation_thread.finished.connect(progress_dialog.close)
            self.rotation_thread.finished.connect(lambda: self.correct_rotation_button.setEnabled(True))
            self.rotation_thread.start()
            #if self.model is None:
            #    self.model = load_model('./model/efficientnetv2_sv_open_images.hdf5', custom_objects={'angle_error': angle_error})

            transform = QTransform()
            '''
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
                    angle=self.rotation_angle_orig
                )
            )
            self.rotation_angle = np.argmax(predictions, axis=1) - self.rotation_angle_orig
            '''
            transform.rotate(self.rotation_thread.rotation_angle)
            self.rotated_image = self.original_image.transformed(transform)
            self.display_images()

            progress_dialog.close()
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
    def autocorrect(self):
        window = Autocorrect(self)
        window.show()

    def save_rotated_image(self):
        if self.rotated_image is not None:
            dialog = Dialog(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self.remove_black_border = dialog.is_remove_black_borders_checked()

                file_dialog = QFileDialog()
                file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
                file_dialog.setDefaultSuffix("png")
                #file_dialog.setNameFilter("Images (*.png *.jpeg *.jpg *.bmp)")
                if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                    filenames = file_dialog.selectedFiles()
                    if filenames:
                        filename = filenames[0]
                        if self.remove_black_border:
                            image = cv2.imread(self.filename[0])
                            if isinstance(self.rotation_angle,int):
                                a = self.rotation_angle
                            else:
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

qdarktheme.enable_hi_dpi()
app = QtWidgets.QApplication(sys.argv)
#app.setStyleSheet(dark_theme)
qdarktheme.setup_theme("auto")
window = MainWindow()
window.show()
app.exec()