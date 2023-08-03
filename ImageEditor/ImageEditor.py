import sys
from PyQt6 import QtWidgets
from PyQt6 import uic
from PyQt6.QtGui import QPixmap, QImage, QTransform
import cv2
from model.correct_rotation import *
from keras.models import load_model
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QProgressDialog,
    QCheckBox
)
import qdarktheme

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
        self.correct_rotation_button.clicked.connect(self.start_rotation_correction)
        # Initialize variables
        self.selected_files = []
        self.output_folder = ""
        self.model = None
    def center_window(self):
        frame_geometry = self.frameGeometry()
        center_point = QApplication.primaryScreen().availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())
    
    def select_files(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)  # Allow multiple file selection
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
        if file_dialog.exec():
            filenames = file_dialog.selectedFiles()
            self.selected_files = filenames
            if len(filenames) == 1:
                self.file_label.setText("Erfolgreich {} Datei ausgewählt".format(len(filenames)))
            else:
                self.file_label.setText("Erfolgreich {} Dateien ausgewählt".format(len(filenames)))
            

    def select_folder(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        file_dialog.setOption(QFileDialog.Option.ReadOnly, True)
        if file_dialog.exec():
            folder = file_dialog.selectedFiles()
            self.selected_files = folder[0]  # Store the selected folder as a list
            self.file_label.setText("Ausgewählter Eingabeordner: ...{}".format(self.selected_files[len(self.selected_files)-20:]))

    def select_output_folder(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        file_dialog.setOption(QFileDialog.Option.ReadOnly, True)
        if file_dialog.exec():
            output_folder = file_dialog.selectedFiles()
            self.output_folder = output_folder[0]  # Store the selected folder as a list
            self.output_folder_label.setText("Ausgewählter Ausgabeordner: ...{}".format(self.output_folder[len(self.output_folder)-20:]))

    def load_model_thread(self):
        model_path = './model/efficientnetv2_sv_open_images.hdf5'  
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
        self.file_label.setText("Keine Dateien oder Ordner ausgewählt")
        self.output_folder_label.setText("Kein Ausgabeordner ausgewählt")

    def closeEvent(self, event):
        if hasattr(self, "rotation_thread") and self.rotation_thread.isRunning():
            self.rotation_thread.quit()
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
        self.autocorrectButton.clicked.connect(self.load_model_thread)
        self.open_button.triggered.connect(self.load_image)
        self.menuDatei.triggered.connect(self.load_model_thread)
        self.correct_rotation_button.clicked.connect(self.correct_rotation)
        self.save_button.clicked.connect(self.save_rotated_image)
        self.horizontalSlider_2.valueChanged.connect(self.update_rotation)
        self.horizontalSlider.valueChanged.connect(self.update_rotation_orig)
        self.plusButton_2.clicked.connect(self.rotate_button_plus_orig)
        self.minusButton_2.clicked.connect(self.rotate_button_minus_orig)
        self.plusButton.clicked.connect(self.rotate_button_plus)
        self.minusButton.clicked.connect(self.rotate_button_minus)
        self.actionDark_mode.triggered.connect(self.change_dark_mode)
        self.actionLight_mode.triggered.connect(self.change_light_mode)
        self.actionAuto.triggered.connect(self.change_auto_mode)

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
                self.horizontalSlider.setValue(0)
    def display_images(self):
        if self.original_image is None:
            original_pixmap = QPixmap.fromImage(self.original_image)
            self.original_label.setPixmap(original_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
        if self.rotated_image is not None:
            rotated_pixmap = QPixmap.fromImage(self.rotated_image)
            self.rotated_label.setPixmap(rotated_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
    def convertQImageToMat(incomingImage):
        #Convert a QImage into an opencv MAT format

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
            
            # Start the rotation correction in a separate thread
            self.rotation_thread = CorrectRotationThread_main(self.filename, self.rotation_angle_orig,self.model)
            self.rotation_thread.started.connect(progress_dialog.show)
            self.rotation_thread.finished.connect(progress_dialog.close)
            self.rotation_thread.rotation_completed.connect(self.rotation_completed)
            self.rotation_thread.finished.connect(lambda: self.correct_rotation_button.setEnabled(True))
            self.rotation_thread.start()
            progress_dialog.close()
    def rotation_completed(self):
        self.rotation_angle = self.rotation_thread.rotation_angle
        if self.rotation_angle > 180:
            update_slider = self.rotation_angle - 360
        elif self.rotation_angle < -180:
            update_slider = self.rotation_angle + 360
        else: 
            update_slider = self.rotation_angle
        if isinstance(update_slider,np.ndarray):
            update_slider = update_slider[0]
        # update slider
        self.horizontalSlider_2.setValue(update_slider)
        transform = QTransform()
        transform.rotate(self.rotation_thread.rotation_angle)
        self.rotated_image = self.original_image.transformed(transform)
        self.display_images()
        
    def load_model_thread(self):
        model_path = './model/efficientnetv2_sv_open_images.hdf5'  
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
    def rotate_button_plus_orig(self):
        self.rotation_angle_orig += 90
        if self.rotation_angle_orig > 180:
            update_slider = self.rotation_angle_orig - 360
        elif self.rotation_angle_orig < -180:
            update_slider = self.rotation_angle_orig + 360
        else: 
            update_slider = self.rotation_angle_orig
        if isinstance(update_slider,np.ndarray):
            update_slider = update_slider[0]
        self.horizontalSlider.setValue(update_slider)
        self.rotate_image_orig()
    def rotate_button_minus_orig(self):
        self.rotation_angle_orig -= 90
        if self.rotation_angle_orig > 180:
            update_slider = self.rotation_angle_orig - 360
        elif self.rotation_angle_orig < -180:
            update_slider = self.rotation_angle_orig + 360
        else: 
            update_slider = self.rotation_angle_orig
        if isinstance(update_slider,np.ndarray):
            update_slider = update_slider[0]
        self.horizontalSlider.setValue(update_slider)
        self.rotate_image_orig()

    def rotate_button_plus(self):
        self.rotation_angle += 90
        if self.rotation_angle > 180:
            update_slider = self.rotation_angle - 360
        elif self.rotation_angle < -180:
            update_slider = self.rotation_angle + 360
        else: 
            update_slider = self.rotation_angle
        if isinstance(update_slider,np.ndarray):
            update_slider = update_slider[0]
        self.horizontalSlider_2.setValue(update_slider)
        self.rotate_image()
    def rotate_button_minus(self):
        self.rotation_angle -= 90
        if self.rotation_angle > 180:
            update_slider = self.rotation_angle - 360
        elif self.rotation_angle < -180:
            update_slider = self.rotation_angle + 360
        else: 
            update_slider = self.rotation_angle
        if isinstance(update_slider,np.ndarray):
            update_slider = update_slider[0]
        self.horizontalSlider_2.setValue(update_slider)
        self.rotate_image()
    
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
                            image = rotate(image, -a)  
                            image = crop_largest_rectangle(image, a, height, width)
                            cv2.imwrite(filename,image)
                           
                        else:
                            self.rotated_image.save(filename)

qdarktheme.enable_hi_dpi()
app = QtWidgets.QApplication(sys.argv)
qdarktheme.setup_theme("auto")
window = MainWindow()
window.show()
app.exec()