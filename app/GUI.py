import os
import sys
from model.correct_rotation import *
import os
import sys
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
    QProgressDialog
)
#sys.setrecursionlimit(5000)  # Increase recursion limit to avoid issues with Keras

class LoadModelThread(QThread):
    model_loaded = pyqtSignal(object)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        model_location = load_model(self.model_path, custom_objects={'angle_error': angle_error})
        self.model_loaded.emit(model_location)

class CorrectRotationThread(QThread):
    rotation_completed = pyqtSignal()
    def __init__(self, selected_files, output_folder, model, parent=None):
        super().__init__(parent)
        self.selected_files = selected_files
        self.output_folder = output_folder
        self.model = model
        self.running = True
    def stop(self):
        self.running = False
    def run(self):
        # Perform the rotation correction on the selected files and save them to the output folder
        #self.output_label.setText('Loading model...')
        #model_location = load_model('./efficientnetv2_sv_open_images.hdf5', custom_objects={'angle_error': angle_error})
        #model_location = load_model('./rotnet_open_images_resnet50_TCML_2.hdf5', custom_objects={'angle_error': angle_error})
        output_path = self.output_folder
        file_path = self.selected_files
        #print(file_path)
        batch_size = 64
        #self.output_label.setText('Processsing input image(s)...')
        if self.model == None:
            self.model = load_model('./model/efficientnetv2_sv_open_images.hdf5', custom_objects={'angle_error': angle_error})
        process_images(self.model, self.selected_files, output_path,
                        batch_size, True)

        #self.output_label.setText("Rotation correction complete!")

        QThread.msleep(500)  # Simulate some processing time
        #QApplication.instance().exit()
        if self.running:
            self.rotation_completed.emit()


class ImageRotationCorrectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up the main window
        self.setWindowTitle("Image Rotation Correction")
        self.setStyleSheet("font-size: 18px;")
        self.setGeometry(400, 300, 600, 300)
        self.center_window()
        # Center the window on the screen
        available_geometry = QApplication.primaryScreen().availableGeometry()
        window_width = min(self.width(), available_geometry.width())
        window_height = min(self.height(), available_geometry.height())
        self.setGeometry(
            available_geometry.center().x() - window_width // 2,
            available_geometry.center().y() - window_height +40 // 2,
            window_width,
            window_height
        )

        # Set up the central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(20)
        self.central_widget.setStyleSheet("margin: 20px;")

        # Set up the file selection label
        self.file_label = QLabel("No files selected")
        self.layout.addWidget(self.file_label)
        self.file_label.setStyleSheet("font-weight: bold;")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


        # Set up the output folder selection label
        self.output_folder_label = QLabel("No output folder selected")
        self.layout.addWidget(self.output_folder_label)
        self.output_folder_label.setStyleSheet("font-weight: bold;")
        self.output_folder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Set up the output label
        self.output_label = QLabel()
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.output_label)

        

        self.file_button = QPushButton("Select Files")
        self.file_button.clicked.connect(self.select_files)
        self.layout.addWidget(self.file_button)
        self.file_button.setStyleSheet("""
        *{
        border: 2px solid '#7cc0d8';
        border-radius: 10px;
        background-color: '#7cc0d8';
        font-size: 20px;
        color: 'white';
        }
        *:hover{
        border: 2px solid '#5d8fa1';
        background-color: '#5d8fa1';
        font-size: 20px;
        color: 'white';
        }
        """)
        self.file_button.clicked.connect(self.button_click)

        #
        # Set up the folder selection button
        self.folder_button = QPushButton("Select Folder")
        self.folder_button.clicked.connect(self.select_folder)
        self.layout.addWidget(self.folder_button)
        self.folder_button.setStyleSheet("""
        *{
        border: 2px solid '#7cc0d8';
        border-radius: 10px;
        background-color: '#7cc0d8';
        font-size: 20px;
        color: 'white';
        }
        *:hover{
        border: 2px solid '#5d8fa1';
        background-color: '#5d8fa1';
        font-size: 20px;
        color: 'white';
        }
        """)
        self.folder_button.clicked.connect(self.button_click)
        # Set up the file selection button
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.file_button)
        buttons_layout.addWidget(self.folder_button)
        self.layout.addLayout(buttons_layout)
        # Set up the output folder selection button
        self.output_folder_button = QPushButton("Select Output Folder")
        self.output_folder_button.clicked.connect(self.select_output_folder)
        self.layout.addWidget(self.output_folder_button)
        self.output_folder_button.setStyleSheet("""
        *{
        border: 2px solid '#7cc0d8';
        border-radius: 10px;
        background-color: '#7cc0d8';
        font-size: 20px;
        color: 'white';
        }
        *:hover{
        border: 2px solid '#5d8fa1';
        background-color: '#5d8fa1';
        font-size: 20px;
        color: 'white';
        }
        """)
        self.output_folder_button.clicked.connect(self.button_click)
        # Connect the load_model_thread method to the output_folder_button click event
        self.folder_button.clicked.connect(self.load_model_thread)
        self.file_button.clicked.connect(self.load_model_thread)
        
        # Set up the correct rotation button
        self.correct_rotation_button = QPushButton("Correct Rotation")
        self.correct_rotation_button.clicked.connect(self.start_rotation_correction)
        self.layout.addWidget(self.correct_rotation_button)
        self.correct_rotation_button.setStyleSheet("""
        *{
        border: 2px solid '#76ce63';
        border-radius: 10px;
        background-color: '#76ce63';
        font-size: 20px;
        color: 'white';
        }
        *:hover{
        border: 2px solid '#4a873d';
        background-color: '#4a873d';
        font-size: 20px;
        color: 'white';
        }
        """)
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
            self.file_label.setText("Successfully selected {} file/s".format(len(file_names)))
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
            self.file_label.setText("Successfully selected 1 folder")
    def select_output_folder(self):
        options = QFileDialog()
        options = options.options()
        options |= QFileDialog.Option.DontUseNativeDialog
        file_dialog = QFileDialog()

        # Open the file dialog and allow the user to select the output folder
        output_folder = file_dialog.getExistingDirectory(
            self, "Select Output Folder", options=options
        )

        # Check if an output folder was selected
        if output_folder:
            self.output_folder = output_folder
            self.output_folder_label.setText("Selected output folder: {}".format(output_folder[:10]))
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
        progress_dialog.setWindowTitle("Rotation Correction")
        progress_dialog.setLabelText("Processing input image(s)...")
        progress_dialog.setCancelButtonText("Cancel")
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
        self.output_label.setText("Rotation correction completed!")

        # Reset the selected files and output folder
        self.selected_files = []
        self.output_folder = ""
        self.file_label.setText("No files selected")
        #self.output_folder_label.setText("No output folder selected")

    def closeEvent(self, event):
        if hasattr(self, "rotation_thread") and self.rotation_thread.isRunning():
            self.rotation_thread.stop()
            self.rotation_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    rotation_correction_app = ImageRotationCorrectionApp()
    rotation_correction_app.show()
    sys.exit(app.exec())
