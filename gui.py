import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QMessageBox, QLabel
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import subprocess # Keep for fallback or other potential uses, but primary will be direct import
import os # For getting current working directory
from qwen_inference import process_user_query, load_model_and_tokenizer # Import the refactored functions

# Worker thread for running the model inference to keep GUI responsive
class InferenceThread(QThread):
    signal_result = pyqtSignal(str)
    signal_error = pyqtSignal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        try:
            result = process_user_query(self.command)
            self.signal_result.emit(result)
        except Exception as e:
            self.signal_error.emit(f"Error during inference: {str(e)}")


class QwenGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Qwen Inference GUI')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.userInput = QLineEdit(self)
        self.userInput.setPlaceholderText("Enter your command here...")
        self.userInput.setMinimumHeight(50) # Make input box taller (increased from 40)
        layout.addWidget(self.userInput)

        # Horizontal layout for current directory label and the button
        info_button_layout = QHBoxLayout()
        
        self.currentDirLabel = QLabel(self)
        self.currentDirLabel.setText(f"CWD: {os.getcwd()}")
        self.currentDirLabel.setWordWrap(True) # Allow text to spill downwards
        info_button_layout.addWidget(self.currentDirLabel, 3) # Label stretch factor

        info_button_layout.addStretch(1) # Spacer stretch factor (3:1 ratio with label for ~50% increase for label space)

        self.runButton = QPushButton('Run Command', self)
        self.runButton.clicked.connect(self.onRunCommand)
        self.runButton.setMinimumHeight(50)
        self.runButton.setFixedWidth(180)
        info_button_layout.addWidget(self.runButton, 0, Qt.AlignmentFlag.AlignRight) # Align button to the right
        
        layout.addLayout(info_button_layout)

        self.outputDisplay = QTextEdit(self)
        self.outputDisplay.setReadOnly(True)
        layout.addWidget(self.outputDisplay)

        self.setLayout(layout)
        self.show()

    def onRunCommand(self):
        user_command = self.userInput.text()
        if not user_command:
            self.outputDisplay.setText("Please enter a command.")
            return

        self.outputDisplay.setText(f"Processing: {user_command}\n...")
        self.runButton.setEnabled(False) # Disable button during processing

        self.inference_thread = InferenceThread(user_command)
        self.inference_thread.signal_result.connect(self.handle_inference_result)
        self.inference_thread.signal_error.connect(self.handle_inference_error)
        self.inference_thread.start()

    def handle_inference_result(self, result):
        self.outputDisplay.setText(f"Output:\n{result}")
        self.currentDirLabel.setText(f"CWD: {os.getcwd()}") # Update CWD
        self.runButton.setEnabled(True) # Re-enable button

    def handle_inference_error(self, error_message):
        self.outputDisplay.setText(error_message)
        self.currentDirLabel.setText(f"CWD: {os.getcwd()}") # Update CWD even on error
        QMessageBox.critical(self, "Inference Error", error_message)
        self.runButton.setEnabled(True) # Re-enable button

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Load model once at startup
    try:
        print("Pre-loading model and tokenizer for the GUI...")
        load_model_and_tokenizer() # This will load globals in qwen_inference
        print("Model and tokenizer pre-loaded successfully.")
    except Exception as e:
        print(f"Failed to pre-load model/tokenizer: {e}")
        # Optionally, show a critical error to the user and exit if model loading is essential
        # QMessageBox.critical(None, "Startup Error", f"Failed to load AI model: {e}\nThe application might not function correctly.")
        # sys.exit(1) # Uncomment to make model loading critical

    ex = QwenGUI()
    sys.exit(app.exec())