import warnings  # Ignore warning messages
import sys  # System related functions
import joblib  # Load and save trained models
import numpy as np  # Operate arrays
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel  # GUI building module
from PyQt5.QtGui import QPixmap  # Handle image display
from PIL import Image  # Handle image files

class MainWindow(QWidget):
    # Create a class for the GUI main window and inherit methods from QWidget, allowing full customization of the window's appearance and operations

    def __init__(self):
        super().__init__()
        # When executing this initialization function, call the parent class (QWidget) method

        self.init_ui()
        self.model = joblib.load('../mnist_784.pth')
        # Use joblib method to load the trained model

    def init_ui(self):
        self.setWindowTitle('Handwritten Digit Recognition')
        # Set the title of this window, which is displayed at the top center of the border when the window is opened

        self.resize(1000, 600)
        # Adjust the size of the window

        layout = QVBoxLayout()
        # Create a vertical layout so that the sub-components added to the vertical layout will be arranged vertically from top to bottom
        # And adjust the size and position of the sub-components in real-time according to the size of the window, evenly distributing them to prevent overlap

        self.btn = QPushButton('Load Image', self)
        # Create a button and display the text "Load Image" horizontally centered on the button
        # A brief explanation of the self in the parentheses: this self specifies this class as the "parent widget" of the button
        # Add the button to this window, and when the window is closed, the button will be automatically destroyed to avoid memory leaks

        self.btn.setFixedSize(200, 200)
        # Adjust the size of the button

        self.btn.clicked.connect(self.load_Image)
        # Connect the button's click signal to the self.loadImage function, so that when the button is clicked, this function is triggered

        layout.addWidget(self.btn)
        # Add this button to the layout

        self.resultLabel = QLabel('The test result is:', self)
        # Create a label to display the final result

        layout.addWidget(self.resultLabel)
        # Add the result label to the layout

        self.imageLabel = QLabel(self)
        # Create a label to display the test image

        layout.addWidget(self.imageLabel)
        # Add this image label to the layout

        self.setLayout(layout)
        # Set the created layout as the layout manager of the current window so that the labels added to the layout can be automatically adjusted and displayed

    def load_Image(self):
        options = QFileDialog.Options()
        # This method creates the file selection box triggered when clicking on the image

        filename, _ = QFileDialog.getOpenFileName(self, "Please select an image", "", "All Files (*)", options=options)
        # Open the file selection box and select the file to pass to the dialog box, "" represents the default directory, "All Files (*)" allows displaying and selecting all types of files

        if filename:
            pixmap = QPixmap(filename)
            # Use QPixmap method to load the selected image. The main reason for using QPixmap is that it is compatible with QLabel and can be directly loaded into imageLabel

            self.imageLabel.setPixmap(pixmap)
            # Set the loaded image as imageLabel so that it can be displayed in the window

            self.imageLabel.adjustSize()
            # Adjust the size of ImageLabel to fit the image

            prediction = self.predict_Digit(filename)
            # Call the predictDigit function for prediction and return the value to prediction

            self.resultLabel.setText(f'The test result is: {prediction}')
            # Add the predicted result to the text displayed in result_Label

    def predict_Digit(self, filename):
        img = Image.open(filename).convert('L')
        # Use Image.open method to open the image in the path and use the convert method to convert it to grayscale

        img = img.resize((28, 28))
        # Compress the image size to a 28*28 format to meet the model's input requirements

        img = np.array(img).reshape(1, -1)
        # Use np.array method to convert img to array type. The first '1' in the reshape method is used to stretch this two-dimensional array into a one-dimensional array
        # The second '-1' allows Numpy to automatically calculate the size of the remaining dimensions, flattening the array into an array containing 784 elements

        prediction = self.model.predict(img)
        # Input the processed img into the model for prediction, and then the model will return the final result based on the three nearest 'neighbors' set
        return prediction[0]
        # Since it is returned in the form of an array, print the first number in it, which is the result, and return that result

if __name__ == '__main__':  # Main function
    warnings.filterwarnings("ignore")
    # Because the feature names during training and now are different, a warning will be generated, but it does not affect the operation and result
    # So just ignore this warning. If you want to solve it, add before testing the model
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])]

    app = QApplication(sys.argv)
    # Create a QApplication object, responsible for managing the control flow and other settings of this program

    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
    # app.exec_() enters the main loop of the program and starts processing the aforementioned events. When exiting, ensure that the program can exit cleanly
