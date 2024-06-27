# This GitHub repository is suitable for beginners in artificial intelligence algorithms

It discusses how to use the K-Nearest Neighbors (KNN) algorithm to implement simple handwritten digit recognition

Since we are using the MNIST dataset for handwritten digit recognition with the KNN algorithm, and this dataset contains grayscale images, it is highly sensitive to color differences. If the quality of the handwritten digit images is poor or there is noise, it is likely to result in recognition errors. However, for beginners, this type of dataset is sufficient to understand and learn the application of the KNN algorithm.

## Below is a brief introduction to the files, all of which have detailed comments

- **Learning**: The file for training the model
- **handwritten_digits**: The file for predicting the digits in images. It contains 30 grayscale images randomly selected from the MNIST dataset, which can be used for predictions
- **Digit_Test**: The file for testing the model
- **Digit_WeightedKNN**: The file for handwritten digit recognition using weighted KNN. (If you are not interested in making a small software application, you can stop here)
- **Digit_Recognition**: The final output file that uses a GUI to display a window, allowing for file uploads and result displays

