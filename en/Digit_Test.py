import warnings   # Ignore warning messages
import joblib   # Load and save trained models
import numpy as np  # Operate arrays
from PIL import Image  # Handle image files

def digit_test():
    warnings.filterwarnings("ignore")
    # Because the feature names during training and the current feature names are different, a warning will be generated but it does not affect the operation and results
    # So just ignore this warning. If you want to solve it, add the following line of code before testing the model
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])]

    model = joblib.load('../mnist_784.pth')
    # Use the joblib method to load the trained model

    filename = '../handwritten_digits/digit_1.png'
    # Store the path of the image file in filename for convenient calling below

    img = Image.open(filename).convert('L')
    # Use the Image.open method to open the image at that path and convert it to grayscale using the convert method

    img = img.resize((28, 28))
    # Compress the image size to a 28*28 format to meet the model's input requirements

    img = np.array(img).reshape(1,-1)
    # Use the np.array method to convert img to array type. The first '1' in the reshape method is used to stretch this two-dimensional array into a one-dimensional array
    # The second '-1' allows Numpy to automatically calculate the size of the remaining dimensions, flattening the array into an array containing 784 elements

    predict = model.predict(img)
    # Input the processed img into the model for prediction, and then the model will return the final result based on the three nearest 'neighbors' set

    print(predict[0])
    # Since it is returned in the form of an array, print the first number in it, which is the result

if __name__ == '__main__': # Main function
    digit_test()
