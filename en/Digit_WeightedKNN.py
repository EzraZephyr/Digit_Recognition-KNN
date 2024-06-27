import warnings   # Ignore warning messages
import joblib   # Load and save trained models
import numpy as np  # Operate arrays
from PIL import Image  # Handle image files

class DigitRecognizer:
    # Create a handwritten digit recognizer class

    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        # Use the joblib method to load the trained model
        self.X_model = self.model._fit_X
        self.y_model = self.model._y
        # Extract the training data and labels from the model

    def compute_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
        # Calculate the Euclidean distance between two points using the formula

    def compute_weight(self, distance):
        a, b = 1, 1
        # Define parameters to adjust the weight, a can be understood as a smoothener to avoid division by zero
        return b / (distance + a)
        # Calculate and return the weight value. This formula shows that the smaller the distance, the greater the weight, and the greater the distance, the smaller the weight

    def predict_digit(self, filename):
        img = Image.open(filename).convert('L')
        img = img.resize((28, 28))
        img = np.array(img).reshape(1, -1)
        # Process the incoming image. This has been explained before, so I won't repeat it

        distances = []
        # Create an empty list to store the distance and label of each training sample with the input image
        for i, X_train in enumerate(self.X_model):
            # Iterate over each sample in the training set

            distance = self.compute_distance(img, X_train.reshape(1, -1))
            # Calculate the distance between the image and the training sample using the compute_distance function. Don't forget to reshape the training sample to match the shape of img

            weight = self.compute_weight(distance)
            # Calculate the weight

            distances.append((weight, self.y_model[i]))
            # Add the weight and corresponding label as a tuple to the list

        distances.sort(key=lambda x: x[0], reverse=True)
        # Use a lambda expression to sort the list in descending order by weight

        k_neighbors = distances[:3]
        # Select the top three neighbors with the highest weights after sorting

        weighted_votes = {}
        # Create an empty dictionary to record the weighted vote results for each label

        for weight, label in k_neighbors:
            # Iterate over the weights and labels of the three neighbors

            if label in weighted_votes:
                weighted_votes[label] += weight
                # If the label is already in the dictionary, simply accumulate the weight

            else:
                weighted_votes[label] = weight
                # Otherwise, if not, create a new entry for this label with the current weight

        predictions = max(weighted_votes, key=weighted_votes.get)
        # Choose the label with the highest weight in the weighted vote results as the final prediction result

        return predictions
        # Return this result

def digit_test():
    warnings.filterwarnings("ignore")
    # Because the feature names during training and the current feature names are different, a warning will be generated but it does not affect the operation and results
    # So just ignore this warning. If you want to solve it, add the following line of code before testing the model
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])]

    recognizer = DigitRecognizer('../mnist_784.pth')
    filename = '../handwritten_digits/digit_1.png'
    prediction = recognizer.predict_digit(filename)
    print(f'The test result is: {prediction}')
    # About the prediction image, it has been explained in previous files, so I won't repeat it

if __name__ == '__main__':  # Main function
    digit_test()
