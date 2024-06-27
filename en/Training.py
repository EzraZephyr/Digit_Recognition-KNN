import pandas as pd    # Handle datasets
import joblib   # Load and save trained models
from sklearn.datasets import fetch_openml   # Handwritten digit images
from sklearn.model_selection import train_test_split    # Split training and test data and adjust the ratio
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors classifier, i.e., KNN algorithm, predicts based on the nearest K neighbors
from sklearn.metrics import accuracy_score  # Calculate the accuracy of the model

mnist = fetch_openml('mnist_784', version=1)
# Fetch the dataset named mnist_784, version 1. This dataset contains 70,000 images of handwritten digits from 0 to 9
# An interesting background, 60,000 training images in this dataset were written by US Census Bureau employees, while 10,000 test images are from US high school students

X = pd.DataFrame(mnist['data'])
# Extract features from this dataset. These features are two-dimensional vectors indexed by 'data' in the dataset, so they need to be handled with DataFrame
y = pd.Series(mnist.target).astype('int')
# For the labels in this dataset, which are the target values corresponding to a 28*28 two-dimensional vector, they are one-dimensional vectors, so they are handled with Series
# As the target values are designed as string type, they need to be converted to int type using astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Use the train_test_split function to split the features and labels (i.e., X and y) in a ratio of 80% and 20%
# So that X_train and y_train are used for training, and X_test and y_test are used for testing

estimator = KNeighborsClassifier(n_neighbors=3)
# Instantiate a K-Nearest Neighbors classifier and set n_neighbors to 3, which means that in future data tests or model usage
# This algorithm will automatically search for the three nearest neighbors in the feature space (generally using Euclidean distance)
# And determine the category of the new data point based on the categories of these three neighbors

estimator.fit(X_train, y_train)
# Start training the model using the fit method of this classifier, passing in the training data and their corresponding labels
# Allow the model to learn and remember the relationship between data features and target values so that it can be used for predictions later

y_pred = estimator.predict(X_test)
# After the model training is complete, use the test data split above to make predictions and calculate the accuracy of the model.
# Call the predict method of the classifier, passing in the test data X_test, it will return the prediction results for each test sample based on the learned rules.

print(accuracy_score(y_test, y_pred))
# Calculate the accuracy of the model by comparing the test results

joblib.dump(estimator, '../mnist_784.pth')
# Finally, call the dump method from the joblib library, passing in the trained model and setting the file name to 'mnist_784.pth'
# This way, the model is saved to disk and can be loaded and used directly in the future without the need to retrain the model
