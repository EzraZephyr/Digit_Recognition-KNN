import warnings   # Ignorar mensajes de advertencia
import joblib   # Cargar y guardar modelos entrenados
import numpy as np  # Operar con matrices
from PIL import Image  # Manejar archivos de imagen

class DigitRecognizer:
    # Crear una clase de reconocimiento de dígitos manuscritos

    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        # Usar el método joblib para cargar el modelo entrenado
        self.X_model = self.model._fit_X
        self.y_model = self.model._y
        # Extraer los datos de entrenamiento y las etiquetas del modelo

    def compute_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
        # Calcular la distancia euclidiana entre dos puntos usando la fórmula

    def compute_weight(self, distance):
        a, b = 1, 1
        # Definir parámetros para ajustar el peso, a puede entenderse como un suavizante para evitar dividir por cero
        return b / (distance + a)
        # Calcular y devolver el valor del peso. A través de esta fórmula se puede ver que cuanto menor es la distancia, mayor es el peso, y cuanto mayor es la distancia, menor es el peso

    def predict_digit(self, filename):
        img = Image.open(filename).convert('L')
        img = img.resize((28, 28))
        img = np.array(img).reshape(1, -1)
        # Procesar la imagen entrante. Esto ya se ha explicado antes, así que no lo repetiré

        distances = []
        # Crear una lista vacía para almacenar la distancia y la etiqueta de cada muestra de entrenamiento con la imagen de entrada
        for i, X_train in enumerate(self.X_model):
            # Recorrer cada muestra en el conjunto de entrenamiento

            distance = self.compute_distance(img, X_train.reshape(1, -1))
            # Calcular la distancia entre la imagen y la muestra de entrenamiento usando la función compute_distance. No olvidar convertir la muestra de entrenamiento para que coincida con la forma de img

            weight = self.compute_weight(distance)
            # Calcular el peso

            distances.append((weight, self.y_model[i]))
            # Añadir el peso y la etiqueta correspondiente como una tupla a la lista

        distances.sort(key=lambda x: x[0], reverse=True)
        # Ordenar la lista en orden descendente por peso usando una expresión lambda

        k_neighbors = distances[:3]
        # Seleccionar los tres vecinos con mayor peso después de ordenar

        weighted_votes = {}
        # Crear un diccionario vacío para registrar los resultados de la votación ponderada de cada etiqueta

        for weight, label in k_neighbors:
            # Recorrer los pesos y las etiquetas de los tres vecinos

            if label in weighted_votes:
                weighted_votes[label] += weight
                # Si la etiqueta ya está en el diccionario, simplemente sumar este peso

            else:
                weighted_votes[label] = weight
                # De lo contrario, si no está, crear una nueva entrada para esta etiqueta con el peso actual

        predictions = max(weighted_votes, key=weighted_votes.get)
        # Elegir la etiqueta con mayor peso en los resultados de la votación ponderada como el resultado de la predicción final

        return predictions
        # Devolver este resultado

def digit_test():
    warnings.filterwarnings("ignore")
    # Porque los nombres de las características durante el entrenamiento y los actuales son diferentes, se generará una advertencia pero no afecta la ejecución ni los resultados
    # Así que simplemente ignore esta advertencia. Si desea resolverlo, agregue la siguiente línea de código antes de probar el modelo
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])]

    recognizer = DigitRecognizer('../mnist_784.pth')
    filename = '../handwritten_digits/digit_1.png'
    prediction = recognizer.predict_digit(filename)
    print(f'El resultado de la prueba es: {prediction}')
    # Sobre la imagen de predicción ya se ha explicado en archivos anteriores, así que no lo repetiré

if __name__ == '__main__':  # Función principal
    digit_test()
