import pandas as pd    # Manejar conjuntos de datos
import joblib   # Cargar y guardar modelos entrenados
from sklearn.datasets import fetch_openml   # Imágenes de dígitos manuscritos
from sklearn.model_selection import train_test_split    # Dividir datos de entrenamiento y prueba y ajustar la proporción
from sklearn.neighbors import KNeighborsClassifier  # Clasificador K-Vecinos Más Cercanos, es decir, algoritmo KNN, predice en función de los K vecinos más cercanos
from sklearn.metrics import accuracy_score  # Calcular la precisión del modelo

mnist = fetch_openml('mnist_784', version=1)
# Obtener el conjunto de datos llamado mnist_784, versión 1. Este conjunto de datos contiene 70,000 imágenes de dígitos manuscritos del 0 al 9
# Un hecho interesante, las 60,000 imágenes de entrenamiento en este conjunto de datos fueron escritas por empleados del Censo de EE.UU., mientras que las 10,000 imágenes de prueba provienen de estudiantes de secundaria de EE.UU.

X = pd.DataFrame(mnist['data'])
# Extraer las características de este conjunto de datos. Estas características son vectores bidimensionales y su índice en los datos es 'data', por lo que deben manejarse con DataFrame
y = pd.Series(mnist.target).astype('int')
# Para las etiquetas en este conjunto de datos, es decir, el valor objetivo correspondiente a un vector bidimensional de 28*28, es un vector unidimensional, por lo que se maneja con Series
# Como los valores objetivo están diseñados como tipo de cadena, deben convertirse a tipo entero con astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Usar la función train_test_split para dividir las características y las etiquetas (es decir, X y y) en proporciones de 80% y 20%
# Así, X_train e y_train se usan para entrenamiento, y X_test e y_test se usan para prueba

estimator = KNeighborsClassifier(n_neighbors=3)
# Instanciar un clasificador K-Vecinos Más Cercanos y establecer n_neighbors a 3, lo que significa que en futuras pruebas de datos o uso del modelo
# Este algoritmo buscará automáticamente los tres vecinos más cercanos en el espacio de características (generalmente usando distancia euclidiana)
# Y determinará la categoría del nuevo punto de datos en función de las categorías de estos tres vecinos

estimator.fit(X_train, y_train)
# Comenzar a entrenar el modelo utilizando el método fit de este clasificador, pasando los datos de entrenamiento y sus etiquetas correspondientes
# Permitir que el modelo aprenda y recuerde la relación entre las características de los datos y los valores objetivo, para que pueda usarse para hacer predicciones en el futuro

y_pred = estimator.predict(X_test)
# Después de completar el entrenamiento del modelo, usar los datos de prueba divididos anteriormente para hacer predicciones y calcular la precisión del modelo.
# Llamar al método predict del clasificador pasando los datos de prueba X_test, devolverá los resultados de la predicción para cada muestra de prueba en función de las reglas aprendidas anteriormente.

print(accuracy_score(y_test, y_pred))
# Calcular la precisión del modelo comparando los resultados de prueba

joblib.dump(estimator, '../mnist_784.pth')
# Finalmente, llamar al método dump de la biblioteca joblib, pasando el modelo entrenado y estableciendo el nombre del archivo como 'mnist_784.pth'
# De esta manera, el modelo se guarda en el disco y se puede cargar y usar directamente sin necesidad de volver a entrenar el modelo.
