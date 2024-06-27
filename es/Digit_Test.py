import warnings   # Ignorar mensajes de advertencia
import joblib   # Cargar y guardar modelos entrenados
import numpy as np  # Operar con matrices
from PIL import Image  # Manejar archivos de imagen

def digit_test():
    warnings.filterwarnings("ignore")
    # Porque los nombres de las características durante el entrenamiento y los actuales son diferentes, se generará una advertencia pero no afecta la ejecución ni los resultados
    # Así que simplemente ignore esta advertencia. Si desea resolverlo, agregue la siguiente línea de código antes de probar el modelo
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])]

    model = joblib.load('../mnist_784.pth')
    # Usar el método joblib para cargar el modelo entrenado

    filename = '../handwritten_digits/digit_1.png'
    # Guardar la ruta del archivo de imagen en filename para que sea conveniente llamarlo directamente a continuación

    img = Image.open(filename).convert('L')
    # Usar el método Image.open para abrir la imagen en esa ruta y convertirla a escala de grises con el método convert

    img = img.resize((28, 28))
    # Comprimir el tamaño de la imagen a un formato de 28*28 para cumplir con los requisitos de entrada del modelo

    img = np.array(img).reshape(1,-1)
    # Usar el método np.array para convertir img en tipo matriz. El primer '1' en el método reshape se usa para estirar esta matriz bidimensional en una matriz unidimensional
    # El segundo '-1' permite a Numpy calcular automáticamente el tamaño del resto de las dimensiones, aplanando la matriz en una matriz que contiene 784 elementos

    predict = model.predict(img)
    # Ingresar img procesada en el modelo para predecir, luego el modelo devolverá el resultado final basado en los tres 'vecinos' más cercanos establecidos

    print(f'El resultado de la prueba es: {predict[0]}')
    # Como se devuelve en forma de matriz, se imprime el primer número en ella, que es el resultado

if __name__ == '__main__': # Función principal
    digit_test()
