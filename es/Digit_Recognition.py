import warnings  # Ignorar mensajes de advertencia
import sys  # Funciones relacionadas con el sistema
import joblib  # Cargar y guardar modelos entrenados
import numpy as np  # Operar con matrices
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel  # Módulo para construir GUI
from PyQt5.QtGui import QPixmap  # Manejar la visualización de imágenes
from PIL import Image  # Manejar archivos de imagen

class MainWindow(QWidget):
    # Crear una clase de ventana principal GUI y heredar métodos de QWidget, se puede personalizar completamente la apariencia y operación de la ventana

    def __init__(self):
        super().__init__()
        # Al ejecutar esta función de inicialización, llamar al método de la clase padre (QWidget)

        self.init_ui()
        self.model = joblib.load('../mnist_784.pth')
        # Usar el método joblib para cargar el modelo entrenado

    def init_ui(self):
        self.setWindowTitle('Reconocimiento de Dígitos Manuscritos')
        # Establecer el título de esta ventana, que se muestra en la parte superior central del borde cuando se abre la ventana

        self.resize(1000, 600)
        # Ajustar el tamaño de la ventana

        layout = QVBoxLayout()
        # Crear un diseño vertical para que los widgets hijos agregados a él se dispongan verticalmente de arriba a abajo
        # Y se ajusten en tamaño y posición en tiempo real según el tamaño de la ventana, distribuyéndose uniformemente para evitar superposiciones

        self.btn = QPushButton('Cargar Imagen', self)
        # Crear un botón y mostrar horizontalmente centrado el texto "Cargar Imagen" en el botón
        # La explicación del self en los paréntesis: especifica que esta clase es el "widget padre" del botón
        # Añadir el botón a esta ventana, y cuando la ventana se cierre, el botón se destruirá automáticamente para evitar fugas de memoria

        self.btn.setFixedSize(200, 200)
        # Ajustar el tamaño del botón

        self.btn.clicked.connect(self.load_Image)
        # Conectar la señal de clic de este botón a la función self.loadImage, para que cuando se haga clic en el botón se dispare esta función

        layout.addWidget(self.btn)
        # Añadir este botón al diseño

        self.resultLabel = QLabel('El resultado de la prueba es:', self)
        # Crear una etiqueta para mostrar el resultado final

        layout.addWidget(self.resultLabel)
        # Añadir la etiqueta de resultado al diseño

        self.imageLabel = QLabel(self)
        # Crear una etiqueta para mostrar la imagen de prueba

        layout.addWidget(self.imageLabel)
        # Añadir la etiqueta de imagen al diseño

        self.setLayout(layout)
        # Establecer el diseño creado como el administrador de diseño de la ventana actual para que las etiquetas agregadas al diseño se ajusten automáticamente y se muestren

    def load_Image(self):
        options = QFileDialog.Options()
        # Este método crea el cuadro de diálogo de selección de archivos que se dispara al hacer clic en la imagen

        filename, _ = QFileDialog.getOpenFileName(self, "Seleccione una imagen", "", "Todos los archivos (*)", options=options)
        # Abrir el cuadro de diálogo de selección de archivos y seleccionar un archivo para pasar al cuadro de diálogo, "" representa el directorio predeterminado, "Todos los archivos (*)" permite mostrar y seleccionar todos los tipos de archivos

        if filename:
            pixmap = QPixmap(filename)
            # Usar el método QPixmap para cargar la imagen seleccionada. Se usa QPixmap principalmente porque es compatible con QLabel y se puede cargar directamente en imageLabel

            self.imageLabel.setPixmap(pixmap)
            # Establecer la imagen cargada como imageLabel para que se muestre en la ventana

            self.imageLabel.adjustSize()
            # Ajustar el tamaño de ImageLabel para adaptarse a la imagen

            prediction = self.predict_Digit(filename)
            # Llamar a la función predictDigit para predecir y devolver el valor a prediction

            self.resultLabel.setText(f'El resultado de la prueba es: {prediction}')
            # Añadir el resultado predicho al contenido de texto que se muestra en result_Label

    def predict_Digit(self, filename):
        img = Image.open(filename).convert('L')
        # Usar el método Image.open para abrir la imagen en la ruta y convertirla a escala de grises con el método convert

        img = img.resize((28, 28))
        # Comprimir el tamaño de la imagen a un formato de 28*28 para cumplir con los requisitos de entrada del modelo

        img = np.array(img).reshape(1, -1)
        # Usar el método np.array para convertir img en tipo matriz. El primer '1' en el método reshape se usa para estirar esta matriz bidimensional en una matriz unidimensional
        # El segundo '-1' permite a Numpy calcular automáticamente el tamaño del resto de las dimensiones, aplanando la matriz en una matriz que contiene 784 elementos

        prediction = self.model.predict(img)
        # Ingresar img procesada en el modelo para predecir, luego el modelo devolverá el resultado final basado en los tres 'vecinos' más cercanos establecidos
        return prediction[0]
        # Como se devuelve en forma de matriz, se imprime el primer número en ella, que es el resultado, y se devuelve ese resultado

if __name__ == '__main__':  # Función principal
    warnings.filterwarnings("ignore")
    # Como los nombres de las características en el entrenamiento y ahora son diferentes, se generará una advertencia, pero no afecta la ejecución ni el resultado
    # Así que simplemente ignore esta advertencia. Si desea resolverlo, agregue antes de probar el modelo
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])]

    app = QApplication(sys.argv)
    # Crear un objeto QApplication, responsable de gestionar el flujo de control de este programa y otras configuraciones

    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
    # app.exec_() entra en el bucle principal del programa y comienza a procesar los eventos mencionados. Al salir, asegúrese de que el programa pueda salir limpiamente
