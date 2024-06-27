# Este repositorio de GitHub es adecuado para principiantes en algoritmos de inteligencia artificial

Se discute cómo usar el algoritmo K-Vecinos Más Cercanos (KNN) para implementar un reconocimiento simple de dígitos manuscritos

Dado que estamos utilizando el conjunto de datos MNIST y el algoritmo KNN para el reconocimiento de dígitos manuscritos, este conjunto de datos contiene imágenes en escala de grises, por lo que es muy sensible a las diferencias de color. Si la calidad de las imágenes de dígitos manuscritos es baja o hay ruido, es posible que se produzcan errores de reconocimiento. Sin embargo, para los principiantes, este tipo de conjunto de datos es suficiente para entender y aprender la aplicación del algoritmo KNN.

## A continuación, se presenta una breve introducción a los archivos, todos los cuales tienen comentarios detallados

- **Learning**: Archivo para entrenar el modelo
- **handwritten_digits**: Archivo para predecir los dígitos de las imágenes. Este archivo contiene 30 imágenes en escala de grises extraídas aleatoriamente del conjunto de datos MNIST, que se pueden usar para predicciones
- **Digit_Test**: Archivo para probar el modelo
- **Digit_WeightedKNN**: Archivo para el reconocimiento de dígitos manuscritos utilizando KNN ponderado. (Si no estás interesado en hacer un pequeño software, puedes detenerte aquí)
- **Digit_Recognition**: Archivo final que utiliza una GUI para mostrar una ventana, permitiendo la carga de archivos y la visualización de resultados
