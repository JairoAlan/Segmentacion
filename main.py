import argparse
import cv2
import numpy as np

"""Para poder ejecutarlo siga las instrucciones que se encuentran en el archivo instrucciones.txt"""


# Los argumentos de entrada definen la ruta a la imagen que será segmentada, así como el número
# de clusters o grupos a hallar mediante la aplicación de k-means.
argumen_parser = argparse.ArgumentParser()
argumen_parser.add_argument('-i', '--image', required=True, type=str, help='Ruta a la imagen')
argumen_parser.add_argument('-k', '--num-clusters', default=3, type=int, help='Número de clusters')
arguments = vars(argumen_parser.parse_args())

# Cargamos la imagen de entrada.
image = cv2.imread(arguments['image'])

# Verificamos que la imagen se haya cargado correctamente.
if image is None:
    print("Error al cargar la imagen. Verifique la ruta y el nombre del archivo.")
    exit()

# Creamos una copia de la imagen para manipularla.
image_copy = np.copy(image)

# Convertimos la imagen en un arreglo de ternas, las cuales representan el valor de cada pixel.
# Se aplana la imagen, volviéndola un vector de puntos en un espacio 3D.
pixel_values = image_copy.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Definimos el criterio de terminación del algoritmo.
stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Número de veces que se correrá K-Means con diferentes inicializaciones.
number_of_attempts = 10

# Estrategia para inicializar los centroides.
centroid_initialization_strategy = cv2.KMEANS_RANDOM_CENTERS

# Ejecutamos K-Means.
_, labels, centers = cv2.kmeans(pixel_values, arguments['num_clusters'], None, stop_criteria,
                                number_of_attempts, centroid_initialization_strategy)

# Convertimos los centros a valores enteros.
centers = np.uint8(centers)

# Aplicamos las etiquetas a los centroides para segmentar los píxeles en su grupo correspondiente.
segmented_data = centers[labels.flatten()]

# Se reestructura el arreglo de datos segmentados con las dimensiones de la imagen original.
segmented_image = segmented_data.reshape(image_copy.shape)

# Convertimos los centros a espacio de color HSV para identificar el cluster verde.
centers_hsv = cv2.cvtColor(np.uint8([centers]), cv2.COLOR_BGR2HSV)[0]

# Definimos el rango de color verde en HSV.
lower_green = np.array([40, 40, 40])
upper_green = np.array([140, 255, 255])

# Identificamos el cluster que representa el color verde.
green_cluster_index = None
for i, center in enumerate(centers_hsv):
    if np.all(center >= lower_green) and np.all(center <= upper_green):
        green_cluster_index = i
        break

if green_cluster_index is not None:
    # Creamos una máscara basada en el cluster verde.
    mask = (labels.flatten() == green_cluster_index).astype(np.uint8)
    mask = mask.reshape(image.shape[:2])  # Aseguramos que la máscara tenga las dimensiones correctas
    mask = mask * 255  # Escalamos la máscara a valores 0 y 255

    # Guardamos la máscara de segmentación.
    cv2.imwrite('mascara_verde.png', mask)

    # Aplicamos la máscara a la imagen original para obtener solo los píxeles verdes.
    green_segmented_image = cv2.bitwise_and(image, image, mask=mask)

    # Mostramos la imagen original, la máscara y la imagen segmentada.
    cv2.imshow('Imagen Original', image)
    cv2.imshow('Máscara Verde', mask)
    cv2.imshow('Imagen Segmentada Verde', green_segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se encontró ningún cluster que represente el color verde.")
