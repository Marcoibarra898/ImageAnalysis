import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_padded_matrix(image, pad_size):
    """
    Recibe una matriz (imagen) y el tamaño de padding.
    Devuelve una nueva matriz con la imagen en el centro y relleno de ceros en la orilla.
    """
    image_row, image_col = image.shape
    pad_height, pad_width = pad_size
    
    # Crear una nueva matriz rellena de ceros con el tamaño aumentado por el padding
    padded_matrix = np.zeros((image_row + 2 * pad_height, image_col + 2 * pad_width))
    
    # Insertar la imagen original en el centro
    padded_matrix[pad_height:pad_height + image_row, pad_width:pad_width + image_col] = image
    
    return padded_matrix

# Leer imagen desde archivo
image_path = '\\China.jpg'  # Ajusta la ruta si es necesario
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Definir tamaño del padding (ejemplo: 2 píxeles en cada lado)
pad_size = (2, 2)

# Crear la matriz con padding
padded_image = create_padded_matrix(image, pad_size)

# Mostrar la imagen original y la imagen con padding
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis('off')

axs[1].imshow(padded_image, cmap='gray')
axs[1].set_title("Padded Image")
axs[1].axis('off')

plt.show()