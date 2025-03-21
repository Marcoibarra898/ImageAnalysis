import numpy as np
import cv2
import matplotlib.pyplot as plt

def conv_helper(fragment, kernel):
    """
    Multiplica dos matrices del mismo tamaño y devuelve la suma de sus productos elemento a elemento.
    """
    return np.sum(fragment * kernel)

def convolution(image, kernel):
    """
    Aplica una convolución sin padding a una imagen con el kernel dado.
    Devuelve la imagen resultante de la operación.
    """
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    
    # Definir el tamaño de la imagen de salida
    output_row = image_row - kernel_row + 1
    output_col = image_col - kernel_col + 1
    output = np.zeros((output_row, output_col))
    
    # Aplicar la convolución
    for row in range(output_row):
        for col in range(output_col):
            fragment = image[row:row + kernel_row, col:col + kernel_col]
            output[row, col] = conv_helper(fragment, kernel)
    
    print(f"Output Image size (no padding): {output.shape}")
    
    # Asegurar valores entre 0-255
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output

def convolve_image_no_padding(image_path, kernel):
    """
    Carga una imagen, la convierte a escala de grises, aplica convolución sin padding y muestra los resultados.
    """
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    convolved_image = convolution(gray_image, kernel)
    
    # Mostrar imágenes: Original, Escala de grises, Convolución
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    axs[1].imshow(gray_image, cmap='gray')
    axs[1].set_title("Escala de grises")
    axs[1].axis('off')
    
    axs[2].imshow(convolved_image, cmap='gray')
    axs[2].set_title("Convolved Output")
    axs[2].axis('off')
    
    plt.show()
    
    return convolved_image

# Leer imagen desde archivo
image_path = '\\China.jpg'  # Ajusta la ruta si es necesario
kernel = np.ones((3, 3))

# Ejecutar convolución sin padding
resultado = convolve_image_no_padding(image_path, kernel)

# Guardar resultado
cv2.imwrite('resultado.jpg', resultado)
