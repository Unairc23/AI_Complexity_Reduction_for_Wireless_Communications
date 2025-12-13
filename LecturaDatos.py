import numpy as np
import h5py
from matplotlib import pyplot as plt

def cargarDatos():
    f = h5py.File('NIST_Samples/NIST_Samples.mat','r')
    data = f['h_AAplant_2G']
    data = np.array(data) # For converting to a NumPy array
    print (data)
    return data

def preprocesarDatos(data, size):
    data_clean = data[:, :size]

    print(data_clean.shape)

    data_complex = data_clean['real'] + 1j * data_clean['imag']
    data_complex = np.nan_to_num(data_complex, nan=0.0)

    mag = np.abs(data_complex)  # magnitud para imagen real
    mag = mag / np.max(mag)

    Y = mag.shape[0]
    nImagenes = Y // size
    mag = mag[:nImagenes * size, :]  # recortar filas extra

    images = mag.reshape(nImagenes, size, size)

    return images

if __name__ == '__main__':
    data = cargarDatos()
    images = preprocesarDatos(data, 128)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    axes = axes.flatten()
    for i in range(9):
        im = axes[i].imshow(images[i], cmap='viridis')
        axes[i].axis('off')
        # Agregar barra de color a cada imagen
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
