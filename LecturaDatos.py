import numpy as np
import h5py
from matplotlib import pyplot as plt

def cargarDatos(dset='h_AAplant_int_5G'):
    f = h5py.File("NIST_Samples/NIST_Samples.mat",'r')
    data = f[dset]
    data = np.array(data) # For converting to a NumPy array
    print (f"Shape original: {data.shape}")
    return data

def preprocesarDatos(data, window, stride, snr_db=20):
    data_clean = data[:, :128]
    print(f"Eliminando valores τ para evitar NaN: {data_clean.shape}")

    data_complex = data_clean['real'] + 1j * data_clean['imag']
    data_complex = np.nan_to_num(data_complex, nan=0.0)

    imagenes = []
    imagenes_ruido = []
    for t in range(0, data_complex.shape[0]-window, stride):
        img = data_complex[t:t + window, :]
        img = np.abs(img)
        eps = 1e-12
        img = 10 * np.log10(img + eps) # Epsilon para evitar hacer log(0)
        img = np.clip(img, -60, 0) # Es raro que haya informacion por debajo de -60db, sueñe ser ruido
        img = (img + 60) / 60 # Normalizar los valores para que el modelo pueda aprender mejor
        img_ruido = añadir_awgn(img, snr_db)
        imagenes.append(img[..., np.newaxis])
        imagenes_ruido.append(img_ruido[..., np.newaxis])

    print(f"Numero de imagenes: {len(imagenes)}")
    return imagenes, imagenes_ruido

def añadir_awgn(señal, snr_db):
    p_señal = np.mean(señal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    p_ruido = p_señal / snr_linear
    ruido = np.random.normal(0, np.sqrt(p_ruido), señal.shape)
    señal_ruido = señal + ruido
    return np.clip(señal_ruido, 0.0, 1.0) # Esto hace que los valores creados por el ruido estén siempre
    # en un rango [0,1]. Esto no es necesario para el ruido, pero ahora mismo los valores de la imagen están normalizados
    # en un rango [0,1], si entrenas con esto pero luego el ruido lo sobrepasa el modelol lo mismo se rompe

if __name__ == '__main__':
    dset = 'h_AAplant_int_5G'
    snr_db = 10
    snrs = [5, 10, 20]

    data = cargarDatos(dset)
    imagenes, imagenes_ruido = preprocesarDatos(data, 128, 16, snr_db)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    axes = axes.flatten()

    im = axes[0].imshow(imagenes[0], cmap='viridis')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    im = axes[1].imshow(imagenes_ruido[0], cmap='viridis')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.show()

    imagenes_ruido_variable = []
    imagenes_bien_variable = []
    for snr in snrs:
        imagenes, imagenes_ruido = preprocesarDatos(data, 128, 16, snr_db)
        imagenes_ruido_variable = imagenes_ruido_variable + imagenes_ruido
        imagenes_bien_variable = imagenes_bien_variable + imagenes
        print(f"Dataset combinado: {len(imagenes_ruido_variable)}")

    np.save(f'data/NIST_{dset}_imagenes.npy', imagenes)
    print(f"Imagenes guardadas en data/NIST_{dset}_imagenes.npy")

    np.save(f'data/NIST_{dset}_imagenes_snr_{snr_db}.npy', imagenes_ruido)
    print(f"Imagenes con SNR {snr_db} guardadas en data/NIST_{dset}_imagenes_snr_{snr_db}.npy")

    np.save(f'data/NIST_{dset}_imagenes_snr_variable.npy', imagenes_ruido_variable)
    print(f"Imagenes con SNR {snrs} guardadas en data/NIST_{dset}_imagenes_snr_variable.npy")

    np.save(f'data/NIST_{dset}_imagenes_limpias_variable.npy', imagenes_bien_variable)
    print(f"Imagenes limpias con repetición guardadas en data/NIST_{dset}_imagenes_limpias_variable.npy")
