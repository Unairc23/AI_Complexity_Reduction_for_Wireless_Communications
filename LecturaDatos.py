import json

import numpy as np
import h5py
from matplotlib import pyplot as plt

with open("config.json", "r", encoding="utf-8") as f:
    conf = json.load(f)

def cargarDatos(dset='h_AAplant_int_5G'):
    f = h5py.File("NIST_Samples/NIST_Samples.mat",'r')
    data = f[dset]
    data = np.array(data)
    print (f"Shape original: {data.shape}")
    return data

def preprocesarDatos(data, window, stride, snr_db=5):
    data_clean = data[:, :128]
    print(f"Eliminando valores τ para evitar NaN: {data_clean.shape}")

    real = data_clean['real']
    imag = data_clean['imag']

    real = np.nan_to_num(real, nan=0.0)
    imag = np.nan_to_num(imag, nan=0.0)

    señal = real + 1j * imag

    imagenes = []
    imagenes_ruido = []

    for t in range(0, señal.shape[0] - window, stride):

        bloque = señal[t:t + window, :]

        # ---- LIMPIO ----
        real_clean = np.real(bloque)
        imag_clean = np.imag(bloque)

        # ---- CON RUIDO ----
        bloque_ruido = añadir_awgn_complejo(bloque, snr_db)
        real_ruido = np.real(bloque_ruido)
        imag_ruido = np.imag(bloque_ruido)

        # Normalización por canal
        real_clean = normalizar(real_clean)
        imag_clean = normalizar(imag_clean)

        real_ruido = normalizar(real_ruido)
        imag_ruido = normalizar(imag_ruido)

        img = np.stack([real_clean, imag_clean], axis=-1)
        img_ruido = np.stack([real_ruido, imag_ruido], axis=-1)

        imagenes.append(img)
        imagenes_ruido.append(img_ruido)

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

def añadir_awgn_complejo(señal, snr_db):
    p_señal = np.mean(np.abs(señal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    p_ruido = p_señal / snr_linear
    sigma = np.sqrt(p_ruido / 2)
    ruido = sigma * (np.random.randn(*señal.shape) +
                     1j * np.random.randn(*señal.shape))
    return señal + ruido

def normalizar(x):
    min_v = x.min()
    max_v = x.max()
    return (x - min_v) / (max_v - min_v + 1e-12)

if __name__ == '__main__':
    dset = 'h_AAplant_int_5G'
    dsets = ['h_AAplant_int_5G', 'h_AAplant_int_2G', 'h_AAplant_5G', 'h_AAplant_2G', 'h_Boil_2G', 'h_Boil_5G', 'h_GBurg_2G', 'h_GBurg_5G']
    snr_db = 5
    snrs = [5, 10, 20]

    if (conf["Data"]["Mixed"]):
        print("============== Juntando datasets ==============")
        data_total =[]
        for d in dsets:
            data_total.append(cargarDatos(d))
        data = np.concatenate(data_total, axis=0)
        dset = "full" # Esto solo para poner el nombre, se podria hacer mejor
    else:
        print("============== Cargando dataset individual ==============")
        data = cargarDatos(dset)

    imagenes, imagenes_ruido = preprocesarDatos(data, 128, 16, snr_db)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    im = axes[0, 0].imshow(imagenes[0][:, :, 0], cmap='viridis')
    axes[0, 0].set_title("Real limpia")
    plt.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(imagenes[0][:, :, 1], cmap='viridis')
    axes[0, 1].set_title("Imag limpia")
    plt.colorbar(im, ax=axes[0, 1])

    im = axes[1, 0].imshow(imagenes_ruido[0][:, :, 0], cmap='viridis')
    axes[1, 0].set_title("Real con ruido")
    plt.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].imshow(imagenes_ruido[0][:, :, 1], cmap='viridis')
    axes[1, 1].set_title("Imag con ruido")
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

    imagenes_ruido_variable = []
    imagenes_bien_variable = []
    for snr in snrs:
        imagenes_var, imagenes_ruido_var = preprocesarDatos(data, 128, 128, snr)
        imagenes_ruido_variable = imagenes_ruido_variable + imagenes_ruido_var
        imagenes_bien_variable = imagenes_bien_variable + imagenes_var
        print(f"Dataset combinado: {len(imagenes_ruido_variable)}")

    np.save(f'data/NIST_{dset}_imagenes.npy', imagenes)
    print(f"Imagenes guardadas en data/NIST_{dset}_imagenes.npy")

    np.save(f'data/NIST_{dset}_imagenes_snr_{snr_db}.npy', imagenes_ruido)
    print(f"Imagenes con SNR {snr_db} guardadas en data/NIST_{dset}_imagenes_snr_{snr_db}.npy")

    np.save(f'data/NIST_{dset}_imagenes_snr_variable.npy', imagenes_ruido_variable)
    print(f"Imagenes con SNR {snrs} guardadas en data/NIST_{dset}_imagenes_snr_variable.npy")

    np.save(f'data/NIST_{dset}_imagenes_limpias_variable.npy', imagenes_bien_variable)
    print(f"Imagenes limpias con repetición guardadas en data/NIST_{dset}_imagenes_limpias_variable.npy")
