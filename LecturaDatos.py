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
    data_clean = data[:, :window]
    print(f"Eliminando valores τ para evitar NaN: {data_clean.shape}")

    real = data_clean['real']
    imag = data_clean['imag']

    # Pasar NaN -> 0
    real = np.nan_to_num(real, nan=0.0)
    imag = np.nan_to_num(imag, nan=0.0)

    señal = real + 1j * imag
    # señal_norm, _ = normalizar_complejo(señal)

    imagenes = []
    imagenes_ruido = []
    snrs = []

    for t in range(0, señal.shape[0] - window + 1, stride):

        bloque = señal[t:t + window, :]
        bloque, _ = normalizar_complejo(bloque)

        # ---- LIMPIO ----
        real_clean = np.real(bloque)
        imag_clean = np.imag(bloque)

        # ---- CON RUIDO ----
        bloque_ruido = añadir_awgn_complejo(bloque, snr_db)
        real_ruido = np.real(bloque_ruido)
        imag_ruido = np.imag(bloque_ruido)

        img = np.stack([real_clean, imag_clean], axis=-1)
        img_ruido = np.stack([real_ruido, imag_ruido], axis=-1)
        snr = calcular_ruido(img, img_ruido)

        imagenes.append(img)
        imagenes_ruido.append(img_ruido)
        snrs.append(snr)

    return imagenes, imagenes_ruido, snrs

def añadir_awgn_complejo(señal, snr_db):
    p_señal = np.mean(np.abs(señal) ** 2) # Potencia de la señal
    snr_linear = 10 ** (snr_db / 10) # Psar SNR de dB a Linear
    p_ruido = p_señal / snr_linear
    sigma = np.sqrt(p_ruido / 2)
    ruido = sigma * (np.random.randn(*señal.shape) +
                     1j * np.random.randn(*señal.shape))
    return señal + ruido

def calcular_ruido(señal_norm, señal_ruido_norm):
    ruido = señal_ruido_norm - señal_norm
    p_señal = np.mean(np.abs(señal_norm) ** 2)
    p_ruido = np.mean(np.abs(ruido) ** 2)
    if p_ruido == 0:
        return np.inf
    return 10 * np.log10(p_señal / p_ruido)

def normalizar_complejo(señal, stats=None, clip=False):
    real = np.real(señal)
    imag = np.imag(señal)

    if stats is None:
        min_r, max_r = real.min(), real.max()
        min_i, max_i = imag.min(), imag.max()
    else:
        min_r, max_r, min_i, max_i = stats

    eps = 1e-12
    real_norm = 2.0 * (real - min_r) / (max_r - min_r + eps) - 1.0
    imag_norm = 2.0 * (imag - min_i) / (max_i - min_i + eps) - 1.0

    if clip:
        real_norm = np.clip(real_norm, -1.0, 1.0)
        imag_norm = np.clip(imag_norm, -1.0, 1.0)

    return real_norm + 1j * imag_norm, (min_r, max_r, min_i, max_i)

def normalizar_por_potencia(bloque):
    potencia = np.sqrt(np.mean(np.abs(bloque) ** 2))
    return bloque / (potencia + 1e-12)

def preprocesarDatosSynt(data, window, stride):
    real = np.real(data)
    imag = np.imag(data)

    # Pasar NaN -> 0
    real = np.nan_to_num(real, nan=0.0)
    imag = np.nan_to_num(imag, nan=0.0)

    imagenes = []

    señal = real + 1j * imag

    for t in range(0, señal.shape[1] - window + 1, stride):
        bloque = señal[:, t:t + window]

        # ---- LIMPIO ----
        real = np.real(bloque)
        imag = np.imag(bloque)

        img = np.stack([real, imag], axis=-1)

        imagenes.append(img)

    return np.array(imagenes)

def plot_señales(imagenes, imagenes_ruido, idx=0):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    im = axes[0, 0].imshow(imagenes[idx][:, :, 0], cmap='viridis')
    axes[0, 0].set_title("Real limpia")
    plt.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(imagenes[idx][:, :, 1], cmap='viridis')
    axes[0, 1].set_title("Imag limpia")
    plt.colorbar(im, ax=axes[0, 1])

    im = axes[1, 0].imshow(imagenes_ruido[idx][:, :, 0], cmap='viridis')
    axes[1, 0].set_title("Real con ruido")
    plt.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].imshow(imagenes_ruido[idx][:, :, 1], cmap='viridis')
    axes[1, 1].set_title("Imag con ruido")
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

def plot_2d(imagenes, imagenes_ruido, idx=0):
    if idx < 0 or idx >= len(imagenes) or idx >= len(imagenes_ruido):
        raise ValueError(f"idx fuera de rango: {idx}")

    if imagenes[idx].shape[1] <= 32 or imagenes_ruido[idx].shape[1] <= 32:
        raise ValueError("No existe el subportador 64 en las imagenes")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    x = np.arange(imagenes[idx].shape[0])

    axes[0, 0].plot(x, imagenes[idx][32, :, 0])
    axes[0, 0].set_title("Real limpia")

    axes[0, 1].plot(x, imagenes[idx][32, :, 1])
    axes[0, 1].set_title("Imag limpia")

    axes[1, 0].plot(x, imagenes_ruido[idx][32, :, 0])
    axes[1, 0].set_title("Real con ruido")

    axes[1, 1].plot(x, imagenes_ruido[idx][32, :, 1])
    axes[1, 1].set_title("Imag con ruido")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    if (conf["Data"]["Sint"] == False):
        dset = conf["Data"]["Dset"]
        dsets = ['h_AAplant_int_5G', 'h_AAplant_int_2G', 'h_AAplant_5G', 'h_AAplant_2G', 'h_Boil_2G', 'h_Boil_5G', 'h_GBurg_2G', 'h_GBurg_5G']
        snr_db = conf["Data"]["Snr_db"]
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

        imagenes, imagenes_ruido, snrs = preprocesarDatos(data, 128, conf["Data"]["Stride"], snr_db)
        print(f"Numero de imagenes: {len(imagenes)}")

        snr_medio = np.median(snrs)
        print(f"SNR medio de las imagenes con ruido: {snr_medio:.2f} dB")

        plot_señales(imagenes, imagenes_ruido, 0)
        plot_2d(imagenes, imagenes_ruido, 0)

        np.save(f'data/NIST_{dset}_imagenes.npy', imagenes)
        print(f"Imagenes guardadas en data/NIST_{dset}_imagenes.npy")

        np.save(f'data/NIST_{dset}_imagenes_snr_{snr_db}.npy', imagenes_ruido)
        print(f"Imagenes con SNR {snr_db} guardadas en data/NIST_{dset}_imagenes_snr_{snr_db}.npy")

    else:
        dset = conf["KDR"]["Y"]
        dset_ruido = conf["KDR"]["X"]

        imagenes = np.load(dset)
        imagenes_ruido = np.load(dset_ruido)

        print(imagenes.shape)
        print(imagenes_ruido.shape)

        print(imagenes)
        print(imagenes_ruido)

        imagenes = preprocesarDatosSynt(imagenes, 128, conf["Data"]["Stride"])
        imagenes_ruido = preprocesarDatosSynt(imagenes_ruido, 128, conf["Data"]["Stride"])

        print(imagenes.shape)
        print(imagenes_ruido.shape)

        plot_señales(imagenes, imagenes_ruido)

        np.save(dset.replace(".npy", "128.npy"), imagenes)
        print(f"Imagenes guardadas en {dset.replace('.npy', '128.npy')}")

        np.save(dset_ruido.replace(".npy", "128.npy"), imagenes_ruido)
        print(f"Imagenes guardadas en {dset_ruido.replace('.npy', '128.npy')}")
