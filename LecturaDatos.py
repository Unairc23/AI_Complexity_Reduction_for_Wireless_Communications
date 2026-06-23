import json
import numpy as np
import h5py
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sympy import false

with open("config.json", "r", encoding="utf-8") as f:
    conf = json.load(f)

def preparar_directorios_folds():
    fold_real_dir = Path("data/folds/real")
    fold_est_dir = Path("data/folds/est")
    fold_real_dir.mkdir(parents=True, exist_ok=True)
    fold_est_dir.mkdir(parents=True, exist_ok=True)
    return fold_real_dir, fold_est_dir

def cargarDatosNist(dset='h_AAplant_int_5G'):
    f = h5py.File("NIST_Samples/NIST_Samples.mat",'r')
    data = f[dset]
    data = np.array(data)
    print (f"Shape original: {data.shape}")
    return data

def preprocesarDatos(data, window, stride, snr_values=[10, 15, 25]):
    data_clean = data[:, :window]
    print(f"Eliminando valores τ para evitar NaN: {data_clean.shape}")

    real = data_clean['real']
    imag = data_clean['imag']

    # Pasar NaN -> 0
    real = np.nan_to_num(real, nan=0.0)
    imag = np.nan_to_num(imag, nan=0.0)

    señal = real + 1j * imag

    imagenes = []
    imagenes_ruido = []
    snrs = []

    for t in range(0, señal.shape[0] - window + 1, stride):

        bloque = señal[t:t + window, :]
        bloque, _ = normalizar_complejo(bloque)

        real_clean = np.real(bloque)
        imag_clean = np.imag(bloque)
        img = np.stack([real_clean, imag_clean], axis=-1)

        for snr_db in snr_values:
            bloque_ruido = añadir_awgn_complejo(bloque, snr_db)

            real_ruido = np.real(bloque_ruido)
            imag_ruido = np.imag(bloque_ruido)

            img_ruido = np.stack([real_ruido, imag_ruido], axis=-1)

            snr_real = calcular_ruido(img, img_ruido)

            imagenes.append(img.copy())
            imagenes_ruido.append(img_ruido)
            snrs.append(snr_real)

    return imagenes, imagenes_ruido, snrs

def añadir_awgn_complejo(señal, snr_db):
    p_señal = np.mean(np.abs(señal) ** 2) # Potencia de la señal
    snr_linear = 10 ** (snr_db / 10) # Pasar SNR de dB a Linear
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

# Stride podria ser fijo, da mejores resultados con 128
def preprocesarDatosSynt(data, window, stride):
    real = np.real(data)
    imag = np.imag(data)

    # Pasar NaN a 0
    real = np.nan_to_num(real, nan=0.0)
    imag = np.nan_to_num(imag, nan=0.0)

    imagenes = []

    señal = real + 1j * imag

    for t in range(0, señal.shape[0] - window + 1, stride):
        bloque = señal[t:t + window, :]
        bloque, _ = normalizar_complejo(bloque)

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

def crear_imagenes_sint():
    dset = conf["KDR"]["Y"]
    dset_ruido = conf["KDR"]["X"]

    imagenes = np.load(dset)
    imagenes_ruido = np.load(dset_ruido)
    imagenes = np.transpose(imagenes, (1, 0))
    imagenes_ruido = np.transpose(imagenes_ruido, (1, 0))
    print(imagenes.shape)
    print(imagenes_ruido.shape)

    indices = np.arange(imagenes.shape[0])
    idx_train, idx_test = train_test_split(indices, test_size=0.3, shuffle=False)
    idx_val, idx_test = train_test_split(idx_test, test_size=0.5, shuffle=False)

    idx_train = np.sort(idx_train)
    idx_test = np.sort(idx_test)
    idx_val = np.sort(idx_val)

    imagenes_train = imagenes[idx_train, :]
    imagenes_test = imagenes[idx_test, :]
    imagenes_val = imagenes[idx_val, :]
    imagenes_ruido_train = imagenes_ruido[idx_train, :]
    imagenes_ruido_test = imagenes_ruido[idx_test, :]
    imagenes_ruido_val = imagenes_ruido[idx_val, :]

    imagenes_train = preprocesarDatosSynt(imagenes_train, 128, conf["Data"]["Stride"])
    imagenes_test = preprocesarDatosSynt(imagenes_test, 128, conf["Data"]["Stride"])
    imagenes_val = preprocesarDatosSynt(imagenes_val, 128, conf["Data"]["Stride"])
    imagenes_ruido_train = preprocesarDatosSynt(imagenes_ruido_train, 128, conf["Data"]["Stride"])
    imagenes_ruido_test = preprocesarDatosSynt(imagenes_ruido_test, 128, conf["Data"]["Stride"])
    imagenes_ruido_val = preprocesarDatosSynt(imagenes_ruido_val, 128, conf["Data"]["Stride"])

    print(imagenes_train.shape)
    print(imagenes_ruido_train.shape)

    plot_señales(imagenes_train, imagenes_ruido_train)
    plot_señales(imagenes_test, imagenes_ruido_test)
    plot_señales(imagenes_val, imagenes_ruido_val)

    np.save(dset.replace(".npy", "_Train.npy"), imagenes_train)
    print(f"Imagenes guardadas en {dset.replace('.npy', '_Train.npy')}")
    np.save(dset.replace(".npy", "_Test.npy"), imagenes_test)
    print(f"Imagenes guardadas en {dset.replace('.npy', '_Test.npy')}")
    np.save(dset_ruido.replace(".npy", "_Val.npy"), imagenes_ruido_val)
    print(f"Imagenes guardadas en {dset_ruido.replace('.npy', '_Val.npy')}")

    np.save(dset_ruido.replace(".npy", "_Train.npy"), imagenes_ruido_train)
    print(f"Imagenes guardadas en {dset_ruido.replace('.npy', '_Train.npy')}")
    np.save(dset_ruido.replace(".npy", "_Test.npy"), imagenes_ruido_test)
    print(f"Imagenes guardadas en {dset_ruido.replace('.npy', '_Test.npy')}")
    np.save(dset.replace(".npy", "_Val.npy"), imagenes_val)
    print(f"Imagenes guardadas en {dset.replace('.npy', '_Val.npy')}")

    if conf["Data"]["Kfold"]:
        folds = KFold(n_splits=5, shuffle=False)
        fold_real_dir, fold_est_dir = preparar_directorios_folds()
        dset_name = Path(dset).stem
        dsetR_name = Path(dset_ruido).stem
        for fold_idx, (train_idx, val_idx) in enumerate(folds.split(idx_train)):
            idx_fold_train = np.sort(idx_train[train_idx])
            idx_fold_val = np.sort(idx_train[val_idx])

            img_fold_train = preprocesarDatosSynt(imagenes[idx_fold_train, :], 128, conf["Data"]["Stride"])
            img_fold_val = preprocesarDatosSynt(imagenes[idx_fold_val, :], 128, conf["Data"]["Stride"])
            img_ruido_fold_train = preprocesarDatosSynt(imagenes_ruido[idx_fold_train, :], 128, conf["Data"]["Stride"])
            img_ruido_fold_val = preprocesarDatosSynt(imagenes_ruido[idx_fold_val, :], 128, conf["Data"]["Stride"])

            real_train_path = fold_real_dir / f'{dset_name}_{fold_idx}_Train.npy'
            real_val_path = fold_real_dir / f'{dset_name}_{fold_idx}_Val.npy'
            est_train_path = fold_est_dir / f'{dsetR_name}_{fold_idx}_Train.npy'
            est_val_path = fold_est_dir / f'{dsetR_name}_{fold_idx}_Val.npy'

            np.save(real_train_path, img_fold_train)
            np.save(real_val_path, img_fold_val)
            np.save(est_train_path, img_ruido_fold_train)
            np.save(est_val_path, img_ruido_fold_val)
            plot_señales(img_fold_train, img_ruido_fold_train)

            print(f"Fold {fold_idx} guardado en {real_train_path} y {est_train_path}")

def crear_imagenes():
    dset = conf["Data"]["Dset"]
    snr_db = conf["Data"]["Snr_db"]

    print("============== Cargando dataset individual ==============")
    data = cargarDatosNist(dset)

    indices = np.arange(data.shape[0])
    idx_train, idx_test = train_test_split(indices, test_size=0.3, shuffle=False)
    idx_val, idx_test = train_test_split(idx_test, test_size=0.5, shuffle=False)

    data_train = data[idx_train]
    data_test = data[idx_test]
    data_val = data[idx_val]

    stride = conf["Data"]["Stride"]

    imgs_train, imgs_ruido_train, snrs_train = preprocesarDatos(data_train, 128, stride, snr_db)
    imgs_test, imgs_ruido_test, snrs_test = preprocesarDatos(data_test, 128, stride, snr_db)
    imgs_val, imgs_ruido_val, snrs_val = preprocesarDatos(data_val, 128, stride, snr_db)
    print(f"Numero de imagenes: {len(imgs_train)}")

    snr_medio = np.median(snrs_train).astype(int)
    print(f"SNR medio de las imagenes con ruido: {snr_medio:2f} dB")
    plot_señales(imgs_train, imgs_ruido_train, 0)
    plot_2d(imgs_train, imgs_ruido_train, 0)

    np.save(f'data/NIST_{dset}_Train.npy', imgs_train)
    print(f"Imagenes guardadas en data/NIST_{dset}_Train.npy")
    np.save(f'data/NIST_{dset}_Test.npy', imgs_test)
    print(f"Imagenes guardadas en data/NIST_{dset}_Test.npy")
    np.save(f'data/NIST_{dset}_Val.npy', imgs_val)
    print(f"Imagenes guardadas en data/NIST_{dset}_Val.npy")

    np.save(f'data/NIST_{dset}_snr_{snr_medio}_Train.npy', imgs_ruido_train)
    print(f"Imagenes con SNR {snr_medio} guardadas en data/NIST_{dset}_snr_{snr_medio}_Train.npy")
    np.save(f'data/NIST_{dset}_snr_{snr_medio}_Test.npy', imgs_ruido_test)
    print(f"Imagenes con SNR {snr_medio} guardadas en data/NIST_{dset}_snr_{snr_medio}_Test.npy")
    np.save(f'data/NIST_{dset}_snr_{snr_medio}_Val.npy', imgs_ruido_val)
    print(f"Imagenes con SNR {snr_medio} guardadas en data/NIST_{dset}snr_{snr_medio}_Val.npy")

    if (conf["Data"]["Kfold"]):
        folds = KFold(n_splits=5, shuffle=False)
        fold_real_dir, fold_est_dir = preparar_directorios_folds()
        for fold_idx, (train_idx, val_idx) in enumerate(folds.split(idx_train)):
            idx_fold_train = np.sort(idx_train[train_idx])
            idx_fold_val = np.sort(idx_train[val_idx])

            data_fold_train = data[idx_fold_train]
            data_fold_val = data[idx_fold_val]

            imgs_fold_train, imgs_ruido_fold_train, snrs_fold_train = preprocesarDatos(data_fold_train, 128, stride, snr_db)
            imgs_fold_val, imgs_ruido_fold_val, snrs_fold_val = preprocesarDatos(data_fold_val, 128, stride, snr_db)

            real_train_path = fold_real_dir / f'NIST_{dset}_{fold_idx}_Train.npy'
            real_val_path = fold_real_dir / f'NIST_{dset}_{fold_idx}_Val.npy'
            est_train_path = fold_est_dir / f'NIST_{dset}_snr_{snr_medio}_{fold_idx}_Train.npy'
            est_val_path = fold_est_dir / f'NIST_{dset}_snr_{snr_medio}_{fold_idx}_Val.npy'

            np.save(real_train_path, imgs_fold_train)
            print(f"Imagenes fold {fold_idx} guardadas en {real_train_path}")
            np.save(real_val_path, imgs_fold_val)
            print(f"Imagenes fold {fold_idx} guardadas en {real_val_path}")
            np.save(est_train_path, imgs_ruido_fold_train)
            print(f"Imagenes con SNR {snr_medio} fold {fold_idx} guardadas en {est_train_path}")
            np.save(est_val_path, imgs_ruido_fold_val)
            print(f"Imagenes con SNR {snr_medio} fold {fold_idx} guardadas en {est_val_path}")

if __name__ == '__main__':
    if not conf["Data"]["Sint"]:
        crear_imagenes()
    else:
        crear_imagenes_sint()
