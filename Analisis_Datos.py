import json
import numpy as np
import matplotlib.pyplot as plt
from LecturaDatos import cargarDatosNist

def graficar_img(dataset):
    N = dataset.shape[0]
    indices = [5, N//3, 2*N//3, N-1]
    vmin = np.min(np.abs(dataset[:, :, :, 0] + 1j * dataset[:, :, :, 1]))
    vmax = np.max(np.abs(dataset[:, :, :, 0] + 1j * dataset[:, :, :, 1]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, idx in zip(axes.ravel(), indices):
        sample = dataset[idx]

        H = sample[:, :, 0] + 1j * sample[:, :, 1]

        im = ax.imshow(np.abs(H), aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f'Muestra {idx}', fontsize=20)
        ax.set_xlabel('Retardo (τ)', fontsize=20)
        ax.set_ylabel('Tiempo (t)', fontsize=20)
        ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout(pad=2.0)
    plt.show()

def graficar(dataset):
    N = dataset.shape[0]
    indices = [5, N//3, 2*N//3, N-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, idx in zip(axes.ravel(), indices):
        sample = dataset[idx]

        H = sample[64, :, 0] + 1j * sample[64, :, 1]
        im = ax.plot(np.abs(H))

        ax.set_title(f'Muestra {idx}', fontsize=18)
        ax.set_xlabel('Retardo (τ)', fontsize=18)
        ax.set_ylabel('Magnitud', fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid()

    plt.tight_layout(pad=2.0)
    plt.show()

with open("config.json", "r", encoding="utf-8") as f:
    conf = json.load(f)

def comp_datasets():
    dset_real = "data/NIST_h_GBurg_5G.npy"
    dset_real_ruido = "data/NIST_h_GBurg_5G_snr_16.npy"
    dset_sint = "data/Datos_Sinteticos/TDL_D_85ns_fd1000_SNR20_h_real.npy"
    dset_sint_ruido = "data/Datos_Sinteticos/TDL_D_85ns_fd1000_SNR20_h_est.npy"

    # real limpia
    real_train = np.load(dset_real.replace(".npy", "_Train.npy"))
    real_test = np.load(dset_real.replace(".npy", "_Test.npy"))
    real_val = np.load(dset_real.replace(".npy", "_Val.npy"))
    R = np.concatenate([real_train, real_test, real_val], axis=0)
    print(f"Real limpia: {R.shape}")
    R_real  = R[:, :, :, 0].ravel()
    R_imag  = R[:, :, :, 1].ravel()
    # real ruido
    realR_train = np.load(dset_real_ruido.replace(".npy", "_Train.npy"))
    realR_test = np.load(dset_real_ruido.replace(".npy", "_Test.npy"))
    realR_val = np.load(dset_real_ruido.replace(".npy", "_Val.npy"))
    Rr = np.concatenate([realR_train, realR_test, realR_val], axis=0)
    print(f"Real ruidosa: {Rr.shape}")
    Rr_real  = Rr[:, :, :, 0].ravel()
    Rr_imag  = Rr[:, :, :, 1].ravel()

    # sint limpia
    sint_train = np.load(dset_sint.replace(".npy", "_Train.npy"))
    sint_test = np.load(dset_sint.replace(".npy", "_Test.npy"))
    sint_val = np.load(dset_sint.replace(".npy", "_Val.npy"))
    S = np.concatenate([sint_train, sint_test, sint_val], axis=0)
    print(S.shape)
    S_real  = S[:, :, :, 0].ravel()
    S_imag  = S[:, :, :, 1].ravel()
    # sint ruido
    sintR_train = np.load(dset_sint_ruido.replace(".npy", "_Train.npy"))
    sintR_test = np.load(dset_sint_ruido.replace(".npy", "_Test.npy"))
    sintR_val = np.load(dset_sint_ruido.replace(".npy", "_Val.npy"))
    Sr = np.concatenate([sintR_train, sintR_test, sintR_val], axis=0)
    print(S.shape)
    Sr_real  = Sr[:, :, :, 0].ravel()
    Sr_imag  = Sr[:, :, :, 1].ravel()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bins = 100
    alpha = 0.6

    # Limpia real
    axes[0,0].hist(R_real,  bins=bins, alpha=alpha, label='real', density=True)
    axes[0,0].hist(R_imag,  bins=bins, alpha=alpha, label='imag', density=True)
    axes[0,0].set_title('Señal real limpia'); axes[0,0].legend()
    axes[0,0].grid()

    # Ruidosa real
    axes[0,1].hist(Rr_real,  bins=bins, alpha=alpha, label='real', density=True)
    axes[0,1].hist(Rr_imag,  bins=bins, alpha=alpha, label='imag', density=True)
    axes[0,1].set_title('Señal real ruidosa'); axes[0,1].legend()
    axes[0,1].grid()

    # Limpia sintética
    axes[1,0].hist(S_real,   bins=bins, alpha=alpha, label='real', density=True)
    axes[1,0].hist(S_imag,   bins=bins, alpha=alpha, label='imag', density=True)
    axes[1,0].set_title('Señal sintética limpia'); axes[1,0].legend()
    axes[1,0].grid()

    # Fase
    axes[1,1].hist(Sr_real, bins=bins, alpha=alpha, label='real', density=True)
    axes[1,1].hist(Sr_imag, bins=bins, alpha=alpha, label='imag', density=True)
    axes[1,1].set_title('Señal sintética ruidosa'); axes[1,1].legend()
    axes[1,1].grid()

    plt.tight_layout()
    plt.savefig('histogramas.png', dpi=150)
    plt.show()

    graficar(R)
    graficar(Rr)
    graficar(S)
    graficar(Sr)

    graficar_img(R)
    graficar_img(Rr)
    graficar_img(Sr)
    graficar_img(Sr)

if __name__ == "__main__":
    comp_datasets()
