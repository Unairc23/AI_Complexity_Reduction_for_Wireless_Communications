import json
import numpy as np
import torch
from matplotlib import pyplot as plt
import datetime
import openpyxl

with open("config.json") as f:
    conf = json.load(f)

def calcular_ruido(señal_norm, señal_ruido_norm):
    ruido = señal_ruido_norm - señal_norm
    p_señal = np.mean(np.abs(señal_norm) ** 2)
    p_ruido = np.mean(np.abs(ruido) ** 2)
    if p_ruido == 0:
        return np.inf
    return 10 * np.log10(p_señal / p_ruido)

# ========================================REPRESENTACION / GUARDADO DATOS===============================================

def graficar(model, dataset, device, idx=0, modelName="modelo", modo="magnitud"):

    model.eval()

    x, y = dataset[idx]                # (C, H, W)
    x_in = x.unsqueeze(0).to(device)   # (1, C, H, W)

    with torch.no_grad():
        y_pred = model(x_in)

    # Mover a numpy
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    y_pred = y_pred.squeeze(0).cpu().numpy()

    if modo == "magnitud":
        # Calcular magnitud
        x_vis = np.sqrt(x[0]**2 + x[1]**2)
        y_vis = np.sqrt(y[0]**2 + y[1]**2)
        ypred_vis = np.sqrt(y_pred[0]**2 + y_pred[1]**2)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(x_vis, cmap="viridis")
        axs[0].set_title("Entrada ruidosa (|z|)")

        axs[1].imshow(y_vis, cmap="viridis")
        axs[1].set_title("Objetivo limpio (|z|)")

        axs[2].imshow(ypred_vis, cmap="viridis")
        axs[2].set_title(f"Reconstrucción {modelName} (|z|)")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    elif modo == "canales":
        fig, axs = plt.subplots(3, 2, figsize=(8, 12))

        # Entrada
        axs[0, 0].imshow(x[0], cmap="viridis")
        axs[0, 0].set_title("Entrada - Real")

        axs[0, 1].imshow(x[1], cmap="viridis")
        axs[0, 1].set_title("Entrada - Imaginario")

        # Objetivo
        axs[1, 0].imshow(y[0], cmap="viridis")
        axs[1, 0].set_title("Objetivo - Real")

        axs[1, 1].imshow(y[1], cmap="viridis")
        axs[1, 1].set_title("Objetivo - Imaginario")

        # Reconstrucción
        axs[2, 0].imshow(y_pred[0], cmap="viridis")
        axs[2, 0].set_title(f"Reconstrucción {modelName} - Real")

        axs[2, 1].imshow(y_pred[1], cmap="viridis")
        axs[2, 1].set_title(f"Reconstrucción {modelName} - Imaginario")

        for ax in axs.flatten():
            ax.axis("off")

        plt.tight_layout()
        plt.show()

def plot_training_curves(histories, title="Training curves"):
    plt.figure(figsize=(10,6))
    eps = 1e-12

    for name, (train_hist, val_hist) in histories.items():
        train_vals = np.maximum(np.asarray(train_hist, dtype=np.float64), eps)
        val_vals = np.maximum(np.asarray(val_hist, dtype=np.float64), eps)
        plt.plot(train_vals, linestyle="--", label=f"{name} - Train")
        plt.plot(val_vals, linestyle="-", label=f"{name} - Val")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def guardar_training_curves(histories):

    wb = openpyxl.Workbook()

    ws_wide = wb.active
    ws_wide.title = "wide"

    header = ["epoch"]
    max_len = 0
    normalized = {}

    for name, (train_hist, val_hist) in histories.items():
        train_vals = list(train_hist)
        val_vals = list(val_hist)
        normalized[name] = (train_vals, val_vals)
        max_len = max(max_len, len(train_vals), len(val_vals))
        header.extend([f"{name}_train", f"{name}_val"])

    ws_wide.append(header)

    for epoch_idx in range(max_len):
        row = [epoch_idx + 1]
        for name in histories.keys():
            train_vals, val_vals = normalized[name]
            train_value = train_vals[epoch_idx] if epoch_idx < len(train_vals) else None
            val_value = val_vals[epoch_idx] if epoch_idx < len(val_vals) else None
            row.extend([train_value, val_value])
        ws_wide.append(row)

    ws_long = wb.create_sheet(title="long")
    ws_long.append(["model", "epoch", "train_loss", "val_loss"])

    for name, (train_vals, val_vals) in normalized.items():
        local_max = max(len(train_vals), len(val_vals))
        for epoch_idx in range(local_max):
            train_value = train_vals[epoch_idx] if epoch_idx < len(train_vals) else None
            val_value = val_vals[epoch_idx] if epoch_idx < len(val_vals) else None
            ws_long.append([name, epoch_idx + 1, train_value, val_value])

    ws_params = wb.create_sheet(title="params")
    flat_conf = flatten_dict(conf)
    for name, value in flat_conf.items():
        ws_params.append([name, value])

    fechaHora = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"results/t{conf["KDR"]["tDepth"]}_s{conf["KDR"]["sDepth"]}_{conf["Data"]["Snr_db"]}db_{fechaHora}.xlsx"
    wb.save(output_path)
    print(f"Curvas de entrenamiento guardadas en {output_path}")

def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def register_hook(model, model_name, container, name, layers=-2):
    if model_name == "DnCNN":
        layer = model.dncnn[layers] # Igual cambiar esto tambien para no usar un indice que se elige manualmente
        print(layer)
    elif model_name == "UNet":
        layer = model.bottleneck
    else:
        raise ValueError(f"Modelo {model_name} no soporta feature hooks")
    layer.register_forward_hook(save_activation(container, name))

# Función para extraer features intermedias del modelo sin necesitar una arquitectura especifica o modificarlo
def save_activation(container, name):
    def hook(module, input, output):
        container[name] = output
    return hook