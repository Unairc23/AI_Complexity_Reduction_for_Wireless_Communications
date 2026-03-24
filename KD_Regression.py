import json
import os
import datetime
from glob import glob
from operator import contains
import openpyxl

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import random_split, Dataset
import torch.nn.functional as F
import random, numpy as np

# Todo: Implementar / quitar modelos no compatibles con regresion
from modelos import DeepNN, LightNN, LightNN_Adaptada, DeepNN_Adaptada, load_resnet, DnCNN, ResNetDenoiser

with open("config.json", "r", encoding="utf-8") as f:
    conf = json.load(f)

MODEL_REGISTRY = {
    "resnet18": lambda: load_resnet(),
    "deep": lambda: DeepNN(num_classes=10),
    "deep_adaptada": lambda: DeepNN_Adaptada(num_classes=10),
    "light": lambda: LightNN(num_classes=10),
    "light_adaptada": lambda: LightNN_Adaptada(num_classes=10),
    "DnCNN": {
        "Student": lambda: DnCNN(depth=conf["KDR"]["sDepth"]),
        "Teacher": lambda: DnCNN(depth=conf["KDR"]["tDepth"])
    },
    "resnet_denoiser": {
        "Student": lambda: ResNetDenoiser(in_channels=2, base_channels=16),
        "Teacher": lambda: ResNetDenoiser(in_channels=2, base_channels=32)
    }
}

# ============================================== BSD500 Loader ========================================================

# Todo: Eliminar / mover a otro lugar los metodos de las imagenes BSD500


class NPYDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]   # (128, 128, 1)
        y = self.Y[idx]

        # (H, W, C) -> (C, H, W)
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
        y = torch.tensor(y, dtype=torch.float32).permute(2, 0, 1)

        if self.transform:
            x = self.transform(x)
        return x, y

# ========================================== Configurar torch / seed ===================================================
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
torch.cuda.empty_cache()

torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================== Crear dataset ========================================================
if (contains(conf["KDR"]["X"], "bsd500")):
    prepare_bsd500_dataset(conf)

X = np.load(conf["KDR"]["X"])
Y = np.load(conf["KDR"]["Y"])

full_dataset = NPYDataset(X, Y)

total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    full_dataset,
    [train_size, val_size, test_size]
)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

# ====================================== Entrenamiento de los modelos =================================================
def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStoppingLoss(patience=conf["KDR"]["patience"])

    train_history = []
    val_history = []

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for X, Y in train_loader:
            # X: Señal ruidoso
            # Y: Señal limpia/original
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            outputs = model(X)

            if (conf["Data"]["Unica"]):
                Y = Y[:, :, 64, :]
                outputs = outputs[:, :, 64, :]

            # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
            # X: The actual images. Vector of dimensionality batch_size
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train_Loss: {train_loss:.8f} | Val_Loss: {val_loss:.8f}")

        if early_stopper.step(val_loss, model):
            print("Parando el entrenamiento")
            break

    early_stopper.restore(model)

    return train_history, val_history

def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, teacher_threshold, alpha, device):
    #Todo: Ahora mismo el teacher calcula el mse de todo el batch, no solo de la prediccion actual
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    early_stopper = EarlyStoppingLoss(patience=conf["KDR"]["patience"])

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    train_history = []
    val_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_pred = teacher(X)
            # Forward pass with the student model
            student_pred = student(X)

            if (conf["Data"]["Unica"]):
                Y = Y[:, :, 64, :]
                teacher_pred = teacher_pred[:, :, 64, :]
                student_pred = student_pred[:, :, 64, :]

            # teacher_y_loss = mse_loss(teacher_pred, Y)
            student_y_loss = mse_loss(student_pred, Y)

            teacher_student_loss = mse_loss(student_pred, teacher_pred)
            loss = alpha * teacher_student_loss + (1.0 - alpha) * student_y_loss

            # if teacher_y_loss < teacher_threshold:
            #     teacher_student_loss = mse_loss(student_pred, teacher_pred)
            #     loss = alpha * teacher_student_loss + (1.0 - alpha) * student_y_loss
            # else:
            #     loss = student_y_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        val_loss = evaluate(student, val_loader, device)

        train_history.append(epoch_loss)
        val_history.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} | Train_Loss: {epoch_loss:.8f} | Val_Loss: {val_loss:.8f}")

        if early_stopper.step(val_loss, student):
            print("Early stopping triggered!")
            break

    early_stopper.restore(student)

    return train_history, val_history

def train_feature_based_kd(teacher, student, train_loader, epochs, learning_rate, alpha, device):
    #Todo: - Implementar bottleneck para los casos en los que la arquitectura de teacher y student sea diferente.
    # (KNOWLEDGE DISTILLATION FOR SPEECH DENOISING BY LATENT REPRESENTATION ALIGNMENT WITH COSINE DISTANCE)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    early_stopper = EarlyStoppingLoss(patience=conf["KDR"]["patience"])

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    train_history = []
    val_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_pred = teacher(X)
                t_latent = teacher_features["latent"]
                # t_latent = bottleneck(t_latent)

            # Forward pass with the student model
            student_pred = student(X)
            s_latent = student_features["latent"]

            if (conf["Data"]["Unica"]):
                Y = Y[:, :, 64, :]
                student_pred = student_pred[:, :, 64, :]

            out_loss = mse_loss(student_pred, Y)
            kd_loss = cosine_kd_loss(s_latent, t_latent)

            loss = out_loss + alpha * kd_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        val_loss = evaluate(student, val_loader, device)

        train_history.append(epoch_loss)
        val_history.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.8f} | Val_Loss: {val_loss:.8f}")

        if early_stopper.step(val_loss, student):
            print("Early stopping triggered!")
            break

    early_stopper.restore(student)

    return train_history, val_history

def cosine_kd_loss(h_s, h_t):  # h_s, h_t: [B, C, H, W]
    h_s = F.normalize(h_s.flatten(1), dim=1)
    h_t = F.normalize(h_t.flatten(1), dim=1)
    return 1 - (h_s * h_t).sum(dim=1).mean()

# ============================================== Early stopping ========================================================
def evaluate(model, loader, device):
    # TODO: Probar (ssim)
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            if (conf["Data"]["Unica"]):
                Y = Y[:, :, 64, :]
                pred = pred[:, :, 64, :]

            total_loss += mse(pred, Y).item()

    return total_loss / len(loader)

# Esto queda un poco inutilizado porque no es una señal tan "Comparable" como seria MSE
def evaluate_psnr(model, loader, device):
    model.eval()
    total_psnr = 0.0
    eps = 1e-10

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            mse = torch.mean((pred - Y) ** 2, dim=[1, 2, 3])
            psnr = 10 * torch.log10(1.0 / (mse + eps))

            total_psnr += psnr.mean().item()

    return total_psnr / len(loader)

class EarlyStoppingLoss:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss + self.min_delta: # Para mse < / Para psnr >
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# ============================================== Carga y Parser ========================================================
def load_model(model, path, device):
    # Todo: Arreglar esto
    """Carga un modelo si el archivo existe, o devuelve uno nuevo."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f" Modelo cargado desde {path}")
    else:
        print(f" No se encontró el archivo {path}, se inicializa un modelo nuevo.")
    return model

def estimate_snr_db(noisy, clean, eps=1e-12):
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((noisy - clean) ** 2)
    return 10.0 * np.log10((signal_power + eps) / (noise_power + eps))

def calcular_ruido(señal_norm, señal_ruido_norm):
    ruido = señal_ruido_norm - señal_norm
    p_señal = np.mean(np.abs(señal_norm) ** 2)
    p_ruido = np.mean(np.abs(ruido) ** 2)
    if p_ruido == 0:
        return np.inf
    return 10 * np.log10(p_señal / p_ruido)

# ==========================================Representar graficamente====================================================
# TODO: Igual mover metodos como este a una clase especifica para evitar centrar todo aquí
def graficar(model, dataset, device, idx=0, modelName="modelo", modo="magnitud"):

    model.eval()

    x, y = dataset[idx]                # (C, H, W)
    x_in = x.unsqueeze(0).to(device)   # (1, C, H, W)

    with torch.no_grad():
        y_pred = model(x_in)

    # Mover a numpy
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    np.save(f"{modelName}_input.npy", x)
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

    fechaHora = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"results/resultados_{fechaHora}.xlsx"
    wb.save(output_path)
    print(f"Curvas de entrenamiento guardadas en {output_path}")

if __name__ == "__main__":
    # Creo que esta linea solo es encesaria en windows, linux hace un fork
    torch.multiprocessing.freeze_support()

    tModel = conf["KDR"]["tModel"]
    sModel = conf["KDR"]["sModel"]

    teacher = MODEL_REGISTRY[tModel]["Teacher"]().to(device)
    student = MODEL_REGISTRY[sModel]["Student"]().to(device)

    teacher_features = {}
    student_features = {}

    # Función para extraer features intermedias del modelo sin necesitar una arquitectura especifica o modificarlo
    def save_activation(container, name):
        def hook(module, input, output):
            container[name] = output
        return hook

    # TODO: Ahora mismo feature based solo funciona con dncnn, cualquier otro modelo rompe
    teacher.dncnn[conf["KDR"]["Features"]["t_layers"]].register_forward_hook(
        save_activation(teacher_features, "latent")
    )

    if conf["KDR"]["train"]:
        if tModel != "resnet18": # Los modelos resnet usados ya están preentrenados con CIFAR10
            print("================ Entrenando teacher ================")
            teacher_hist = train(model=teacher, train_loader=train_loader, epochs=conf["KDR"]["tEpoch"], learning_rate=conf["KDR"]["lr"],
                  device=device)
        print("================ Entrenando no_KD_student ================")
        student_hist = train(model=student, train_loader=train_loader, epochs=conf["KDR"]["sEpoch"], learning_rate=conf["KDR"]["lr"],
              device=device)
        torch.save(teacher.state_dict(), f"model/teacher_{tModel}.pth")
        torch.save(student.state_dict(), f"model/student_{sModel}.pth")
    else:
        teacher = load_model(teacher, path=f"model/{tModel}.pth", device=device)
        student = load_model(student, path=f"model/{sModel}.pth", device=device)

    # Comparar tamaño teacher / modelo sin destilar
    teacher_params = "{:,}".format(sum(p.numel() for p in teacher.parameters()))
    teacher_size = os.path.getsize(f"model/teacher_{tModel}.pth") / 1024 ** 2
    print(f"Teacher Params: {teacher_params}")
    print(f"Teacher Size: {teacher_size}")

    student_params = "{:,}".format(sum(p.numel() for p in student.parameters()))
    student_size = os.path.getsize(f"model/student_{tModel}.pth") / 1024 ** 2
    print(f"Student Params: {student_params}")
    print(f"Student Size: {student_size}")

    print(f"Diferencia de tamaño: {teacher_size / student_size:.2f}x ({teacher_size:.2f}MB -> {student_size:.2f}MB)")

    # Comparacion de resultados entre teacher y modelo sin destilar
    idx = int(conf["KDR"].get("plot_idx", 0))
    idx = max(0, min(idx, len(test_ds) - 1))
    print(f"Mostrando muestra de test idx={idx}")
    graficar(student, test_ds, device, idx=idx, modelName="no_kd_student", modo="canales")
    graficar(teacher, test_ds, device, idx=idx, modelName="teacher", modo="canales")

    # xs = []
    snrs = []
    for data in test_ds:
        x, y = data
        snr_db = calcular_ruido(señal_ruido_norm=x.numpy(), señal_norm=y.numpy())
        snrs.append(snr_db)
        xs.append(x.numpy()[1,:,:])
    print(f"SNR medio del dataset de test: {np.mean(snrs):.2f} dB")
    # xs = np.stack(xs)
    # np.save("input.npy", xs)
    # print(f"Guardadas {xs.shape} muestras X en input.npy")

    # ============================================== KD clasica ========================================================
    print("================ Entrenando kd_student ================")
    kd_student = MODEL_REGISTRY[sModel]["Student"]().to(device)

    kd_student.dncnn[conf["KDR"]["Features"]["s_layers"]].register_forward_hook(
        save_activation(student_features, "latent")
    )

    kd_hist = train_knowledge_distillation(teacher=teacher, student=kd_student, train_loader=train_loader,
                                 epochs=conf["KDR"]["sEpoch"], learning_rate=conf["KDR"]["lr"], device=device,
                                 teacher_threshold=0.001, alpha=conf["KDR"]["alpha"])

    # Comparacion entre teacher y modelo destilado
    graficar(kd_student, test_ds, device, idx=idx, modelName="kd_student", modo="canales")

    # =========================================== Feature based KD =====================================================
    print("================ Entrenando feature_kd_student ================")
    kd_student_feature = MODEL_REGISTRY[sModel]["Student"]().to(device)

    kd_student_feature.dncnn[conf["KDR"]["Features"]["s_layers"]].register_forward_hook(
        save_activation(student_features, "latent")
    )

    fkd_hist = train_feature_based_kd(teacher=teacher, student=kd_student_feature, train_loader=train_loader, epochs=conf["KDR"]["sEpoch"],
                           learning_rate=conf["KDR"]["lr"], device=device, alpha=conf["KDR"]["alpha"])

    graficar(kd_student_feature, test_ds, device, idx=idx, modelName="kd_student_feature", modo="canales")

    historial = {
        "Teacher": teacher_hist,
        "Student": student_hist,
        "KD": kd_hist,
        "FKD": fkd_hist
    }
    plot_training_curves(historial)
    guardar_training_curves(historial)