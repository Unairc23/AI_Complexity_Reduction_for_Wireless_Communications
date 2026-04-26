import json
import os
import datetime
from glob import glob
from operator import contains
import openpyxl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.nn.quantized as nnq
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import random_split, Dataset
import torch.nn.functional as F
import random, numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Todo: Implementar / quitar modelos no compatibles con regresion
from modelos import DeepNN, LightNN, LightNN_Adaptada, DeepNN_Adaptada, load_resnet, DnCNN, ResNetDenoiser, UNetDenoiser
from utils import *
from Cuantizacion import *

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
    },
    "UNet": {
        "Student": lambda: UNetDenoiser(in_channels=2, base_channels=3),
        "Teacher": lambda: UNetDenoiser(in_channels=2, base_channels=64)
    }
}

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
batch_size = conf["KDR"]["batch_size"]

full_dataset = NPYDataset(X, Y)

total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    full_dataset,
    [train_size, val_size, test_size]
)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

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
        student_running_loss = 0.0 # Usado simplemente para ver la diferencia train/val, usando la del KD esta siempre sera mejor
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
            student_running_loss += student_y_loss.item()

        epoch_loss = student_running_loss / len(train_loader)
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
    #Todo: - Tratar de mejorar el bottleneck (warming)
    mse_loss = nn.MSELoss()
    early_stopper = EarlyStoppingLoss(patience=conf["KDR"]["patience"]+8)

    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.train()

    with torch.no_grad():
        dummy = next(iter(train_loader))[0][:1].to(device)
        teacher(dummy)
        t_ch = teacher_features["latent"].shape[1]
        student(dummy)
        s_ch = fkd_features["latent"].shape[1]
    print(f"Teacher shape: {teacher_features["latent"].shape}")
    print(f"Student shape: {fkd_features["latent"].shape}")

    bottleneck_proj = nn.Conv2d(t_ch, s_ch, kernel_size=1, bias=False).to(device)
    print(f"Linear bottleneck: {t_ch} → {s_ch} canales")

    optimizer = optim.Adam([
        {"params": student.parameters(), "lr": learning_rate},
        {"params": bottleneck_proj.parameters(), "lr": learning_rate * 10}
    ])

    train_history = []
    val_history = []

    for epoch in range(epochs):
        student.train()
        bottleneck_proj.train()
        running_loss = 0.0
        student_running_loss = 0.0

        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher(X)
                t_latent = teacher_features["latent"].detach()

            student_pred = student(X)
            s_latent = fkd_features["latent"]

            if conf["Data"]["Unica"]:
                Y = Y[:, :, 64, :]
                student_pred = student_pred[:, :, 64, :]

            t_latent_proj = bottleneck_proj(t_latent)

            out_loss = mse_loss(student_pred, Y)
            kd_loss = cosine_kd_loss(s_latent, t_latent_proj)
            print(f"Loss: {out_loss:.8f} / KD_Loss: {kd_loss:.8f}")
            loss = out_loss + alpha * kd_loss

            loss.backward()
            optimizer.step()

            student_running_loss += out_loss.item()
            running_loss += loss.item()

        epoch_loss = student_running_loss / len(train_loader)
        val_loss = evaluate(student, val_loader, device)

        train_history.append(epoch_loss)
        val_history.append(val_loss)

        if early_stopper.step(val_loss, student):
            print("Early stopping triggered!")
            break

    early_stopper.restore(student)
    return train_history, val_history

def cosine_kd_loss(h_s, h_t):  # h_s, h_t: [B, C, H, W]
    B, C_s, H, W = h_s.shape
    _, C_t, _, _ = h_t.shape
    print(C_t, C_s)

    if (conf["KDR"]["features"] == "spatial"):
        h_s = h_s.permute(0, 2, 3, 1).reshape(B * H * W, C_s)
        h_t = h_t.permute(0, 2, 3, 1).reshape(B * H * W, C_t)
    elif (conf["KDR"]["features"] == "channel"):
        h_s = h_s.reshape(B, C_s, H*W)
        h_t = h_t.reshape(B, C_t, H*W)
    elif (conf["KDR"]["features"] == "global"):
        h_s = h_s.reshape(B, C_s * H * W)
        h_t = h_t.reshape(B, C_t * H * W)

    return 1 - F.cosine_similarity(h_s, h_t, dim=-1).mean()

# def cosine_kd_loss(h_s, h_t):
#     return 1 - ((np.dot(h_s, h_t)) / (np.linalg.norm(h_s) * np.linalg.norm(h_t)))


def train_attention_kd(teacher, student, t_feats, s_feats, train_loader, epochs, learning_rate, alpha, device):
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    early_stopper = EarlyStoppingLoss(patience=conf["KDR"]["patience"])

    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.train()

    train_history = []
    val_history = []

    for epoch in range(epochs):
        student_running_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher(X)
                t_enc1 = t_feats["enc1"].detach()
                t_enc2 = t_feats["enc2"].detach()
                t_bottle = t_feats["bottleneck"].detach()

            student_pred = student(X)
            s_enc1 = s_feats["enc1"]
            s_enc2 = s_feats["enc2"]
            s_bottle = s_feats["bottleneck"]

            out_loss = mse_loss(student_pred, Y)
            at_loss = (
                    attention_transfer_loss(s_enc1, t_enc1) +
                    attention_transfer_loss(s_enc2, t_enc2) +
                    attention_transfer_loss(s_bottle, t_bottle)
            )
            print(f"Loss: {out_loss:.8f} / ATLoss: {at_loss:.8f}") # AtLoss es dos ordenes de magnitud mas pequeño
            loss = out_loss + alpha * at_loss

            loss.backward()
            optimizer.step()
            student_running_loss += out_loss.item()

        epoch_loss = student_running_loss / len(train_loader)
        val_loss = evaluate(student, val_loader, device)
        train_history.append(epoch_loss)
        val_history.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Train_Loss: {epoch_loss:.8f} | Val_Loss: {val_loss:.8f}")

        if early_stopper.step(val_loss, student):
            print("Early stopping triggered!")
            break

    early_stopper.restore(student)
    return train_history, val_history


# ============================================== Early stopping ========================================================
def evaluate(model, loader, device):
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

# ============================================== CARGA MODELOS ========================================================
def load_model(model, path, device):
    # Todo: Arreglar esto (Como tal funciona, pero cuidado de no cargar con diferentes tamaños al guardado)
    """Carga un modelo si el archivo existe, o devuelve uno nuevo."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Modelo cargado desde {path}")
    else:
        print(f"No se encontó el archivo {path}, se inicializa un modelo nuevo.")
    return model

if __name__ == "__main__":
    # Creo que esta linea solo es encesaria en windows, linux hace un fork
    torch.multiprocessing.freeze_support()

    tModel = conf["KDR"]["tModel"]
    t_tamaño = conf["KDR"]["tDepth"]
    sModel = conf["KDR"]["sModel"]
    s_tamaño = conf["KDR"]["sDepth"]
    snr = conf["Data"]["Snr_db"]

    teacher = MODEL_REGISTRY[tModel]["Teacher"]().to(device)
    student = MODEL_REGISTRY[sModel]["Student"]().to(device)

    teacher_features = {}
    teacher_attentions = {}

    register_hook(teacher, tModel, teacher_features, "latent", conf["KDR"]["Features"]["t_layers"])
    register_hooks_at(teacher, teacher_attentions)

    if conf["KDR"]["train"]:
        if tModel != "resnet18": # Los modelos resnet usados ya están preentrenados con CIFAR10
            print("================ Entrenando teacher ================")
            teacher_hist = train(model=teacher, train_loader=train_loader, epochs=conf["KDR"]["tEpoch"], learning_rate=conf["KDR"]["lr"],
                  device=device)
        print("================ Entrenando no_KD_student ================")
        student_hist = train(model=student, train_loader=train_loader, epochs=conf["KDR"]["sEpoch"], learning_rate=conf["KDR"]["lr"],
              device=device)

        torch.save(teacher.state_dict(), f"model/{tModel}_{t_tamaño}l_{snr}snr.pth")
        torch.save(student.state_dict(), f"model/{sModel}_{s_tamaño}l_{snr}snr.pth")
    else:
        teacher = load_model(teacher, path=f"model/{tModel}_{t_tamaño}l_{snr}snr.pth", device=device)
        student = load_model(student, path=f"model/{sModel}_{s_tamaño}l_{snr}snr.pth", device=device)

    # Comparar tamaño teacher / modelo sin destilar
    teacher_params = "{:,}".format(sum(p.numel() for p in teacher.parameters()))
    teacher_size = os.path.getsize(f"model/{tModel}_{t_tamaño}l_{snr}snr.pth") / 1024 ** 2
    print(f"\nTeacher Params: {teacher_params}")
    print(f"Teacher Size: {teacher_size} \n")

    student_params = "{:,}".format(sum(p.numel() for p in student.parameters()))
    student_size = os.path.getsize(f"model/{sModel}_{s_tamaño}l_{snr}snr.pth") / 1024 ** 2
    print(f"Student Params: {student_params}")
    print(f"Student Size: {student_size} \n")

    print(f"Diferencia de tamaño: {teacher_size / student_size:.2f}x ({teacher_size:.2f}MB -> {student_size:.2f}MB)")

    # Comparacion de resultados entre teacher y modelo sin destilar
    idx = int(conf["KDR"].get("plot_idx", 0))
    idx = max(0, min(idx, len(test_ds) - 1))
    print(f"Mostrando muestra de test idx={idx}")
    graficar(student, test_ds, device, idx=idx, modelName="no_kd_student", modo="canales")
    graficar(teacher, test_ds, device, idx=idx, modelName="teacher", modo="canales")

    snr = []
    mseT = evaluate(teacher, val_loader, device)
    mseS = evaluate(student, val_loader, device)
    for x,y in val_loader: # For simple para comprobar que el ruido está correctamente aplicado en el datasr (se puede quitar)
        snr_db = calcular_ruido(señal_ruido_norm=x.numpy(), señal_norm=y.numpy())
        snr.append(snr_db)
    print(f"MSE teacher: {mseT:.8f}")
    print(f"MSE student: {mseS:.8f}")
    print(f"SNR medio del dataset de test: {np.mean(snr):.2f} dB")

    # ============================================== Cuantizar ========================================================

    if (conf["KDR"]["cuantizar"]):
        print("============================ Cuantizando ============================")

        teacher_q = cuantizar_estatica(teacher, device, val_loader)
        student_q = cuantizar_estatica(student, device, val_loader)
        cpu = torch.device("cpu")
        teacher_q.to(cpu)
        student_q.to(cpu) # La cuantización se aplica sobre cpu siempre, pasar modelos explicitamente a gpu

        # Comparacion de tamaños
        torch.save(teacher_q.state_dict(), "model/teacher_q.pth")
        torch.save(student_q.state_dict(), "model/student_q.pth")

        quantized_teacher = os.path.getsize("model/teacher_q.pth") / 1024 ** 2
        print(f"Quantized Teacher Size: {quantized_teacher}")
        quantized_student = os.path.getsize("model/student_q.pth") / 1024 ** 2
        print(f"Quantized Student Size: {quantized_student}\n")
        print(f"Diferencia de tamaño teacher: {quantized_teacher / teacher_size:.2f}x ({quantized_teacher:.2f}MB -> {teacher_size:.2f}MB)")
        print(f"Diferencia de tamaño student: {quantized_student / student_size:.2f}x ({quantized_student:.2f}MB -> {student_size:.2f}MB)")

        idx = int(conf["KDR"].get("plot_idx", 0))
        idx = max(0, min(idx, len(test_ds) - 1))
        print(f"Mostrando muestra de test idx={idx}\n")
        graficar(teacher_q, test_ds, cpu, idx=idx, modelName="teacher_q", modo="canales")
        graficar(student_q, test_ds, cpu, idx=idx, modelName="student_q", modo="canales")

        mseTq = evaluate(teacher_q, val_loader, cpu)
        mseSq = evaluate(student_q, val_loader, cpu)
        print(f"MSE teacherq: {mseTq:.8f} / teacher_noQ: {mseT:.8f}: ")
        print(f"MSE studentq: {mseSq:.8f} / student_noQ: {mseS:.8f}: ")
        print(f"Diferencia Teacher: {mseTq - mseT:.8f} ({(mseTq - mseT)/mseT:.2f}%)")
        print(f"Diferencia Student: {mseSq - mseS:.8f} ({(mseSq - mseS)/mseS:.2f}%)")

    # ============================================== KD clasica ========================================================
    if (conf["KDR"]["KD"]):
        print("================ Entrenando kd_student ================")
        kd_student = MODEL_REGISTRY[sModel]["Student"]().to(device)

        kd_hist = train_knowledge_distillation(teacher=teacher, student=kd_student, train_loader=train_loader,
                                     epochs=conf["KDR"]["sEpoch"], learning_rate=conf["KDR"]["lr"], device=device,
                                     teacher_threshold=0.001, alpha=conf["KDR"]["alpha"])

        # Comparacion entre teacher y modelo destilado
        graficar(kd_student, test_ds, device, idx=idx, modelName="kd_student", modo="canales")

    # =========================================== Feature based KD =====================================================
    if (conf["KDR"]["FKD"]):
        print("================ Entrenando feature_kd_student ================")
        kd_student_feature = MODEL_REGISTRY[sModel]["Student"]().to(device)

        fkd_features = {}
        register_hook(kd_student_feature, sModel, fkd_features, "latent", conf["KDR"]["Features"]["s_layers"])

        fkd_hist = train_feature_based_kd(teacher=teacher, student=kd_student_feature, train_loader=train_loader, epochs=conf["KDR"]["sEpoch"],
                               learning_rate=conf["KDR"]["lr"], device=device, alpha=conf["KDR"]["alpha"])

        graficar(kd_student_feature, test_ds, device, idx=idx, modelName="kd_student_feature", modo="canales")

    # =========================================== Attention based KD =====================================================
    if (conf["KDR"]["AKD"]):
        print("================ Entrenando attention_kd_student ================")
        kd_student_attention = MODEL_REGISTRY[sModel]["Student"]().to(device)

        akd_features = {}
        register_hooks_at(kd_student_attention, akd_features)

        akd_hist = train_attention_kd(teacher=teacher, student=kd_student_attention, t_feats=teacher_attentions,
                                            s_feats=akd_features, train_loader=train_loader, epochs=conf["KDR"]["sEpoch"],
                                            learning_rate=conf["KDR"]["lr"], device=device, alpha=conf["KDR"]["alpha"])

        graficar(kd_student_attention, test_ds, device, idx=idx, modelName="kd_student_attention", modo="canales")

    # Carga el historial del teacher / student solo si se han entrenado, si se han cargado no porque no están guardados en ningun lado por ahora
    historial = {}
    if conf["KDR"]["train"]:
        historial = {
            "Teacher": teacher_hist,
            "Student": student_hist
        }
    if conf["KDR"]["KD"]:
        historial["KD_Student"] = kd_hist
    if conf["KDR"]["FKD"]:
        historial["FKD_Student"] = fkd_hist
    if conf["KDR"]["AKD"]:
        historial["AKD_Student"] = akd_hist

    plot_training_curves(historial)
    guardar_training_curves(historial)