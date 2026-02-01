import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import random_split, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import random, numpy as np

from modelos import DeepNN, LightNN, LightNN_Adaptada, DeepNN_Adaptada, load_resnet, DnCNN

with open("config.json", "r", encoding="utf-8") as f:
    conf = json.load(f)

MODEL_REGISTRY = {
    "resnet18": lambda: load_resnet(),
    "deep": lambda: DeepNN(num_classes=10),
    "deep_adaptada": lambda: DeepNN_Adaptada(num_classes=10),
    "light": lambda: LightNN(num_classes=10),
    "light_adaptada": lambda: LightNN_Adaptada(num_classes=10),
    "DnCNN": lambda: DnCNN(depth=conf["KDR"]["sDepth"])
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

torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================== Crear dataset ========================================================
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

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

# ====================================== Entrenamiento de los modelos =================================================
def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for X, Y in train_loader:
            # X: Señal ruidoso
            # Y: Señal limpia/original
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            outputs = model(X)

            # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
            # X: The actual images. Vector of dimensionality batch_size
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f}")

def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, teacher_threshold, alpha, device):
    #Todo: Ahora mismo el teacher calcula el mse de todo el batch, no solo de la prediccion actual
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

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

            teacher_y_loss = mse_loss(teacher_pred, Y)
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
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f}")

# ============================================== Carga y Parser ========================================================
def load_model(model, path, device):
    """Carga un modelo si el archivo existe, o devuelve uno nuevo."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f" Modelo cargado desde {path}")
    else:
        print(f" No se encontró el archivo {path}, se inicializa un modelo nuevo.")
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Training with Knowledge Distillation")

    parser.add_argument(
        "--teacher",
        type=str,
        default="resnet18",
        choices=["resnet18", "deep", "deep_adaptada", "DnCNN"],
        help="Teacher model"
    )

    parser.add_argument(
        "--student",
        type=str,
        default="light_adaptada",
        choices=["light", "light_adaptada", "DnCNN"],
        help="Student model"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["load", "full"],
        default="load",
        help="Training mode: load, full"
    )

    return parser.parse_args()

# ==========================================Representar graficamente====================================================
def graficar(model, dataset, device, idx=0):
    model.eval()

    x, y = dataset[idx]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        y_hat = model(x)

    # Quitar batch y canal
    x = x.squeeze().cpu().numpy()      # (128,128)
    y = y.squeeze().cpu().numpy()      # (128,128)
    y_hat = y_hat.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(x, cmap="viridis")
    axs[0].set_title("Entrada ruidosa")

    axs[1].imshow(y, cmap="viridis")
    axs[1].set_title("Objetivo (limpia)")

    axs[2].imshow(y_hat, cmap="viridis")
    axs[2].set_title("Reconstrucción")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Creo que esta linea solo es encesaria en windows, linux hace un fork
    torch.multiprocessing.freeze_support()

    args = parse_args()
    print(args)

    teacher = MODEL_REGISTRY[args.teacher]().to(device)
    student = MODEL_REGISTRY[args.student]().to(device)

    if args.mode == "full":
        if args.teacher != "resnet18": # Los modelos resnet usados ya están preentrenados con CIFAR10
            train(model=teacher, train_loader=train_loader, epochs=conf["KDR"]["tEpoch"], learning_rate=["KDR"]["lr"],
                  device=device)
        train(model=student, train_loader=train_loader, epochs=conf["KDR"]["tEpoch"], learning_rate=["KDR"]["lr"],
              device=device)

        torch.save(teacher.state_dict(), f"model/{args.teacher}.pth")
        torch.save(student.state_dict(), f"model/{args.student}.pth")

    else:
        teacher = load_model(teacher, path=f"model/{args.teacher}.pth", device=device)
        student = load_model(student, path=f"model/{args.student}.pth", device=device)

    teacher_params = "{:,}".format(sum(p.numel() for p in teacher.parameters()))
    print(f"Teacher Params: {teacher_params}")
    student_params = "{:,}".format(sum(p.numel() for p in student.parameters()))
    print(f"Student Params: {student_params}")

    graficar(student, test_ds, device, idx=5)

    kd_student = MODEL_REGISTRY[args.student]().to(device)
    train_knowledge_distillation(teacher=teacher, student=kd_student, train_loader=train_loader,
                                 epochs=conf["KDR"]["tEpoch"], learning_rate=["KDR"]["lr"], device=device,
                                 teacher_threshold=0.001, alpha=conf["KDR"]["alpha"])

    graficar(kd_student, test_ds, device, idx=5)