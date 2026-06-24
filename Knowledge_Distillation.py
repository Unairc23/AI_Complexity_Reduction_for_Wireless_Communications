import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import argparse
import random, numpy as np
import wandb
from modelos import DeepNN, LightNN, LightNN_Adaptada, DeepNN_Adaptada, load_resnet

MODEL_REGISTRY = {
    "resnet18": lambda: load_resnet(),
    "deep": lambda: DeepNN(num_classes=10),
    "deep_adaptada": lambda: DeepNN_Adaptada(num_classes=10),
    "light": lambda: LightNN(num_classes=10),
    "light_adaptada": lambda: LightNN_Adaptada(num_classes=10),
}

# La estructura general del código está basada en https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
# Los modelos _Adaptada están hechos siguiendo el paper "Simplified Knowledge Distillation for Deep Neural Networks"

# Para instalar torch con cuda:
# "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126"

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

# Crear val para early stopping
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

# Dataloaders
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

class EarlyStoppingAcc:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_acc = -float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_acc, model):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# ========================== ENTRENAMIENTO Y EVALUACION ============================
def train(model, train_loader, val_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    early_stopper = EarlyStoppingAcc(patience=5)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_acc = evaluate_accuracy(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if early_stopper.step(val_acc, model):
            print("Early stopping triggered!")
            break

    early_stopper.restore(model)

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, pred = torch.max(logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    print(f"Accuracy: {100 * correct / total}")
    return correct / total


# ========================== ENTRENAMIENTO KD ============================
def train_knowledge_distillation(teacher, student, train_loader, val_loader, epochs, learning_rate, T, alpha, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.train()

    early_stopper = EarlyStoppingAcc(patience=3)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)

            # Loss KD estándar usando KLDiv entre log_softmax(student/T) y softmax(teacher/T).
            log_p_student = F.log_softmax(student_logits / T, dim=1)
            q_teacher = F.softmax(teacher_logits / T, dim=1)

            kd_loss = F.kl_div(log_p_student, q_teacher, reduction='batchmean') * (T * T)
            label_loss = ce_loss(student_logits, labels)

            loss = alpha * kd_loss + (1.0 - alpha) * label_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        val_acc = evaluate_accuracy(student, val_loader, device)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        if early_stopper.step(val_acc, student):
            print(f"No ha habido mejoras mayores a {early_stopper.min_delta} en {early_stopper.patience} epochs")
            break

    early_stopper.restore(student)

def train_kd_wandb(teacher):
    wandb.init()
    config = wandb.config

    student = MODEL_REGISTRY[args.student]().to(device)

    train_knowledge_distillation(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        T=config.T,
        alpha=config.alpha,
        device=device
    )

    # Evaluación del modelo student
    acc = test(student, test_loader, device)

    # Reportar métrica al sweep
    wandb.log({"accuracy": acc})

def load_model(model, path, device):
    # Carga un modelo si el archivo existe, o devuelve uno nuevo.
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
        choices=["resnet18", "deep", "deep_adaptada"],
        help="Teacher model"
    )

    parser.add_argument(
        "--student",
        type=str,
        default="light_adaptada",
        choices=["light", "light_adaptada"],
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


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    args = parse_args()
    print(args)

    teacher = MODEL_REGISTRY[args.teacher]().to(device)
    student = MODEL_REGISTRY[args.student]().to(device)

    if args.mode == "full":
        if args.teacher != "resnet18": # Los modelos resnet usados ya están preentrenados con CIFAR10
            train(teacher, train_loader, val_loader, epochs=50, learning_rate=0.01, device=device)
        train(student, train_loader, val_loader, epochs=10, learning_rate=0.01, device=device)

        torch.save(teacher.state_dict(), f"model/{args.teacher}.pth")
        torch.save(student.state_dict(), f"model/{args.student}.pth")

        teacher_size = os.path.getsize(f"model/{args.teacher}.pth") / 1024 ** 2
        print(f"Teacher Size: {teacher_size} \n")
        student_size = os.path.getsize(f"model/{args.student}.pth") / 1024 ** 2
        print(f"Student Size: {student_size} \n")

        print(f"Diferencia de tamaño: {teacher_size / student_size:.2f}x ({teacher_size:.2f}MB -> {student_size:.2f}MB)")

    else:
        teacher = load_model(teacher, path=f"model/{args.teacher}.pth", device=device)
        student = load_model(student, path=f"model/{args.student}.pth", device=device)

    teacher_params = "{:,}".format(sum(p.numel() for p in teacher.parameters()))
    print(f"Teacher Params: {teacher_params}")
    student_params = "{:,}".format(sum(p.numel() for p in student.parameters()))
    print(f"Student Params: {student_params}")
    test_accuracy_deep = test(teacher, test_loader, device)
    test_accuracy_light_ce = test(student, test_loader, device)
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    print(f"Student accuracy: {test_accuracy_light_ce:.2f}%")

# ============================ Experimento con distintos pesos alpha ============================
    alphas = [0.1, 0.25, 0.5, 0.75, 0.9]
    Ts = [t for t in range(1,20)]
    results = {}

    for alpha in alphas:
        results[alpha] = {}
        for T in Ts:
            new_student = MODEL_REGISTRY[args.student]()
            print(f"\n=== Training student with alpha={alpha} / T={T} ===")
            train_knowledge_distillation(
                teacher=teacher,
                student=new_student.to(device),
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=10,
                learning_rate=0.01,
                T=T,
                alpha=alpha,
                device=device
            )
            test_accuracy_light_ce_and_kd = test(new_student, test_loader, device)
            results[alpha][T] = test_accuracy_light_ce_and_kd
            print(f"Student accuracy (alpha={alpha} / T={T}): {test_accuracy_light_ce_and_kd:.2f}%")

    print("\n=== Summary ===")
    for alpha, temp_dict in results.items():
        for T, acc in temp_dict.items():
            print(f"α={alpha:.2f} / T={T} → Accuracy: {acc:.2f}%")

# ============================ WANDB ============================
#
#     sweep_config = {
#         'method': 'bayes',  # bayesian, random, grid
#         'metric': {
#             'name': 'accuracy',
#             'goal': 'maximize'
#         },
#         'parameters': {
#             'alpha': {
#                 'min': 0.6,
#                 'max': 0.8
#             },
#             'T': {
#                 'min': 1,
#                 'max': 20
#             },
#             'learning_rate': {
#                 'values': [0.001]
#             },
#             'epochs': {
#                 'values': [10]
#             }
#         }
#     }
#
#     sweep_id = wandb.sweep(sweep_config, project="Adaptada_Res_KD")
#     wandb.agent(sweep_id, lambda:train_kd_wandb(teacher))
