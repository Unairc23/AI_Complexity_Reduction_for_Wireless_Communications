import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
# Para instalar torch junto con cuda:
# "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126"

# Check if the current `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# is available, and if not, use the CPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

# Below we are preprocessing data for CIFAR-10. We use an arbitrary batch size of 128.
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading the CIFAR-10 dataset:
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

#Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# ========================== MODELOS ============================
# Deeper neural network class to be used as teacher:
class DeepNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 2
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 4 (opcional, si tu GPU lo permite)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Clasificador completamente conectado
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Lightweight neural network class to be used as student:
class LightNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ========================== ENTRENAMIENTO Y EVALUACION ============================
def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: A collection of batch_size images
            # labels: A vector of dimensionality batch_size with integers denoting class of each image
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
            # labels: The actual labels of the images. Vector of dimensionality batch_size
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

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
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# ========================== ENTRENAMIENTO KD ============================
def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, alpha, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

def load_model(model_class, path, device):
    """Carga un modelo si el archivo existe, o devuelve uno nuevo."""
    model = model_class().to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f" Modelo cargado desde {path}")
    else:
        print(f" No se encontró el archivo {path}, se inicializa un modelo nuevo.")
    return model

if __name__ == "__main__":
    # Creo que esta linea solo es encesaria en windows, linux hace un fork
    torch.multiprocessing.freeze_support()

    print(sys.argv)

    if len(sys.argv)>1 and sys.argv[1] == "full":
        torch.manual_seed(42)
        nn_deep = DeepNN(num_classes=10).to(device)
        train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)

        torch.manual_seed(42)
        nn_light = LightNN(num_classes=10).to(device)
        torch.manual_seed(42)
        total_params_deep = "{:,}".format(sum(p.numel() for p in nn_deep.parameters()))
        print(f"DeepNN parameters: {total_params_deep}")
        total_params_light = "{:,}".format(sum(p.numel() for p in nn_light.parameters()))
        print(f"LightNN parameters: {total_params_light}")
        train(nn_light, train_loader, epochs=10, learning_rate=0.001, device=device)

        torch.save(nn_deep.state_dict(), "model/DeepNN.pth")
        torch.save(nn_light.state_dict(), "model/student_no_kd.pth")

    else:
        nn_deep = load_model(DeepNN, "model/DeepNN.pth", device)
        nn_light = load_model(LightNN, "model/student_no_kd.pth", device)

    test_accuracy_deep = test(nn_deep, test_loader, device)
    test_accuracy_light_ce = test(nn_light, test_loader, device)
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    print(f"Student accuracy: {test_accuracy_light_ce:.2f}%")

    # --- Experimento con distintos pesos alpha ---
    alphas = [0.75]
    Ts = [3, 4, 5, 6, 7, 8, 9, 10]
    results = {}

    for alpha in alphas:
        results[alpha] = {}
        for T in Ts:
            new_nn_light = LightNN(num_classes=10).to(device)
            print(f"\n=== Training student with alpha={alpha} / T={T} ===")
            train_knowledge_distillation(
                teacher=nn_deep,
                student=new_nn_light.to(device),
                train_loader=train_loader,
                epochs=10,
                learning_rate=0.001,
                T=T,
                alpha=0.25,
                device=device
            )
            test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device)
            results[alpha][T] = test_accuracy_light_ce_and_kd
            print(f"Student accuracy (alpha={alpha} / T={T}): {test_accuracy_light_ce_and_kd:.2f}%")

    print("\n=== Summary ===")
    for alpha, temp_dict in results.items():
        for T, acc in temp_dict.items():
            print(f"α={alpha:.2f} / T={T} → Accuracy: {acc:.2f}%")

