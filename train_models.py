import json
import statistics

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from Cuantizacion import cuantizar_estatica, cuantizar_qat, entrenar_qat
from utils import attention_transfer_loss, register_hook, register_hooks_at, evaluate_psnr

with open("config.json", "r", encoding="utf-8") as f:
    conf = json.load(f)

def train_basic(model, train_loader, val_loader, epochs, learning_rate, device, patience):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStoppingLoss(patience=patience)

    train_history = []
    val_history = []

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for X, Y in train_loader:
            # X: Señal ruidoso
            # Y: Señal limpia
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            outputs = model(X)

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

def train_kd(teacher, student, train_loader, val_loader, epochs, learning_rate, alpha, device, patience):
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    early_stopper = EarlyStoppingLoss(patience=patience)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher no tiene que aprender
    student.train()

    train_history = []
    val_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        student_running_loss = 0.0 
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_pred = teacher(X)
            student_pred = student(X)

            student_y_loss = mse_loss(student_pred, Y)
            teacher_student_loss = mse_loss(student_pred, teacher_pred)
            # Pérdida de RKD
            loss = alpha * teacher_student_loss + (1.0 - alpha) * student_y_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            student_running_loss += student_y_loss.item()

        epoch_loss = student_running_loss / len(train_loader)
        val_loss = evaluate(student, val_loader, device)

        train_history.append(epoch_loss)
        val_history.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} | Train_Loss: {epoch_loss:.8f} | Val_Loss: {val_loss:.8f}")

        if(conf["KDR"]["wandb"]):
            wandb.log({
                "epoch": epoch + 1,
                "mse": val_loss
            })

        if early_stopper.step(val_loss, student):
            print("Early stopping triggered!")
            break

    early_stopper.restore(student)

    return train_history, val_history

def train_fkd(teacher, student, t_features, s_features, train_loader, val_loader, epochs, learning_rate, alpha, beta, device, patience):

    mse_loss = nn.MSELoss()
    early_stopper = EarlyStoppingLoss(patience=patience + 8)

    teacher.to(device)
    student.to(device)
    teacher.eval()

    with torch.no_grad():
        dummy = next(iter(train_loader))[0][:1].to(device)
        teacher(dummy)
        t_ch = t_features["latent"].shape[1]
        student(dummy)
        s_ch = s_features["latent"].shape[1]

    bottleneck_proj = nn.Conv2d(t_ch, s_ch, kernel_size=1, bias=False).to(device)
    print(f"Linear bottleneck: {t_ch} a {s_ch} canales")

    # Warmup del bottleneck (student congelado)
    w_epochs = max(5, epochs // 50)

    # Congela el student
    for param in student.parameters():
        param.requires_grad = False

    warmup_optimizer = optim.Adam(
        bottleneck_proj.parameters(),
        lr=learning_rate * 20
    )

    print(f"\n Bottleneck warmup ({w_epochs} epochs)")
    for epoch in range(w_epochs):
        bottleneck_proj.train()
        running_kd = 0.0

        for X, _ in train_loader:
            X = X.to(device)
            warmup_optimizer.zero_grad()

            with torch.no_grad():
                teacher(X)
                t_latent = t_features["latent"].detach()
                student(X)
                s_latent = s_features["latent"].detach()  # fijo, no aprende aún

            t_latent_proj = bottleneck_proj(t_latent)
            kd_loss = cosine_kd_loss(s_latent, t_latent_proj)
            kd_loss.backward()
            warmup_optimizer.step()
            running_kd += kd_loss.item()

        print(f"Warmup {epoch+1}/{w_epochs} | KD_Loss: {running_kd/len(train_loader):.8f}")

    # Descongelar el student
    for param in student.parameters():
        param.requires_grad = True

    # Entrenamiento conjunto
    optimizer = optim.Adam([
        {"params": student.parameters(), "lr": learning_rate},
        {"params": bottleneck_proj.parameters(), "lr": learning_rate}
    ])

    train_history, val_history = [], []

    print(f"\n Entrenamiento conjunto ({epochs} epochs)")
    for epoch in range(epochs):
        student.train()
        bottleneck_proj.train()
        running_loss = 0.0
        student_running_loss = 0.0

        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_pred = teacher(X)
                t_latent = t_features["latent"].detach()

            student_pred = student(X)
            s_latent = s_features["latent"]
            student_y_loss = mse_loss(student_pred, Y)
            kd_loss = mse_loss(student_pred, teacher_pred)

            t_latent_proj = bottleneck_proj(t_latent)
            feature_loss = cosine_kd_loss(s_latent, t_latent_proj)
            # Pérdida de FKD
            loss = (1.0-alpha) * student_y_loss + alpha * kd_loss + beta * feature_loss

            loss.backward()
            optimizer.step()

            student_running_loss += student_y_loss.item()
            running_loss += loss.item()

        epoch_loss = student_running_loss / len(train_loader)
        val_loss = evaluate(student, val_loader, device)
        train_history.append(epoch_loss)
        val_history.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Train_Loss: {epoch_loss:.8f} | Val_Loss: {val_loss:.8f}")

        if (conf["KDR"]["wandb"]):
            wandb.log({
                "epoch": epoch + 1,
                "mse": val_loss
            })

        if early_stopper.step(val_loss, student):
            print("Early stopping triggered!")
            break

    early_stopper.restore(student)
    return train_history, val_history


def train_akd(teacher, student, t_attentions, s_attentions, train_loader, val_loader, epochs, learning_rate, alpha, device, patience, beta):
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    early_stopper = EarlyStoppingLoss(patience=patience)

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
                teacher_pred = teacher(X)
                t_enc1 = t_attentions["enc1"].detach()
                t_enc2 = t_attentions["enc2"].detach()
                t_bottle = t_attentions["bottleneck"].detach()

            student_pred = student(X)
            s_enc1 = s_attentions["enc1"]
            s_enc2 = s_attentions["enc2"]
            s_bottle = s_attentions["bottleneck"]

            out_loss = mse_loss(student_pred, Y)
            at_loss = (
                    attention_transfer_loss(s_enc1, t_enc1) +
                    attention_transfer_loss(s_enc2, t_enc2) +
                    attention_transfer_loss(s_bottle, t_bottle)
            )
            kd_loss = mse_loss(student_pred, teacher_pred)
            # Pérdida de AKD
            loss = (1.0-alpha) * out_loss + alpha * kd_loss + beta * at_loss

            loss.backward()
            optimizer.step()
            student_running_loss += out_loss.item()

        epoch_loss = student_running_loss / len(train_loader)
        val_loss = evaluate(student, val_loader, device)
        train_history.append(epoch_loss)
        val_history.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Train_Loss: {epoch_loss:.8f} | Val_Loss: {val_loss:.8f}")

        if (conf["KDR"]["wandb"]):
            wandb.log({
                "epoch": epoch + 1,
                "mse": val_loss
            })

        if early_stopper.step(val_loss, student):
            print("Early stopping triggered!")
            break

    early_stopper.restore(student)
    return train_history, val_history


def cosine_kd_loss(h_s, h_t):  # h_s, h_t: [B, C, H, W]
    B, C_s, H, W = h_s.shape
    _, C_t, _, _ = h_t.shape

    h_s = h_s.reshape(B, C_s, H*W)
    h_t = h_t.reshape(B, C_t, H*W)

    return 1 - F.cosine_similarity(h_s, h_t, dim=-1).mean()

# ============================================== Early stopping ========================================================

def evaluate(model, loader, device):
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            total_loss += mse(pred, Y).item()

    return total_loss / len(loader)

class EarlyStoppingLoss:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss + self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    # Función para cargar el mejor estado del modelo
    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# ========================================= K-Fold Cross Validation ====================================================
def run_with_kfold(train_fn, model_fn, load_fold_fn, device, batch, **kwargs,):
    n_folds = conf["Data"].get("n_folds", 4)
    quant = kwargs["cuant"]
    kwargs.pop("cuant", None)
    fold_mses = []
    fold_psnrs = []
    val_hists = []
    train_hists = []
    model = None

    for i in range(0, n_folds):
        device_fold = device
        kwargs_fold = dict(kwargs)
        print(f"\n Fold {i+1}/{n_folds}")
        student = model_fn().to(device_fold) # En cada fold se crea un nuevo student
        train_loader, val_loader = load_fold_fn(i, batch)

        if("t_features" in kwargs_fold): # Hookea las features del student en caso de existir las del teacher
            s_features = {}
            register_hook(student, conf["KDR"]["sModel"], s_features, "latent")
            kwargs_fold["s_features"] = s_features
        elif("t_attentions" in kwargs_fold): # Hookea los attention maps del student en caso de exisitir los del teacher
            s_attentions = {}
            register_hooks_at(student, s_attentions)
            kwargs_fold["s_attentions"] = s_attentions

        if(quant == "pre") or (quant == "both"): # Introduce los nodos de fake quantizarion en el student
            student = entrenar_qat(student, device_fold, val_loader)

        if("teacher" in kwargs_fold):
            train_hist, val_hist = train_fn(
                train_loader=train_loader, val_loader=val_loader, device=device_fold, student=student, **kwargs_fold
            )
        else:
            train_hist, val_hist = train_fn(
                train_loader=train_loader, val_loader=val_loader, device=device_fold, model=student, **kwargs_fold
            )

        train_hists.append(train_hist)
        val_hists.append(val_hist)

        if(quant == "post"): # Aplica PTQ y evalua el modelo cuantizado
            device_fold = torch.device("cpu")
            student = student.to(device_fold)
            student = cuantizar_estatica(student, device_fold, val_loader)
            fold_mse = evaluate(student, val_loader, device_fold)
        elif (quant == "pre") or (quant == "both"): # Aplica QAT y evalua el modelo cuantizado
            device_fold = torch.device("cpu")
            student = student.to(device_fold)
            if (quant == "both"): # Fine tunning usando teacher cuantizado
                teacherq = kwargs_fold["teacher"].to(device_fold)
                teacherq = cuantizar_estatica(teacherq, device_fold, val_loader)
                kwargs_fold["teacher"] = teacherq
                kwargs_fold["epochs"] = 100
                train_fn( train_loader=train_loader, val_loader=val_loader, device=device_fold, student=student, **kwargs_fold)
            student = cuantizar_qat(student)
            fold_mse = evaluate(student, val_loader, device_fold)
        else:
            fold_mse = val_hist[-1]

        model = student # Devuelve el modelo para analizar luego su peso
        fold_mses.append(fold_mse)
        fold_psnr = evaluate_psnr(student, val_loader, device_fold)
        fold_psnrs.append(fold_psnr)
        print(f"Fold {i+1} | MSE: {fold_mse:.6f} | PSNR: {fold_psnr:.6f}")

        if conf["KDR"].get("wandb"):
            wandb.log({f"mse_fold_{i}": fold_mse, f"psnr_fold_{i}": fold_psnr})

    mse_mean = statistics.mean(fold_mses)
    mse_std  = statistics.stdev(fold_mses) if n_folds > 1 else 0.0
    psnr_mean = statistics.mean(fold_psnrs)
    psnr_std = statistics.stdev(fold_psnrs) if n_folds > 1 else 0.0
    print(f"\nK-Fold: Mean: {mse_mean:.6f} | Std: {mse_std:.6f} | PSNR: {psnr_mean:.6f} | PSNR Std: {psnr_std:.6f}")

    if conf["KDR"].get("wandb"):
        wandb.log({"mse_mean": mse_mean, "mse_std": mse_std, "psnr_mean": psnr_mean, "psnr_std": psnr_std})

    results = {
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "fold_mses": fold_mses,
        "fold_psnrs": fold_psnrs,
        "train_hists": train_hists,
        "val_hists": val_hists,
    }

    return results, model