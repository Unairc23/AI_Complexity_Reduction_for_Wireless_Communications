import json
import statistics

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from utils import attention_transfer_loss, register_hook, register_hooks_at

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
            # Y: Señal limpia/original
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            outputs = model(X)

            if (conf["Data"]["Unica"]):
                Y = Y[:, :, 64, :]
                outputs = outputs[:, :, 64, :]

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
    teacher.eval()  # Teacher sno tiene que aprender
    student.train()

    train_history = []
    val_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        student_running_loss = 0.0 # Usado simplemente para ver la diferencia train/val
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_pred = teacher(X)
            student_pred = student(X)

            if (conf["Data"]["Unica"]):
                Y = Y[:, :, 64, :]
                teacher_pred = teacher_pred[:, :, 64, :]
                student_pred = student_pred[:, :, 64, :]

            student_y_loss = mse_loss(student_pred, Y)
            teacher_student_loss = mse_loss(student_pred, teacher_pred)
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
                "mse": epoch_loss
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
    print(f"Linear bottleneck: {t_ch} → {s_ch} canales")

    # Warmup del bottleneck (student congelado)
    WARMUP_EPOCHS = max(5, epochs // 50)

    # Congela el student
    for param in student.parameters():
        param.requires_grad = False

    warmup_optimizer = optim.Adam(
        bottleneck_proj.parameters(),
        lr=learning_rate * 20
    )

    print(f"\n── Bottleneck warmup ({WARMUP_EPOCHS} epochs, student congelado) ──")
    for epoch in range(WARMUP_EPOCHS):
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

        print(f"  Warmup {epoch+1}/{WARMUP_EPOCHS} | KD_Loss: {running_kd/len(train_loader):.8f}")

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

            if conf["Data"]["Unica"]:
                Y = Y[:, :, 64, :]
                student_pred = student_pred[:, :, 64, :]

            student_y_loss = mse_loss(student_pred, Y)
            kd_loss = mse_loss(student_pred, teacher_pred)

            t_latent_proj = bottleneck_proj(t_latent)
            feature_loss = cosine_kd_loss(s_latent, t_latent_proj)
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
                "mse": epoch_loss
            })

        if early_stopper.step(val_loss, student):
            print("Early stopping triggered!")
            break

    early_stopper.restore(student)
    return train_history, val_history


def train_akd(teacher, student, t_attentions, s_attentions, train_loader, val_loader, epochs, learning_rate, alpha, device, patience):
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
                teacher(X)
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
            #print(f"Loss: {out_loss:.8f} / ATLoss: {at_loss:.8f}") # AtLoss es dos ordenes de magnitud mas pequeño
            loss = out_loss + alpha * at_loss

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
                "mse": epoch_loss
            })

        if early_stopper.step(val_loss, student):
            print("Early stopping triggered!")
            break

    early_stopper.restore(student)
    return train_history, val_history


def cosine_kd_loss(h_s, h_t):  # h_s, h_t: [B, C, H, W]
    B, C_s, H, W = h_s.shape
    _, C_t, _, _ = h_t.shape

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

# ========================================= K-Fold Cross Validation ====================================================

def run_with_kfold(train_fn, model_fn, load_fold_fn, device, batch, **kwargs,):
    n_folds = conf["Data"].get("n_folds", 4)
    fold_mses = []

    for i in range(n_folds):
        print(f"\n Fold {i+1}/{n_folds}")
        student = model_fn().to(device)
        train_loader, val_loader = load_fold_fn(i, batch)

        if("t_features" in kwargs):
            s_features = {}
            register_hook(student, conf["KDR"]["sModel"], s_features, "latent", conf["KDR"]["f_layers"]["s_layers"])
            kwargs["s_features"] = s_features
        elif("t_attentons" in kwargs):
            s_attentons = {}
            register_hooks_at(student, s_attentons)
            kwargs["s_attentons"] = s_attentons

        _, val_hist = train_fn(
            train_loader=train_loader, val_loader=val_loader, device=device, student=student, **kwargs
        )
        fold_mse = val_hist[-1]
        fold_mses.append(fold_mse)
        print(f"Fold {i+1} | MSE: {fold_mse:.6f}")

        if conf["KDR"].get("wandb"):
            wandb.log({f"mse_fold_{i}": fold_mse})

    mse_mean = statistics.mean(fold_mses)
    mse_std  = statistics.stdev(fold_mses) if n_folds > 1 else 0.0
    print(f"\nK-Fold → Mean: {mse_mean:.6f} | Std: {mse_std:.6f}")

    if conf["KDR"].get("wandb"):
        wandb.log({"mse_mean": mse_mean, "mse_std": mse_std})

    return mse_mean, mse_std