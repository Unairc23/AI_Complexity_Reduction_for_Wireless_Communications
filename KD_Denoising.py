import os
import wandb
import random
from torch.utils.data import Dataset
from pathlib import Path

from modelos import DnCNN, UNetDenoiser
from utils import *
from Cuantizacion import *
from train_models import train_basic, train_akd, train_kd, train_fkd, run_with_kfold, evaluate

with open("config.json", "r", encoding="utf-8") as f:
    conf = json.load(f)

fechaHora = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = f"results/{fechaHora}"
os.makedirs(results_dir)

MODEL_REGISTRY = {
    "DnCNN": {
        "Student": lambda: DnCNN(depth=conf["Model"]["sDepth"]),
        "Teacher": lambda: DnCNN(depth=conf["Model"]["tDepth"])
    },
    "UNet": {
        "Student": lambda: UNetDenoiser(in_channels=2, base_channels=conf["Model"]["sDepth"]),
        "Teacher": lambda: UNetDenoiser(in_channels=2, base_channels=conf["Model"]["tDepth"])
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
# Para instalar torch con cuda:
# "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126"

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
batch_size = conf["Model"]["batch_size"]

X_train = np.load(conf["KDR"]["X"].replace(".npy", "_Train.npy"))
X_test = np.load(conf["KDR"]["X"].replace(".npy", "_Test.npy"))
X_val = np.load(conf["KDR"]["X"].replace(".npy", "_Val.npy"))
Y_train = np.load(conf["KDR"]["Y"].replace(".npy", "_Train.npy"))
Y_test = np.load(conf["KDR"]["Y"].replace(".npy", "_Test.npy"))
Y_val = np.load(conf["KDR"]["Y"].replace(".npy", "_Val.npy"))

train_ds = NPYDataset(X_train, Y_train)
test_ds = NPYDataset(X_test, Y_test)
val_ds = NPYDataset(X_val, Y_val)

print(f"Tamaño Train: {len(train_ds)}")
print(f"Tamaño Test: {len(test_ds)}")
print(f"Tamaño Val: {len(val_ds)}")

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

def cargar_fold(i, batch=16):
    dset = conf["KDR"]["Y"]
    dset_ruido = conf["KDR"]["X"]
    dset_name = Path(dset).stem
    dset_ruido_name = Path(dset_ruido).stem
    X_train = np.load(f"data/folds/est/{dset_ruido_name}_{i}_Train.npy")
    X_val = np.load(f"data/folds/est/{dset_ruido_name}_{i}_Val.npy")
    Y_train = np.load(f"data/folds/real/{dset_name}_{i}_Train.npy")
    Y_val = np.load(f"data/folds/real/{dset_name}_{i}_Val.npy")

    train_ds = NPYDataset(X_train, Y_train)
    val_ds = NPYDataset(X_val, Y_val)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)

    return train_loader, val_loader

def train_kd_wandb(teacher):
    wandb.init(
        # Estos parametros se añaden aqui unicamente para tener luego registro de ello en wandb
        config={
            'model': conf["KDR"]["sModel"],
            'sint': conf["Data"]["Sint"],
            'mixedSNR': conf["Data"]["MixedSNR"],
            'SNR': conf["Data"]["Snr_db"],
            'tSize': conf["Model"]["tDepth"],
            'sSize': conf["Model"]["sDepth"],
        }
    )
    cfg = wandb.config

    run_with_kfold(train_fn=train_fkd, model_fn=MODEL_REGISTRY[sModel]["Student"], load_fold_fn=cargar_fold,
                   device=device, batch=cfg.batch_size, teacher=teacher, alpha=cfg.alpha, patience=cfg.patience,
                   epochs=cfg.epochs, learning_rate=cfg.learning_rate, beta=cfg.beta, t_features=teacher_features, cuant="none")

# ============================================== CARGA MODELOS ========================================================
def load_model(model, path, device):
    # Carga un modelo si el archivo existe, o devuelve uno nuevo.
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
    t_tamaño = conf["Model"]["tDepth"]
    sModel = conf["KDR"]["sModel"]
    s_tamaño = conf["Model"]["sDepth"]
    snr = conf["Data"]["Snr_db"]
    snr_med = np.median(snr).astype(int)
    print(snr_med)

    teacher_hist = None
    student_hist = None
    kd_hist = None
    fkd_hist = None
    akd_hist = None

    teacher = MODEL_REGISTRY[tModel]["Teacher"]().to(device)
    student = MODEL_REGISTRY[sModel]["Student"]().to(device)

    teacher_features = {}
    teacher_attentions = {}

    register_hook(teacher, tModel, teacher_features, "latent")
    register_hooks_at(teacher, teacher_attentions)

    if conf["KDR"]["t_train"]:
        print("\n================ Entrenando teacher ================")
        if not(conf["Data"]["Kfold"]):
            teacher_hist = train_basic(model=teacher, train_loader=train_loader, val_loader=val_loader,
                                       epochs=conf["Model"]["tEpoch"], learning_rate=conf["Model"]["lr"],
                                       device=device, patience=conf["Model"]["patience"])
            torch.save(teacher.state_dict(), f"model/{tModel}_{t_tamaño}l_{snr_med}snr.pth")
            with open(f"{results_dir}/hist_teacher.json", "w") as f:
                json.dump(teacher_hist, f)
        else:
            results, _ = run_with_kfold(train_fn=train_basic, model_fn=MODEL_REGISTRY[sModel]["Teacher"], load_fold_fn=cargar_fold,
                           device=device, batch=batch_size,learning_rate=conf["Model"]["lr"],
                           patience=conf["Model"]["patience"], epochs=conf["Model"]["sEpoch"], cuant="none")
            with open(f"{results_dir}/kfold_teacher.json", "w") as f:
                json.dump(results, f)
    else:
        teacher = load_model(teacher, path=f"model/{tModel}_{t_tamaño}l_{snr_med}snr.pth", device=device)

    if conf["KDR"]["s_train"]:
        print("\n================ Entrenando no_KD_student ================")
        if not(conf["Data"]["Kfold"]):
            student_hist = train_basic(model=student, train_loader=train_loader, val_loader=val_loader,
                                 epochs=conf["Model"]["sEpoch"], learning_rate=conf["Model"]["lr"],
                                 device=device, patience=conf["Model"]["patience"])
            torch.save(student.state_dict(), f"model/{sModel}_{s_tamaño}l_{snr_med}snr.pth")
            with open(f"{results_dir}/hist_baseline.json", "w") as f:
                json.dump(student_hist, f)
        else:
            results, _ = run_with_kfold(train_fn=train_basic, model_fn=MODEL_REGISTRY[sModel]["Student"], load_fold_fn=cargar_fold,
                           device=device, batch=batch_size, learning_rate=conf["Model"]["lr"],
                           patience=conf["Model"]["patience"], epochs=conf["Model"]["sEpoch"], cuant="none")
            with open(f"{results_dir}/kfold_baseline.json", "w") as f:
                json.dump(results, f)
    else:
        student = load_model(student, path=f"model/{sModel}_{s_tamaño}l_{snr_med}snr.pth", device=device)

    teacher.eval()
    x, y = test_ds[0]
    x_in = x.unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred = teacher(x_in)
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    y_pred = y_pred.squeeze(0).cpu().numpy()

    X = x[0, 64, :] + 1j * x[1, 64, :]
    Y = y[0, 64, :] + 1j * y[1, 64, :]
    Y_pred = y_pred[0, 64, :] + 1j * y_pred[1, 64, :]

    fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    ax[0].plot(np.abs(X), label="Señal con ruido")
    ax[0].plot(np.abs(Y), label="Señal limpia")
    ax[0].legend()
    ax[0].set_xlabel("Retardo (τ)")
    ax[0].set_ylabel("Magnitud")

    ax[1].plot(np.abs(Y_pred), label="Señal tras denoising")
    ax[1].plot(np.abs(Y), label="Señal limpia")
    ax[1].legend()
    ax[1].set_xlabel("Retardo (τ)")
    ax[1].set_ylabel("Magnitud")
    fig.suptitle("Denoising")
    plt.show()

    # =========================================== Comparar modelos======================================================

    # Comparar tamaño teacher / modelo sin destilar
    teacher_params = "{:,}".format(sum(p.numel() for p in teacher.parameters()))
    teacher_size = os.path.getsize(f"model/{tModel}_{t_tamaño}l_{snr_med}snr.pth") / 1024 ** 2
    print(f"\nTeacher Params: {teacher_params}")
    print(f"Teacher Size: {teacher_size} \n")

    student_params = "{:,}".format(sum(p.numel() for p in student.parameters()))
    student_size = os.path.getsize(f"model/{sModel}_{s_tamaño}l_{snr_med}snr.pth") / 1024 ** 2
    print(f"Student Params: {student_params}")
    print(f"Student Size: {student_size} \n")

    print(f"Diferencia de tamaño: {teacher_size / student_size:.2f}x ({teacher_size:.2f}MB -> {student_size:.2f}MB)")

    # Comparacion de resultados entre teacher y modelo sin destilar:
    idx = int(conf["KDR"].get("plot_idx", 0))
    idx = max(0, min(idx, len(test_ds) - 1))
    graficar(student, test_ds, device, idx=idx, modelName="no_kd_student", modo="canales")
    graficar(teacher, test_ds, device, idx=idx, modelName="teacher", modo="canales")

    snr = []
    mseT = evaluate(teacher, test_loader, device)
    psnrT = evaluate_psnr(teacher, test_loader, device)
    mean_latT, std_latT, mean_per_sampleT = medir_latencia_gpu(teacher, test_loader, device)

    mseS = evaluate(student, test_loader, device)
    psnrS = evaluate_psnr(student, test_loader, device)
    mean_latS, std_latS, mean_per_sampleS = medir_latencia_gpu(student, test_loader, device)
    for x,y in test_loader: # For simple para comprobar que el ruido está correctamente aplicado en el dataset
        snr_db = calcular_ruido(señal_ruido_norm=x.numpy(), señal_norm=y.numpy())
        snr.append(snr_db)
    print(f"MSE teacher: {mseT:.8f}")
    print(f"MSE student: {mseS:.8f}")
    print(f"PSNR teacher: {psnrT:.8f}")
    print(f"PSNR student: {psnrS:.8f}")
    print(f"Latencia teacher: {mean_per_sampleT:.8f}ms")
    print(f"Latencia student: {mean_per_sampleS:.8f}ms")
    print(f"SNR medio del dataset de test: {np.mean(snr):.2f} dB")

    graficar_attention_maps(teacher, test_ds, device, idx=idx, modelName="Teacher")
    graficar_attention_maps(student, test_ds, device, idx=idx, modelName="Student")

    results_baseline = {
        "teacher": {
            "MSE": mseT,
            "PSNR": psnrT,
            "Latencia": mean_per_sampleT,
            "Size_MB": teacher_size,
            "Params": teacher_params
        },
        "baseline": {
            "MSE": mseS,
            "PSNR": psnrS,
            "Latencia": mean_per_sampleS,
            "Size_MB": student_size,
            "Params": student_params
        }
    }

    with open(f"{results_dir}/baseline_results.json", "w") as f:
        json.dump(results_baseline, f)

    # ============================================== KD clasica ========================================================
    if (conf["KDR"]["KD"]):
        print("\n================ Entrenando kd_student ================")
        kd_student = MODEL_REGISTRY[sModel]["Student"]().to(device)

        if not (conf["Data"]["Kfold"]):
            kd_hist = train_kd(teacher=teacher, student=kd_student, train_loader=train_loader, val_loader=val_loader,
                               epochs=conf["Model"]["sEpoch"], learning_rate=conf["Model"]["lr"], device=device,
                               alpha=0.7755, patience=conf["Model"]["patience"])
            with open(f"{results_dir}/hist_RKD.json", "w") as f:
                json.dump(kd_hist, f)
        else:
            results, _ = run_with_kfold(train_fn=train_kd, model_fn=MODEL_REGISTRY[sModel]["Student"], load_fold_fn=cargar_fold,
                           device=device, batch=batch_size, teacher=teacher, alpha=conf["KDR"]["alpha"],
                           patience=conf["Model"]["patience"], epochs=conf["Model"]["sEpoch"],
                           learning_rate=conf["Model"]["lr"])
            with open(f"{results_dir}/kfold_RKD.json", "w") as f:
                json.dump(results, f)

        # Comparacion entre teacher y modelo destilado
        torch.save(kd_student.state_dict(), f"model/kd_{sModel}_{s_tamaño}l_{snr_med}snr.pth")
        graficar(kd_student, test_ds, device, idx=idx, modelName="kd_student", modo="canales")

    # =========================================== Feature based KD =====================================================
    if (conf["KDR"]["FKD"]):
        print("\n================ Entrenando feature_kd_student ================")
        kd_student_feature = MODEL_REGISTRY[sModel]["Student"]().to(device)

        fkd_features = {}
        register_hook(kd_student_feature, sModel, fkd_features, "latent")

        if not (conf["Data"]["Kfold"]):
            fkd_hist = train_fkd(teacher=teacher, student=kd_student_feature,
                                 t_features=teacher_features, s_features=fkd_features,
                                 train_loader=train_loader, val_loader=val_loader, epochs=conf["Model"]["sEpoch"],
                                 learning_rate=conf["Model"]["lr"], device=device, alpha=0.6997,
                                 patience=conf["Model"]["patience"], beta=0.5227)
            with open(f"{results_dir}/hist_FKD.json", "w") as f:
                json.dump(fkd_hist, f)
        else:
            results, _ = run_with_kfold(train_fn=train_fkd, model_fn=MODEL_REGISTRY[sModel]["Student"], load_fold_fn=cargar_fold,
                           device=device, batch=batch_size, teacher=teacher, alpha=conf["KDR"]["alpha"],
                           patience=conf["Model"]["patience"], epochs=conf["Model"]["sEpoch"],
                           learning_rate=conf["Model"]["lr"], beta=conf["KDR"]["beta"], t_features=teacher_features)
            with open(f"{results_dir}/kfold_FKD.json", "w") as f:
                json.dump(results, f)

        torch.save(kd_student_feature.state_dict(), f"model/fkd_{sModel}_{s_tamaño}l_{snr_med}snr.pth")
        graficar(kd_student_feature, test_ds, device, idx=idx, modelName="kd_student_feature", modo="canales")

    # =========================================== Attention based KD =====================================================
    if (conf["KDR"]["AKD"]):
        print("\n================ Entrenando attention_kd_student ================")
        kd_student_attention = MODEL_REGISTRY[sModel]["Student"]().to(device)

        akd_features = {}
        register_hooks_at(kd_student_attention, akd_features)

        if not (conf["Data"]["Kfold"]):
            akd_hist = train_akd(teacher=teacher, student=kd_student_attention,
                                 t_attentions=teacher_attentions, s_attentions=akd_features,
                                 train_loader=train_loader, val_loader=val_loader, epochs=conf["Model"]["sEpoch"],
                                 learning_rate=conf["Model"]["lr"], device=device, alpha=conf["KDR"]["alpha"],
                                 patience=conf["Model"]["patience"], beta=conf["KDR"]["beta"])
            with open(f"{results_dir}/hist_AKD.json", "w") as f:
                json.dump(akd_hist, f)
        else:
            results, _ = run_with_kfold(train_fn=train_akd, model_fn=MODEL_REGISTRY[sModel]["Student"], load_fold_fn=cargar_fold,
                           device=device, batch=batch_size, teacher=teacher, alpha=0.649,
                           patience=conf["Model"]["patience"], epochs=conf["Model"]["sEpoch"],
                           learning_rate=conf["Model"]["lr"], beta=0.6246, t_attentions=teacher_attentions)
            with open(f"{results_dir}/kfold_AKD.json", "w") as f:
                json.dump(results, f)

        torch.save(kd_student_attention.state_dict(), f"model/akd_{sModel}_{s_tamaño}l_{snr_med}snr.pth")
        graficar(kd_student_attention, test_ds, device, idx=idx, modelName="kd_student_attention", modo="canales")

    with open(f"{results_dir}/config.json", "w") as f:
        json.dump(conf, f)

# ========================================== HIPERPARÁMETROS WANDB =====================================================
    if (conf["KDR"]["wandb"]):
        with open("config_wandb.json", "r", encoding="utf-8") as f:
            conf_wandb = json.load(f)

        sweep_config = conf_wandb
        sweep_id = wandb.sweep(sweep_config, project="Denoising_Sint_FKD_Final")
        wandb.agent(sweep_id, lambda:train_kd_wandb(teacher))

# ============================================== PRUEBAS CUANT =========================================================

    if (conf["KDR"]["cuantizar"] in {"pre", "post", "both"}):
        print("\n============================ Cuantizando ============================")

        teacher_q = cuantizar_estatica(teacher, device, val_loader)
        student_q = cuantizar_estatica(student, device, val_loader)
        cpu = torch.device("cpu")
        teacher_q.to(cpu)
        student_q.to(cpu) # La cuantización se aplica sobre cpu siempre, pasar modelos explicitamente a gpu

        # Comparacion de tamaños
        torch.save(teacher_q.state_dict(), "model/teacher_q.pth")
        torch.save(student_q.state_dict(), "model/baseline_q.pth")

        quantized_teacher = os.path.getsize("model/teacher_q.pth") / 1024 ** 2
        print(f"Quantized Teacher Size: {quantized_teacher}")
        quantized_student = os.path.getsize("model/student_q.pth") / 1024 ** 2
        print(f"Quantized Student Size: {quantized_student}\n")
        print(f"Diferencia de tamaño teacher: {quantized_teacher / teacher_size:.2f}x "
              f"({quantized_teacher:.2f}MB -> {teacher_size:.2f}MB)")
        print(f"Diferencia de tamaño student: {quantized_student / student_size:.2f}x "
              f"({quantized_student:.2f}MB -> {student_size:.2f}MB)")

        idx = int(conf["KDR"].get("plot_idx", 0))
        idx = max(0, min(idx, len(test_ds) - 1))
        print(f"Mostrando muestra de test idx={idx}\n")
        graficar(teacher_q, test_ds, cpu, idx=idx, modelName="teacher_q", modo="canales")
        graficar(student_q, test_ds, cpu, idx=idx, modelName="student_q", modo="canales")

        mseTq = evaluate(teacher_q, test_loader, cpu)
        psnrTq = evaluate_psnr(teacher_q, test_loader, cpu)
        mseSq = evaluate(student_q, test_loader, cpu)
        psnrSq = evaluate_psnr(student_q, test_loader, cpu)
        print(f"MSE teacherq: {mseTq:.8f} / teacher_noQ: {mseT:.8f}: ")
        print(f"MSE studentq: {mseSq:.8f} / student_noQ: {mseS:.8f}: ")
        print(f"PSNR teacherq: {psnrTq:.8f} / teacher_noQ: {psnrT:.8f}: ")
        print(f"PSNR studentq: {psnrSq:.8f} / student_noQ: {psnrS:.8f}: ")
        print(f"Diferencia MSE Teacher: {mseTq - mseT:.8f} ({(mseTq - mseT)/mseT:.2f}%)")
        print(f"Diferencia MSE Student: {mseSq - mseS:.8f} ({(mseSq - mseS)/mseS:.2f}%)")

        cuant_baseline = {
            "teacher": {
                "MSE": mseTq,
                "PSNR": psnrTq,
                "Size_MB": quantized_teacher,
            },
            "baseline": {
                "MSE": mseSq,
                "PSNR": psnrSq,
                "Size_MB": quantized_student,
            }
        }
        with open(f"{results_dir}/baseline_cuant.json", "w") as f:
            json.dump(cuant_baseline, f)

        cpu = torch.device("cpu")
        # Quant post-kd
        resultados_cuant, model = run_with_kfold(train_fn=train_kd,
                                                 model_fn=MODEL_REGISTRY[sModel]["Student"],
                                                 load_fold_fn=cargar_fold,
                                                 device=device,
                                                 batch=batch_size,
                                                 teacher=teacher,
                                                 alpha=conf["KDR"]["alpha"],
                                                 patience=conf["Model"]["patience"],
                                                 epochs=conf["Model"]["sEpoch"],
                                                 learning_rate=conf["Model"]["lr"],
                                                 cuant=conf["KDR"]["cuantizar"],)
        torch.save(model.state_dict(), "model/student_q_post.pth")
        quantized_student = os.path.getsize("model/student_q_post.pth") / 1024 ** 2
        print(f"post KD Quantized Student Size: {quantized_student}\n")
        mean_latT, std_latT, mean_per_sampleT = medir_latencia_gpu(model, test_loader, cpu)

        resultados_cuant["size"] = quantized_student
        resultados_cuant["mean_per_sampleT"] = mean_per_sampleT

        with open(f"{results_dir}/cuant_{conf["KDR"]["cuantizar"]}.json", "w") as f:
            json.dump(resultados_cuant, f)