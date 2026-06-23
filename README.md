# AI_Complexity_Reduction

Este proyecto explora cómo reducir la complejidad de redes neuronales para denoising de señales complejas, usando **Knowledge Distillation** (clásica, basada en features y en atención) y **cuantización** en PyTorch. El objetivo es comparar modelos "teacher" grandes frente a versiones "student" más ligeras, evaluando el compromiso entre tamaño, latencia y calidad de reconstrucción (MSE/PSNR).

## 1. Flujo de trabajo

El repositorio está organizado en dos etapas, que se ejecutan en el siguiente orden:

1. **Preparación de datos** (`LecturaDatos.py`): carga los datos NIST o sintéticos, genera las imágenes complejas a partir de ellos y construye los folds usados para K-Fold en caso de estar indicado.
2. **Entrenamiento** (`KD_Denoising.py`): entrena el modelo teacher y distintas variantes de student (sin destilar, KD clásica, FKD, AKD).

## 2. Estructura principal del repositorio

| Archivo | Función |
|---|---|
| `Knowledge_Distillation.py` | Ejemplo de KD clásico sobre CIFAR-10. |
| `KD_Denoising.py` | Script principal para denoising con teacher/student, KD, FKD, AKD, K-Fold, W&B y cuantización. |
| `LecturaDatos.py` | Carga, preprocesado y generación de datasets/folds. |
| `train_models.py` | Funciones de entrenamiento para denoising: `train_basic`, `train_kd`, `train_fkd`, `train_akd` y `run_with_kfold`. |
| `Cuantizacion.py` | Cuantización estática y QAT. |
| `modelos.py` | Arquitecturas de redes (UNet, DnCNN, ResNet, etc.). |
| `Analisis_Datos.py` | Análisis visual del dataset y generación de histogramas. |
| `utils.py` | Gráficas, hooks, métricas, PSNR, latencia y utilidades varias. |
| `config.json` | Configuración principal del experimento. |
| `config_wandb.json` | Configuración del sweep de Weights & Biases. |

---

## 3. Instalación

1. Crear un entorno virtual.
2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

> Nota: si vas a usar GPU, asegúrate de instalar una versión de `torch` compatible con tu CUDA.

## 5. Cómo usar el código
### A. Denoising principal

`KD_Denoising.py` es el script principal del proyecto. Lee toda su configuración desde `config.json` (no usa argumentos de línea de comandos) y hace lo siguiente:

1. **Carga de datos**: lee los `.npy` de train/test/val indicados en `KDR.X` (entrada con ruido) y `KDR.Y` (objetivo limpio), y construye los `DataLoader` correspondientes. Si `Data.Kfold` es `true`, en su lugar usa `cargar_fold` para leer los folds desde `data/folds/est` y `data/folds/real`.
2. **Teacher**: si `KDR.t_train` es `true`, lo entrena con `train_basic` (o con `run_with_kfold` si `Data.Kfold` es `true`); si es `false`, carga los pesos guardados en `model/{tModel}_{tDepth}l_{snr_med}snr.pth`.
3. **Student base (sin destilar)**: igual que el teacher, controlado por `KDR.s_train`, guardado en `model/{sModel}_{sDepth}l_{snr_med}snr.pth`.
4. **Comparativa teacher vs. student**: grafica una muestra de test (señal con ruido / limpia / reconstruida), calcula MSE, PSNR y latencia de ambos modelos y guarda la comparación en `baseline_results.json`.
5. **KD clásica, FKD y AKD**: cada una se entrena solo si su flag (`KDR.KD`, `KDR.FKD`, `KDR.AKD`) está activo, usando `train_kd`, `train_fkd` y `train_akd` respectivamente (o sus equivalentes con K-Fold). Los pesos resultantes se guardan como `model/kd_*.pth`, `model/fkd_*.pth` y `model/akd_*.pth`.
6. **Cuantización**: si `KDR.cuantizar` es `pre`, `post` o `both`, cuantiza estáticamente teacher y student ya entrenados (comparando tamaño/MSE/PSNR antes y después), y además relanza un entrenamiento KD completo con cuantización integrada (`pre`, `post` o `both`) sobre los folds.
7. **Sweep de W&B**: si `KDR.wandb` es `true`, lanza un sweep bayesiano (definido en `config_wandb.json`) que entrena repetidamente con FKD variando `alpha`, `beta`, `learning_rate`, etc.
Todos los resultados de cada ejecución (históricos, métricas, configuración usada) se guardan en una carpeta nueva `results/<fecha_hora>/`.

Ejecutar:

```powershell
python KD_Denoising.py
```
### B. Knowledge Distillation clásica en CIFAR-10

`Knowledge_Distillation.py` es un script independiente (no depende de `config.json`) que entrena o carga un teacher y un student sobre CIFAR-10, y luego lanza un barrido manual de KD variando `alpha` y la temperatura `T`.

Argumentos disponibles:

| Argumento | Valores | Default | Descripción |
|---|---|---|---|
| `--teacher` | `resnet18`, `deep`, `deep_adaptada` | `resnet18` | Arquitectura del teacher. |
| `--student` | `light`, `light_adaptada` | `light_adaptada` | Arquitectura del student. |
| `--mode` | `load`, `full` | `load` | `full` entrena teacher (salvo `resnet18`, que ya viene preentrenado) y student desde cero y guarda los pesos en `model/`. `load` carga los pesos ya guardados en `model/{teacher}.pth` y `model/{student}.pth`. |

Ejecutar:

```powershell
python Knowledge_Distillation.py --mode load
```

Opciones útiles:

```powershell
python Knowledge_Distillation.py --mode full --teacher resnet18 --student light_adaptada
python Knowledge_Distillation.py --mode load --teacher deep --student light
```

### C. Análisis de datos

Ejecutar:

```powershell
python Analisis_Datos.py
```

Esto genera visualizaciones del dataset y, en el flujo actual.

---

## 6. Configuraciones disponibles

### 6.1 `config.json`

| Sección | Clave | Valor actual | Descripción corta |
|---|---:|---:|---|
| `KDR` | `X` | `data/NIST_h_GBurg_5G_snr_16.npy` | Dataset ruidoso de entrada. |
| `KDR` | `Y` | `data/NIST_h_GBurg_5G.npy` | Dataset limpio / objetivo. |
| `KDR` | `sModel` | `UNet` | Arquitectura del student. |
| `KDR` | `tModel` | `UNet` | Arquitectura del teacher. |
| `KDR` | `t_train` | `true` | Entrena el teacher en lugar de cargarlo. |
| `KDR` | `s_train` | `true` | Entrena el student base en lugar de cargarlo. |
| `KDR` | `KD` | `true` | Activa Knowledge Distillation clásica. |
| `KDR` | `FKD` | `true` | Activa Feature-based KD. |
| `KDR` | `AKD` | `true` | Activa Attention-based KD. |
| `KDR` | `cuantizar` | `pre` | Tipo de prueba de cuantización (`pre`, `post`, `both`). |
| `KDR` | `wandb` | `false` | Activa logging y sweeps con Weights & Biases. |
| `KDR` | `plot_idx` | `1437` | Índice de muestra para gráficas comparativas. |
| `KDR` | `alpha` | `0.62` | Peso de la pérdida KD. |
| `KDR` | `beta` | `0.6` | Peso para pérdidas de FKD / AKD. |
| `Model` | `tDepth` | `64` | Tamaño/profundidad del teacher. |
| `Model` | `tEpoch` | `700` | Épocas del teacher. |
| `Model` | `sDepth` | `2` | Tamaño/profundidad del student. |
| `Model` | `sEpoch` | `700` | Épocas del student. |
| `Model` | `lr` | `0.0001` | Learning rate base. |
| `Model` | `batch_size` | `32` | Tamaño de batch. |
| `Model` | `patience` | `5` | Patience de early stopping. |
| `Data` | `Dset` | `h_GBurg_5G` | Nombre base del dataset NIST. |
| `Data` | `Sint` | `true` | Indica si se usa un dataset sintético. |
| `Data` | `MixedSNR` | `true` | Mezcla varios valores de SNR. |
| `Data` | `Stride` | `128` | Paso de ventana para generar muestras. |
| `Data` | `Snr_db` | `[10, 13, 15, 17, 20, 22, 25]` | Lista de SNRs de entrada. |
| `Data` | `Kfold` | `false` | Activa validación cruzada. |

### 6.2 `config_wandb.json`

| Clave | Valor actual | Descripción corta |
|---|---|---|
| `method` | `bayes` | Estrategia del sweep. |
| `metric.name` | `mse` | Métrica objetivo a optimizar. |
| `metric.goal` | `minimize` | Objetivo: minimizar la métrica. |
| `parameters.alpha` | `0.0` a `1.0` | Rango de búsqueda para `alpha`. |
| `parameters.beta` | `0.0` a `1.0` | Rango de búsqueda para `beta`. |
| `parameters.learning_rate` | `0.0001` | Valores de búsqueda para el learning rate. |
| `parameters.epochs` | `700` | Épocas probadas en el sweep. |
| `parameters.batch_size` | `32` | Batch size usado en el sweep. |
| `parameters.patience` | `5` | Patience usada en el sweep. |

---

## 7. Salidas generadas

Durante la ejecución se generan los siguientes archivos:

- **`model/*.pth`**: pesos guardados de los modelos (teacher, student, KD, FKD, AKD y sus versiones cuantizadas).
- **`results/<fecha_hora>/`**: carpeta de resultados de cada ejecución, con:
  - `hist_*.json`: históricos de entrenamiento/validación cuando no se usa K-Fold (teacher, baseline, RKD, FKD, AKD).
  - `kfold_*.json`: métricas agregadas (MSE/PSNR medio y desviación) y por fold cuando se usa K-Fold.
  - `baseline_results.json` y `baseline_cuant.json`: evaluación de MSE, PSNR, latencia y tamaño de los modelos teacher y baseline y sus versiones de cuantizadas en caso de aplicarse.
  - `cuant_<tipo>.json`: resultados de la combinación KD + cuantización (`pre`, `post` o `both`), incluyendo tamaño final y latencia.
  - `config.json`: copia de la configuración usada en esa ejecución, para trazabilidad de resultados.
  - Todos los JSON se guardan con `indent=4` para que sean legibles, no en una sola línea.
- **`wandb/`**: logs y artefactos de Weights & Biases, si `KDR.wandb` está activado.
- Gráficas de comparación (señal con ruido vs. señal limpia vs. señal reconstruida, mapas de atención) se muestran en pantalla durante la ejecución mediante `matplotlib`.
