# Estrategias de optimización y compresión de modelos de IA para su despliegue en redes inalambricas

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

> **Nota:** si vas a usar GPU, instala antes una versión de `torch` compatible con CUDA, antes de ejecutar `pip install -r requirements.txt`. Este proyecto ha usado:
>
> ```powershell
> pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```

## 4. Ejecución
### A. Knowledge Distillation para denoising 

`KD_Denoising.py` es el script principal del proyecto, configurable mediante el archivo config.json. En caso de no contar con un dataset con las especificaciones necesarias para el proyecto, es necesario ejecutar `LecturaDatos.py`

Ejecutar:

```powershell
python LecturaDatos.py
```

Esto creará los datasets en la dirección indicada, para poder ser usados por el script principal:

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

## 5. Configuraciones disponibles

### 5.1 `config.json`

| Sección | Clave | Valor actual | Descripción corta |
|---|---:|---:|---|
| `KDR` | `X` | `data/Datos_Sinteticos/TDL_D_85ns_fd1000_SNR20_h_est.npy` | Dataset ruidoso de entrada. |
| `KDR` | `Y` | `data/Datos_Sinteticos/TDL_D_85ns_fd1000_SNR20_h_real.npy` | Dataset limpio / objetivo. |
| `KDR` | `sModel` | `UNet` | Arquitectura del student. |
| `KDR` | `tModel` | `UNet` | Arquitectura del teacher. |
| `KDR` | `t_train` | `true` | Entrena el teacher en lugar de cargarlo. |
| `KDR` | `s_train` | `true` | Entrena el student base en lugar de cargarlo. |
| `KDR` | `KD` | `true` | Activa Knowledge Distillation clásica. |
| `KDR` | `FKD` | `true` | Activa Feature-based KD. |
| `KDR` | `AKD` | `true` | Activa Attention-based KD. |
| `KDR` | `cuantizar` | `pre` | Tipo de prueba de cuantización (`pre`, `post`, `both`).|
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

### 5.2 `config_wandb.json`

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

## 6. Salidas generadas

Durante la ejecución se generan los siguientes archivos:

- **`model/*.pth`**: pesos guardados de los modelos (teacher, student, KD, FKD, AKD y sus versiones cuantizadas).
- **`results/<fecha_hora>/`**: carpeta de resultados de cada ejecución, con:
  - `hist_*.json`: históricos de entrenamiento/validación cuando no se usa K-Fold (teacher, baseline, RKD, FKD, AKD).
  - `kfold_*.json`: métricas agregadas (MSE/PSNR medio y desviación) y por fold cuando se usa K-Fold (teacher, baseline, RKD, FKD, AKD). 
  - `baseline_results.json`: evaluación de MSE, PSNR, latencia y tamaño de los modelos teacher y baseline.
  - `baseline_cuant.json`: evaluación de MSE, PSNR y tamaño de los modelos teacher y baseline cuantizados (sin latencia).
  - `cuant_<tipo>.json`: resultados de la combinación KD clásica + cuantización (`pre`, `post` o `both`), incluyendo tamaño final y latencia.
  - `config.json`: copia de la configuración usada en esa ejecución, para trazabilidad de resultados.
- **`wandb/`**: logs y artefactos de Weights & Biases, si `KDR.wandb` está activado.
