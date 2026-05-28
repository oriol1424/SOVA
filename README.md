# SafeGuard AI — Detección de EPIs en Obras de Construcción

Sistema de visión por computador para la detección automática de Equipos de Protección Individual (EPIs) en obras de construcción. Compara dos arquitecturas de detección de objetos —**SSD300-VGG16** con fine-tuning y **YOLOv8n** con fine-tuning— partiendo ambas de pesos preentrenados para una comparación justa, e incluye un pipeline de vigilancia en tiempo real con registro de infracciones.

---

## Tabla de contenidos

- [Descripción](#descripción)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Dataset](#dataset)
- [Modelos](#modelos)
- [Preprocesamiento](#preprocesamiento)
- [Instalación](#instalación)
- [Flujo de trabajo](#flujo-de-trabajo)
- [Pipeline en tiempo real](#pipeline-en-tiempo-real)
- [Resultados](#resultados)

---

## Descripción

El sistema detecta si los trabajadores llevan casco y chaleco reflectante y emite alertas visuales en tiempo real. Cuando se detecta una infracción, guarda automáticamente una captura del frame y un registro CSV con timestamp.

**Clases detectadas:**

| ID | Clase | Descripción |
|----|-------|-------------|
| 1 | `helmet` | Casco de seguridad |
| 2 | `no-helmet` | Persona sin casco |
| 3 | `vest` | Chaleco reflectante |
| 4 | `no-vest` | Persona sin chaleco |
| 5 | `person` | Persona genérica |

> YOLOv8 detecta 10 clases originales del dataset que se mapean a las 5 anteriores en el pipeline de inferencia.

---

## Estructura del proyecto

```
seguridad_obra/
├── main.ipynb                  <- Notebook principal (local, CPU)
├── colab_train.ipynb           <- Fine-tuning SSD en Google Colab (GPU)
├── requirements.txt
│
├── data/
│   ├── css-data/               <- Dataset principal (train / valid / test)
│   ├── results_yolov8n_100e/   <- Pesos YOLOv8 ya entrenados (best.pt)
│   └── source_files/           <- Videos e imagenes de prueba
│
├── modelos/
│   ├── config.py               <- Rutas, clases, hiperparametros
│   ├── dataset.py              <- Dataset PyTorch + preprocesamiento SSD
│   ├── train_ssd.py            <- Fine-tuning SSD300-VGG16 con HPO Optuna
│   ├── train_yolo.py           <- Fine-tuning YOLOv8 (Ultralytics)
│   ├── evaluate.py             <- Metricas: precision, recall, F1, IoU
│   ├── ppe_data.yaml           <- Configuracion YAML para YOLO
│   └── checkpoints/
│       └── ssd_best.pth        <- Checkpoint SSD (generado en Colab)
│
├── webcam/
│   └── pipeline.py             <- Pipeline de deteccion en tiempo real
│
└── outputs/
    ├── ssd_*.jpg               <- Imagenes de evaluacion con bboxes SSD
    └── logs/
        ├── infractions_*.csv   <- Registro de infracciones (timestamp, personas, n infracciones)
        └── capture_*.jpg       <- Capturas automaticas cuando hay infraccion
```

---

## Dataset

**Construction Site Safety** — Roboflow / Kaggle  
[https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)

| Split | Imagenes |
|-------|----------|
| Train | 2.605 |
| Valid | 114 |
| Test  | 82 |
| **Total** | **2.801** |

El dataset contiene 10 clases originales. Se filtran a 5 clases relevantes para seguridad laboral mediante el mapeo definido en `config.py` (`ORIGINAL_TO_SSD`).

---

## Modelos

Ambos modelos parten de pesos preentrenados para garantizar una comparacion justa basada en transfer learning.

### SSD300-VGG16 (fine-tuning desde ImageNet)

- **Backbone:** VGG16 preentrenado en ImageNet — extrae caracteristicas visuales generales
- **Cabeza SSD:** inicializada aleatoriamente y entrenada para las 6 clases del proyecto (5 + background)
- **Optimizacion de hiperparametros:** Optuna con TPE (busqueda bayesiana)
  - `lr` cabeza: rango dinamico `[SSD_LEARNING_RATE x 0.1, SSD_LEARNING_RATE x 10]`
  - `batch_size`: {4, 8, 16}
  - 12 trials x 4 epocas cada uno (~45 min en T4)
- **Entrenamiento final:**
  - Epocas maximas: **100**
  - Early stopping: **patience = 10** (para si 10 epocas consecutivas sin mejorar val_loss)
  - Scheduler: MultiStepLR (LR x 0.1 en epocas 40 y 70)
  - LR diferenciado: backbone `LR x 0.1`, cabeza `LR`
- **Entrenamiento:** Google Colab (GPU T4), checkpoint en `modelos/checkpoints/ssd_best.pth`

### YOLOv8n (fine-tuning desde COCO)

- **Modelo base:** `yolov8n.pt` (Ultralytics, preentrenado en COCO 80 clases)
- **Fine-tuning:** 100 epocas sobre el dataset CSS con early stopping
- **Pesos disponibles:** `data/results_yolov8n_100e/.../best.pt`
- Se carga directamente sin necesidad de reentrenar

---

## Preprocesamiento

El preprocesamiento esta implementado en `modelos/dataset.py`.

### Pipeline train (con augmentacion)

| Paso | Parametros | Justificacion |
|------|-----------|---------------|
| RandomHorizontalFlip | p=0.5 | Trabajadores y EPIs aparecen en cualquier orientacion horizontal. Las cajas se invierten simetricamente. |
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05 | Iluminacion variable en obras: amanecer, mediodia, cielo cubierto, focos artificiales. Parametros moderados para no distorsionar colores diagnosticos (casco amarillo, chaleco naranja). |
| GaussianBlur | kernel=3x3, p=0.2 | Simula camaras de seguridad de baja resolucion y desenfoque por movimiento de trabajadores. p=0.2: afecta 1 de cada 5 imagenes. |
| ToTensor | — | Convierte PIL [0,255] a tensor PyTorch [0,1] |

### Pipeline val / test (sin augmentacion)

| Paso | Justificacion |
|------|---------------|
| ToTensor | Unica transformacion para garantizar reproducibilidad de las metricas |

### Normalizacion

La normalizacion ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) la aplica **internamente** el modelo SSD300-VGG16 de torchvision a traves de `GeneralizedRCNNTransform`. No se aplica en el dataset para evitar doble normalizacion.

---

## Instalacion

### Requisitos

- Python 3.10+
- GPU NVIDIA con CUDA 12.1 para entrenamiento (recomendado, o Google Colab gratuito)

### CPU (solo inferencia local)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### GPU (entrenamiento local)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Dependencias principales

```
torch >= 2.2.0
torchvision >= 0.17.0
ultralytics >= 8.0.0
opencv-python >= 4.8.0
numpy >= 1.24.0
pillow >= 10.0.0
matplotlib >= 3.7.0
pyyaml >= 6.0
optuna              <- solo para colab_train.ipynb (HPO)
```

---

## Flujo de trabajo

```
1. [COLAB]  colab_train.ipynb  ->  Seccion 3 (HPO opcional) + Seccion 4 (fine-tuning SSD)
2. [LOCAL]  Descargar ssd_best.pth  ->  pegar en modelos/checkpoints/
3. [LOCAL]  main.ipynb  ->  exploracion, evaluacion, comparacion, demo
```

### `colab_train.ipynb` (Google Colab, GPU)

| Seccion | Descripcion |
|---------|-------------|
| 1. Setup | Monta Drive, instala deps, verifica GPU |
| 2. Entorno | Comprueba dataset, rutas y dispositivo |
| 3. HPO *(opcional)* | Optuna busca `lr` y `batch_size` optimos (guarda `hpo_best_params.json`) |
| 4. Fine-tuning | 100 epocas max, early stopping patience=10, guarda `ssd_best.pth` |
| 5. Instrucciones | Como descargar el checkpoint a local |

> Las secciones 3 y 4 son **idempotentes**: si ya existen `hpo_best_params.json` y `ssd_best.pth`, se saltan automaticamente. Borrar los archivos para reentrenar.

### `main.ipynb` (local, CPU)

| Seccion | Descripcion |
|---------|-------------|
| 0. Setup | Detecta dispositivo, instala deps si faltan |
| 1. Dataset | Estadisticas de clases y visualizacion de muestras con bboxes |
| 2. SSD | Carga checkpoint desde `ssd_best.pth` (o entrena brevemente en CPU si no existe) |
| 3. YOLOv8 | Carga `best.pt` ya disponible en `data/` |
| 4. Evaluacion | Precision / Recall por clase, tabla comparativa SSD vs YOLO, graficas |
| 5. Demo | Pipeline en tiempo real (webcam o video) |

---

## Pipeline en tiempo real

```python
from webcam.pipeline import run_webcam
from modelos.config import YOLO_PRETRAINED, CHECKPOINTS_DIR

# Webcam en tiempo real con YOLOv8
run_webcam(model_type="yolo", weights_path=YOLO_PRETRAINED, source=0)

# Video con SSD
run_webcam(model_type="ssd", weights_path=CHECKPOINTS_DIR / "ssd_best.pth", source="video.mp4")
```

**Controles:** `Q` para salir.

### Logica de seguridad

Cada persona detectada se evalua de forma independiente:

- **SEGURO** — tiene casco Y chaleco, sin detecciones negativas
- **INFRACCION** — falta casco, falta chaleco, o deteccion explicita de ausencia de EPI

Cuando hay infracciones activas, cada 5 segundos se guarda:
- Captura JPG en `outputs/logs/capture_<timestamp>.jpg`
- Fila en `outputs/logs/infractions_<fecha>.csv` con timestamp, n personas y n infracciones

---

## Resultados

Los resultados de evaluacion (precision, recall por clase) se generan en la **seccion 4** de `main.ipynb` sobre el split de test (82 imagenes).

Las imagenes de inferencia del SSD se guardan en `outputs/` durante la evaluacion.

### Umbrales de inferencia

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| `SCORE_THRESHOLD` | 0.40 | Confianza minima para aceptar una deteccion |
| `IOU_THRESHOLD` | 0.50 | IoU minimo para considerar True Positive |
| `LOG_COOLDOWN_SECONDS` | 5.0 | Tiempo minimo entre entradas de log |

### Comparacion de modelos

| | SSD300-VGG16 | YOLOv8n |
|--|-------------|---------|
| Preentrenamiento | ImageNet (backbone) | COCO (modelo completo) |
| Epocas fine-tuning | 100 max (early stopping) | 100 |
| HPO | Optuna TPE | No (parametros fijos) |
| Preprocesamiento | HFlip + ColorJitter + Blur | Augmentacion interna Ultralytics |
| Inferencia local | CPU / GPU | CPU / GPU |
