"""
config.py — Rutas, clases e hiperparámetros del proyecto seguridad_obra.

Importa este módulo desde cualquier otro .py o desde el notebook:
    from modelos.config import DATA_DIR, SSD_CLASS_NAMES, ...
"""
from pathlib import Path

# ──────────────────────────────────────────
# Rutas base
# ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent          # seguridad_obra/
DATA_DIR = BASE_DIR / "data" / "css-data"        # dataset principal
CHECKPOINTS_DIR = BASE_DIR / "modelos" / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "outputs" / "logs"

# Aseguramos que existen las carpetas de salida
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────
# Pesos YOLOv8 ya disponibles
# ──────────────────────────────────────────
# best.pt entrenado 100 épocas (ya en data/)
YOLO_PRETRAINED: Path = (
    BASE_DIR
    / "data"
    / "results_yolov8n_100e"
    / "kaggle"
    / "working"
    / "runs"
    / "detect"
    / "train"
    / "weights"
    / "best.pt"
)
# Modelo base de Ultralytics (se descarga automáticamente la primera vez)
YOLO_BASE_MODEL: str = "yolov8n.pt"

# YAML de datos para YOLO (se genera automáticamente con train_yolo.generate_data_yaml)
DATA_YAML: Path = BASE_DIR / "modelos" / "ppe_data.yaml"

# ──────────────────────────────────────────
# Definición de clases
# ──────────────────────────────────────────
# Clases originales del dataset (10 clases, formato YOLO)
YOLO_CLASS_NAMES: dict[int, str] = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
}

# Mapeo: clase original → clase SSD (solo 5 relevantes para seguridad laboral)
ORIGINAL_TO_SSD: dict[int, int] = {
    0: 1,  # Hardhat        → helmet
    2: 2,  # NO-Hardhat     → no-helmet
    7: 3,  # Safety Vest    → vest
    4: 4,  # NO-Safety Vest → no-vest
    5: 5,  # Person         → person
}

# Nombres de clases en espacio SSD (0 = background, obligatorio en SSD)
SSD_CLASS_NAMES: dict[int, str] = {
    0: "background",
    1: "helmet",
    2: "no-helmet",
    3: "vest",
    4: "no-vest",
    5: "person",
}
NUM_CLASSES_SSD: int = 6   # 5 clases + background

# ──────────────────────────────────────────
# Colores (BGR para OpenCV, RGB para PIL)
# ──────────────────────────────────────────
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    1: (0, 200, 0),      # helmet    → verde
    2: (0, 0, 220),      # no-helmet → rojo
    3: (255, 130, 0),    # vest      → naranja
    4: (0, 140, 255),    # no-vest   → naranja claro
    5: (160, 160, 160),  # person    → gris
}

CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    1: (0, 200, 0),
    2: (220, 0, 0),
    3: (0, 130, 255),
    4: (255, 140, 0),
    5: (160, 160, 160),
}

SAFE_COLOR_BGR: tuple[int, int, int] = (0, 210, 0)       # verde  → trabajador seguro
INFRACTION_COLOR_BGR: tuple[int, int, int] = (0, 0, 220)  # rojo   → infracción

# ──────────────────────────────────────────
# Hiperparámetros — SSD300 (fine-tuning)
# ──────────────────────────────────────────
# Fine-tuning desde backbone VGG16 preentrenado (ImageNet).
# LR más bajo que entrenamiento desde cero porque el backbone ya converge.
SSD_BATCH_SIZE: int = 8
SSD_NUM_EPOCHS: int = 100                 # techo máximo; early stopping lo detiene antes
SSD_LEARNING_RATE: float = 0.001          # LR cabeza de detección (referencia HPO)
SSD_BACKBONE_LR_FACTOR: float = 0.1      # backbone recibe LR * factor (= 0.0001)
SSD_MOMENTUM: float = 0.9
SSD_WEIGHT_DECAY: float = 5e-4
SSD_LR_MILESTONES: list[int] = [40, 70]  # ajustados para 100 épocas
SSD_LR_GAMMA: float = 0.1
SSD_GRAD_CLIP_NORM: float = 10.0
SSD_INPUT_SIZE: int = 300

# Factores para el espacio de búsqueda Optuna (relativos a SSD_LEARNING_RATE)
# → rango HPO = [SSD_LEARNING_RATE * LOW,  SSD_LEARNING_RATE * HIGH]
# Con LR=0.001: [1e-4, 1e-2]  — si cambias el LR, el rango se adapta solo.
SSD_HPO_LR_LOW_FACTOR: float = 0.1    # lr_min = SSD_LEARNING_RATE × 0.1
SSD_HPO_LR_HIGH_FACTOR: float = 10.0  # lr_max = SSD_LEARNING_RATE × 10

# ──────────────────────────────────────────
# Hiperparámetros — YOLOv8
# ──────────────────────────────────────────
YOLO_EPOCHS: int = 30
YOLO_IMGSZ: int = 640
YOLO_BATCH: int = 16
YOLO_PATIENCE: int = 10   # early stopping

# ──────────────────────────────────────────
# Umbrales de inferencia (compartidos)
# ──────────────────────────────────────────
SCORE_THRESHOLD: float = 0.40   # confianza mínima para aceptar una detección
IOU_THRESHOLD: float = 0.50     # IoU mínimo para considerar True Positive
LOG_COOLDOWN_SECONDS: float = 5.0  # tiempo mínimo entre entradas de log
