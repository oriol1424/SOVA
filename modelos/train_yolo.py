"""
train_yolo.py — Entrenamiento / fine-tuning de YOLOv8 con Ultralytics.

Funciones exportadas:
    generate_data_yaml(output_path)         → Path (crea el .yaml necesario)
    build_yolo_model(weights_path)          → YOLO
    train_yolo(epochs, from_pretrained, ...) → Path al best.pt resultante

Uso desde el notebook:
    from modelos.train_yolo import generate_data_yaml, train_yolo
    yaml_path = generate_data_yaml()
    best_pt = train_yolo(epochs=30, from_pretrained=True)
"""
from pathlib import Path

import yaml

from modelos.config import (
    DATA_DIR,
    DATA_YAML,
    YOLO_BASE_MODEL,
    YOLO_BATCH,
    YOLO_EPOCHS,
    YOLO_IMGSZ,
    YOLO_PATIENCE,
    YOLO_PRETRAINED,
    YOLO_CLASS_NAMES,
)


# ──────────────────────────────────────────
# YAML de datos
# ──────────────────────────────────────────

def generate_data_yaml(output_path: Path = DATA_YAML) -> Path:
    """
    Genera el archivo .yaml que YOLO necesita para saber dónde están los datos.

    Usa rutas absolutas derivadas de DATA_DIR (css-data/).
    Se sobreescribe si ya existe.

    Args:
        output_path: ruta donde guardar el .yaml (default: modelos/ppe_data.yaml)

    Returns:
        Path del archivo generado.

    Ejemplo:
        yaml_path = generate_data_yaml()
        print(yaml_path)
    """
    data = {
        "path": str(DATA_DIR),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(YOLO_CLASS_NAMES),
        "names": [YOLO_CLASS_NAMES[i] for i in sorted(YOLO_CLASS_NAMES)],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print(f"YAML generado en: {output_path}")
    return output_path


# ──────────────────────────────────────────
# Modelo
# ──────────────────────────────────────────

def build_yolo_model(weights_path: str | Path | None = None):
    """
    Carga un modelo YOLOv8.

    Args:
        weights_path:
            - None o "yolov8n.pt"  → modelo base de Ultralytics (se descarga si no existe)
            - Path al best.pt      → carga modelo ya entrenado / para fine-tuning

    Returns:
        Objeto YOLO de Ultralytics.

    Ejemplo:
        from modelos.config import YOLO_PRETRAINED
        model = build_yolo_model(YOLO_PRETRAINED)   # carga best.pt ya disponible
        model = build_yolo_model()                   # modelo base
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics no está instalado. Ejecuta: pip install ultralytics"
        )

    if weights_path is None:
        weights_path = YOLO_BASE_MODEL
    return YOLO(str(weights_path))


# ──────────────────────────────────────────
# Entrenamiento / fine-tuning
# ──────────────────────────────────────────

def train_yolo(
    epochs: int = YOLO_EPOCHS,
    imgsz: int = YOLO_IMGSZ,
    batch: int = YOLO_BATCH,
    patience: int = YOLO_PATIENCE,
    from_pretrained: bool = True,
    data_yaml: Path = DATA_YAML,
    project_dir: Path | None = None,
    run_name: str = "yolo_finetune",
) -> Path:
    """
    Entrena (o hace fine-tuning de) un modelo YOLOv8n.

    Args:
        epochs:          número de épocas
        imgsz:           tamaño de imagen de entrada
        batch:           tamaño de batch (-1 = auto)
        patience:        épocas sin mejora antes de early stopping
        from_pretrained: True  → fine-tune desde YOLO_PRETRAINED (best.pt ya disponible)
                         False → entrenar desde el modelo base yolov8n.pt
        data_yaml:       path al .yaml del dataset (se genera con generate_data_yaml si no existe)
        project_dir:     directorio de salida (default: modelos/checkpoints/yolo_runs)
        run_name:        nombre del experimento

    Returns:
        Path al best.pt resultante del entrenamiento.

    Ejemplo:
        yaml_path = generate_data_yaml()
        best_pt = train_yolo(epochs=10, from_pretrained=True)
        print(f"Modelo guardado en: {best_pt}")
    """
    from ultralytics import YOLO
    from modelos.config import CHECKPOINTS_DIR

    # Generar yaml si no existe
    if not data_yaml.exists():
        print("YAML no encontrado, generando...")
        generate_data_yaml(data_yaml)

    # Seleccionar pesos de partida
    if from_pretrained and YOLO_PRETRAINED.exists():
        weights = str(YOLO_PRETRAINED)
        print(f"Fine-tuning desde: {weights}")
    else:
        weights = YOLO_BASE_MODEL
        print(f"Entrenando desde modelo base: {weights}")

    if project_dir is None:
        project_dir = CHECKPOINTS_DIR / "yolo_runs"

    model = YOLO(weights)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    # Ruta al mejor modelo guardado por Ultralytics
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nEntrenamiento YOLO completado. Best model: {best_pt}")
    return best_pt
