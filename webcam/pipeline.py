"""
pipeline.py — Pipeline de detección en tiempo real (webcam o vídeo).

Soporta tanto SSD300 como YOLOv8. Todas las funciones son independientes
y pueden llamarse por separado desde el notebook.

Funciones exportadas:
    load_ssd_for_inference(checkpoint_path, device)    → model
    load_yolo_for_inference(weights_path)              → YOLO model
    preprocess_frame(frame, device)                    → tensor  [SSD]
    run_inference_ssd(model, frame_tensor, device)     → dict
    run_inference_yolo(model, frame)                   → dict  (mismo formato)
    filter_by_score(preds, threshold)                  → dict filtrado
    associate_equipment_to_persons(person_boxes, equip) → list[(idx, is_safe)]
    draw_equipment_boxes(frame, boxes, labels, scores) → None  (in-place)
    draw_person_boxes(frame, person_boxes, safety)     → None  (in-place)
    draw_counters(frame, n_persons, n_infractions)     → None  (in-place)
    save_log_entry(frame, n_persons, n_inf, csv_path)  → None
    run_webcam(model_type, weights_path, source)       → None  (loop principal)

Uso desde el notebook:
    from webcam.pipeline import run_webcam
    from modelos.config import YOLO_PRETRAINED
    run_webcam(model_type="yolo", weights_path=YOLO_PRETRAINED)
"""
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from modelos.config import (
    CHECKPOINTS_DIR,
    CLASS_COLORS_BGR,
    INFRACTION_COLOR_BGR,
    LOG_COOLDOWN_SECONDS,
    LOGS_DIR,
    ORIGINAL_TO_SSD,
    SAFE_COLOR_BGR,
    SCORE_THRESHOLD,
    SSD_CLASS_NAMES,
    YOLO_PRETRAINED,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
CLASS_NAMES = SSD_CLASS_NAMES


# ──────────────────────────────────────────
# Carga de modelos
# ──────────────────────────────────────────

def load_ssd_for_inference(
    checkpoint_path: Path = CHECKPOINTS_DIR / "ssd_best.pth",
    device: torch.device | None = None,
) -> torch.nn.Module:
    """
    Carga el modelo SSD300 desde un checkpoint .pth en modo inferencia.

    Args:
        checkpoint_path: ruta al .pth guardado con save_checkpoint
        device:          dispositivo; si None, detecta CUDA automáticamente

    Returns:
        Modelo en modo eval listo para inferencia.
    """
    from modelos.train_ssd import build_ssd_model, load_checkpoint

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ssd_model().to(device)
    load_checkpoint(model, checkpoint_path, device=device)
    model.eval()
    return model


def load_yolo_for_inference(weights_path: Path = YOLO_PRETRAINED):
    """
    Carga un modelo YOLOv8 desde un .pt en modo inferencia.

    Args:
        weights_path: ruta al .pt (usa YOLO_PRETRAINED por defecto)

    Returns:
        Objeto YOLO de Ultralytics.
    """
    from modelos.train_yolo import build_yolo_model
    return build_yolo_model(weights_path)


# ──────────────────────────────────────────
# Preprocesado (SSD)
# ──────────────────────────────────────────

def preprocess_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convierte un frame OpenCV (BGR, uint8) a tensor float32 RGB en [0, 1].

    Args:
        frame:  frame capturado por cv2 (HxWx3, BGR)
        device: dispositivo destino

    Returns:
        Tensor (3, H, W) en el dispositivo indicado.
    """
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


# ──────────────────────────────────────────
# Inferencia
# ──────────────────────────────────────────

def run_inference_ssd(
    model: torch.nn.Module,
    frame_tensor: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Ejecuta el modelo SSD sobre un único frame preprocesado.

    Returns:
        Dict {"boxes": ndarray, "labels": ndarray, "scores": ndarray}
        con valores en CPU.
    """
    with torch.no_grad():
        predictions = model([frame_tensor])
    pred = predictions[0]
    return {k: v.cpu() for k, v in pred.items()}


def run_inference_yolo(model, frame: np.ndarray) -> dict:
    """
    Ejecuta YOLOv8 sobre un frame OpenCV y devuelve predicciones
    mapeadas al espacio de clases SSD (mismas 5 clases relevantes).

    Args:
        model: objeto YOLO de Ultralytics
        frame: frame BGR de OpenCV

    Returns:
        Dict {"boxes": tensor, "labels": tensor, "scores": tensor}
        con el mismo formato que run_inference_ssd, en CPU.
    """
    results = model.predict(frame, verbose=False)[0]
    boxes_list, labels_list, scores_list = [], [], []

    if results.boxes is not None:
        for box_data in results.boxes:
            orig_cls = int(box_data.cls.item())
            if orig_cls not in ORIGINAL_TO_SSD:
                continue
            ssd_cls = ORIGINAL_TO_SSD[orig_cls]
            xyxy = box_data.xyxy[0].cpu().tolist()
            conf = float(box_data.conf.item())
            boxes_list.append(xyxy)
            labels_list.append(ssd_cls)
            scores_list.append(conf)

    return {
        "boxes":  torch.tensor(boxes_list, dtype=torch.float32) if boxes_list else torch.zeros((0, 4)),
        "labels": torch.tensor(labels_list, dtype=torch.int64),
        "scores": torch.tensor(scores_list, dtype=torch.float32),
    }


# ──────────────────────────────────────────
# Filtrado y análisis
# ──────────────────────────────────────────

def filter_by_score(predictions: dict, threshold: float = SCORE_THRESHOLD) -> dict:
    """
    Elimina predicciones con score < threshold.

    Returns:
        Dict con boxes/labels/scores filtrados como ndarrays.
    """
    if isinstance(predictions["scores"], torch.Tensor):
        mask = predictions["scores"] >= threshold
        return {
            "boxes":  predictions["boxes"][mask].numpy(),
            "labels": predictions["labels"][mask].numpy(),
            "scores": predictions["scores"][mask].numpy(),
        }
    # Si ya son ndarrays
    mask = predictions["scores"] >= threshold
    return {
        "boxes":  predictions["boxes"][mask],
        "labels": predictions["labels"][mask],
        "scores": predictions["scores"][mask],
    }


def associate_equipment_to_persons(
    person_boxes: np.ndarray,
    equipment: list[tuple[int, np.ndarray]],
) -> list[tuple[int, bool]]:
    """
    Asocia equipos (casco, chaleco) a personas usando el centro geométrico.

    Para cada persona, comprueba si el centro de cada caja de equipo
    cae dentro del bounding box de la persona.
    Una persona es SEGURA si tiene al menos un casco Y un chaleco,
    y NO tiene ningún no-casco o no-chaleco.

    Args:
        person_boxes: array (N, 4) con cajas de personas [x1,y1,x2,y2]
        equipment:    lista de (cls_id, box) para equipos (clases 1–4)

    Returns:
        Lista de (idx_persona, is_safe) para cada persona detectada.
    """
    results = []
    for i, pb in enumerate(person_boxes):
        has_helmet    = False
        has_vest      = False
        has_no_helmet = False
        has_no_vest   = False

        for cls_id, eb in equipment:
            cx = (eb[0] + eb[2]) / 2.0
            cy = (eb[1] + eb[3]) / 2.0
            inside = pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]
            if not inside:
                continue
            if cls_id == 1:
                has_helmet = True
            elif cls_id == 3:
                has_vest = True
            elif cls_id == 2:
                has_no_helmet = True
            elif cls_id == 4:
                has_no_vest = True

        is_safe = has_helmet and has_vest and not has_no_helmet and not has_no_vest
        results.append((i, is_safe))
    return results


# ──────────────────────────────────────────
# Dibujo sobre frame
# ──────────────────────────────────────────

def draw_equipment_boxes(
    frame: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
) -> None:
    """Dibuja cajas de equipos (no personas) sobre el frame en-lugar."""
    for box, label, score in zip(boxes, labels, scores):
        cls = int(label)
        if cls == 5:   # personas las dibuja draw_person_boxes
            continue
        color = CLASS_COLORS_BGR.get(cls, (200, 200, 200))
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{CLASS_NAMES.get(cls, '?')} {score:.2f}"
        cv2.putText(frame, text, (x1, max(0, y1 - 5)), FONT, 0.45, color, 1)


def draw_person_boxes(
    frame: np.ndarray,
    person_boxes: np.ndarray,
    person_safety: list[tuple[int, bool]],
) -> None:
    """Dibuja cajas de personas en verde (seguro) o rojo (infracción) en-lugar."""
    for pid, is_safe in person_safety:
        box = person_boxes[pid]
        color = SAFE_COLOR_BGR if is_safe else INFRACTION_COLOR_BGR
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = "OK" if is_safe else "INFRACCION"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)), FONT, 0.6, color, 2)


def draw_counters(
    frame: np.ndarray,
    person_count: int,
    infraction_count: int,
) -> None:
    """Dibuja un overlay con el contador de personas e infracciones."""
    cv2.rectangle(frame, (0, 0), (270, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Personas: {person_count}",
                (10, 30), FONT, 0.9, (255, 255, 255), 2)
    inf_color = INFRACTION_COLOR_BGR if infraction_count > 0 else (0, 210, 0)
    cv2.putText(frame, f"Infracciones: {infraction_count}",
                (10, 65), FONT, 0.9, inf_color, 2)


# ──────────────────────────────────────────
# Logging
# ──────────────────────────────────────────

def save_log_entry(
    frame: np.ndarray,
    person_count: int,
    infraction_count: int,
    csv_path: Path,
) -> None:
    """
    Guarda una entrada en el CSV de infracciones y una captura de pantalla.

    Args:
        frame:            frame BGR actual (se guarda como .jpg)
        person_count:     número de personas detectadas
        infraction_count: número de infracciones
        csv_path:         ruta al archivo CSV (se crea si no existe)
    """
    timestamp  = datetime.now().isoformat()
    safe_ts    = timestamp.replace(":", "-").replace(".", "-")
    screenshot = f"capture_{safe_ts}.jpg"
    cv2.imwrite(str(LOGS_DIR / screenshot), frame)

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "personas", "infracciones", "captura"])
        writer.writerow([timestamp, person_count, infraction_count, screenshot])


def _open_csv_path() -> Path:
    """Devuelve una ruta CSV con timestamp en LOGS_DIR."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"infractions_{ts}.csv"


# ──────────────────────────────────────────
# Loop principal
# ──────────────────────────────────────────

def run_webcam(
    model_type: str = "yolo",
    weights_path: Path | None = None,
    source: int | str = 0,
    score_threshold: float = SCORE_THRESHOLD,
) -> None:
    """
    Ejecuta el pipeline de detección en tiempo real sobre webcam o vídeo.

    Args:
        model_type:      "yolo" o "ssd"
        weights_path:    ruta al .pt (YOLO) o .pth (SSD).
                         Si None, usa los defaults de config.
        source:          índice de cámara (0 = webcam) o ruta a vídeo
        score_threshold: umbral de confianza mínimo

    Controles:
        Q → salir

    Ejemplo:
        from webcam.pipeline import run_webcam
        run_webcam(model_type="yolo")                    # webcam
        run_webcam(model_type="yolo", source="video.mp4") # vídeo
    """
    model_type = model_type.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | Modelo: {model_type.upper()}")

    # ── Cargar modelo ──
    if model_type == "ssd":
        if weights_path is None:
            weights_path = CHECKPOINTS_DIR / "ssd_best.pth"
        if not weights_path.exists():
            print(f"ERROR: No se encontró checkpoint SSD en {weights_path}")
            print("Entrena primero con train_ssd.train_ssd()")
            return
        model = load_ssd_for_inference(weights_path, device)
    elif model_type == "yolo":
        if weights_path is None:
            weights_path = YOLO_PRETRAINED
        if not weights_path.exists():
            print(f"ERROR: No se encontró modelo YOLO en {weights_path}")
            return
        model = load_yolo_for_inference(weights_path)
    else:
        raise ValueError(f"model_type debe ser 'ssd' o 'yolo', no '{model_type}'")

    # ── Abrir fuente de vídeo ──
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: No se puede abrir la fuente de vídeo: {source}")
        return

    csv_path      = _open_csv_path()
    last_log_time = 0.0
    print(f"Pipeline activo ({model_type.upper()}). Pulsa 'Q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del vídeo o error al leer frame.")
            break

        # ── Inferencia ──
        if model_type == "ssd":
            frame_tensor = preprocess_frame(frame, device)
            raw_preds    = run_inference_ssd(model, frame_tensor, device)
        else:
            raw_preds = run_inference_yolo(model, frame)

        preds  = filter_by_score(raw_preds, score_threshold)
        boxes  = preds["boxes"]
        labels = preds["labels"]
        scores = preds["scores"]

        # ── Separar personas de equipos ──
        person_mask  = labels == 5
        person_boxes = boxes[person_mask]
        equipment    = [(int(labels[i]), boxes[i]) for i in range(len(labels)) if labels[i] != 5]

        person_safety    = associate_equipment_to_persons(person_boxes, equipment)
        person_count     = len(person_boxes)
        infraction_count = sum(1 for _, is_safe in person_safety if not is_safe)

        # ── Dibujar ──
        draw_equipment_boxes(frame, boxes, labels, scores)
        draw_person_boxes(frame, person_boxes, person_safety)
        draw_counters(frame, person_count, infraction_count)

        # ── Log de infracciones ──
        now = time.time()
        if infraction_count > 0 and (now - last_log_time) >= LOG_COOLDOWN_SECONDS:
            save_log_entry(frame.copy(), person_count, infraction_count, csv_path)
            last_log_time = now

        cv2.imshow(f"SafeGuard AI — {model_type.upper()}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Log guardado en: {csv_path}")
