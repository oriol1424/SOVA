"""
pipeline_tracking.py — SSD con Tracking + Inferencia Selectiva.

Ejecuta SSD solo cada `inference_interval` frames.
En frames intermedios usa CentroidTracker (IoU-based) para mantener
posiciones e IDs sin coste computacional adicional.

Mejoras respecto a pipeline.py (SSD estándar):
  - FPS visualmente fluido aunque SSD sea lento en CPU.
  - Cada persona tiene un ID estable entre frames.
  - Logging inteligente: captura solo cuando el estado de una persona cambia.

Uso desde el notebook:
    from webcam.pipeline_tracking import run_webcam_ssd_tracking
    run_webcam_ssd_tracking()
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
    INFRACTION_COLOR_BGR,
    LOGS_DIR,
    SAFE_COLOR_BGR,
)
from webcam.pipeline import (
    associate_equipment_to_persons,
    draw_counters,
    draw_equipment_boxes,
    filter_by_score,
    load_ssd_for_inference,
    preprocess_frame,
    run_inference_ssd,
)
from webcam.tracker import CentroidTracker

FONT = cv2.FONT_HERSHEY_SIMPLEX
SSD_SCORE_THRESHOLD = 0.40   # umbral calibrado para tiempo real (0.10 maximiza recall pero genera falsos positivos)


# ──────────────────────────────────────────
# Dibujo con IDs
# ──────────────────────────────────────────

def draw_tracked_persons(frame: np.ndarray, tracks: list[tuple]) -> None:
    """Dibuja cajas de personas con su ID y estado de seguridad."""
    for bbox, track_id, is_safe in tracks:
        color = SAFE_COLOR_BGR if is_safe else INFRACTION_COLOR_BGR
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        state = "OK" if is_safe else "INFRACCION"
        cv2.putText(frame, f"#{track_id} {state}", (x1, max(0, y1 - 6)),
                    FONT, 0.6, color, 2)


def _draw_fps(frame: np.ndarray, fps: float) -> None:
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    cv2.rectangle(frame, (w - tw - 14, 4), (w - 4, th + 14), (0, 0, 0), -1)
    cv2.putText(frame, text, (w - tw - 10, th + 8), FONT, 0.7, (0, 255, 180), 2)


def _draw_mode_indicator(frame: np.ndarray, is_inference_frame: bool) -> None:
    """Indicador visual: naranja cuando SSD corre, gris cuando solo trackea."""
    color = (0, 140, 255) if is_inference_frame else (80, 80, 80)
    label = "SSD" if is_inference_frame else "TRACK"
    cv2.circle(frame, (20, 105), 8, color, -1)
    cv2.putText(frame, label, (33, 110), FONT, 0.5, color, 1)


# ──────────────────────────────────────────
# Logging inteligente por ID
# ──────────────────────────────────────────

class _StateLogger:
    """Captura y registra solo cuando el estado de una persona cambia."""

    def __init__(self, csv_path: Path):
        self.csv_path   = csv_path
        self._last: dict[int, bool] = {}

    def check_and_log(self, frame: np.ndarray, tracks: list[tuple]) -> None:
        for _, track_id, is_safe in tracks:
            if is_safe is None:
                continue
            if self._last.get(track_id) == is_safe:
                continue
            self._last[track_id] = is_safe
            self._save(frame.copy(), track_id, is_safe)

    def _save(self, frame: np.ndarray, track_id: int, is_safe: bool) -> None:
        timestamp  = datetime.now().isoformat()
        safe_ts    = timestamp.replace(":", "-").replace(".", "-")
        screenshot = f"capture_tracking_{safe_ts}_id{track_id}.jpg"
        cv2.imwrite(str(LOGS_DIR / screenshot), frame)

        file_exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "persona_id", "estado", "captura"])
            writer.writerow([timestamp, track_id,
                             "OK" if is_safe else "INFRACCION", screenshot])


def _new_csv_path() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"infractions_tracking_{ts}.csv"


# ──────────────────────────────────────────
# Loop principal
# ──────────────────────────────────────────

def run_webcam_ssd_tracking(
    weights_path: Path | None = None,
    source: int | str = 0,
    score_threshold: float = SSD_SCORE_THRESHOLD,
    inference_interval: int = 5,
) -> None:
    """
    Pipeline SSD con Tracking + Inferencia Selectiva.

    Args:
        weights_path:       .pth del SSD. Si None usa el checkpoint por defecto.
        source:             índice de cámara (0) o ruta a vídeo.
        score_threshold:    umbral de confianza SSD (default 0.10).
        inference_interval: ejecuta SSD cada N frames; en los demás usa solo tracker.

    Controles:
        Q → salir
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if weights_path is None:
        weights_path = CHECKPOINTS_DIR / "ssd_best.pth"
    if not weights_path.exists():
        print(f"ERROR: checkpoint SSD no encontrado en {weights_path}")
        return

    print(f"Dispositivo: {device} | SSD + Tracking (inferencia cada {inference_interval} frames)")
    model   = load_ssd_for_inference(weights_path, device)
    tracker = CentroidTracker(max_missed=15, min_iou=0.25)
    logger  = _StateLogger(_new_csv_path())

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: no se puede abrir la fuente: {source}")
        return

    # Cajas de equipo de la última inferencia, se mantienen entre frames
    last_eq_boxes  = np.zeros((0, 4))
    last_eq_labels = np.array([], dtype=int)
    last_eq_scores = np.array([])

    frame_idx = 0
    fps       = 0.0
    t_prev    = time.time()

    print("Pipeline activo (SSD + Tracking). Pulsa 'Q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        is_inf = (frame_idx % inference_interval == 0)

        if is_inf:
            tensor = preprocess_frame(frame, device)
            raw    = run_inference_ssd(model, tensor, device)
            preds  = filter_by_score(raw, score_threshold)
            boxes, labels, scores = preds["boxes"], preds["labels"], preds["scores"]

            person_mask  = labels == 5
            person_boxes = boxes[person_mask]
            equipment    = [(int(labels[i]), boxes[i])
                            for i in range(len(labels)) if labels[i] != 5]

            safety = associate_equipment_to_persons(person_boxes, equipment)
            dets   = [(person_boxes[i], is_safe) for i, is_safe in safety]
            tracks = tracker.update(dets)

            eq_mask        = ~person_mask
            last_eq_boxes  = boxes[eq_mask]
            last_eq_labels = labels[eq_mask].astype(int)
            last_eq_scores = scores[eq_mask]
        else:
            tracks = tracker.propagate()

        n_persons     = len(tracks)
        n_infractions = sum(1 for _, _, s in tracks if s is False)

        logger.check_and_log(frame, tracks)

        draw_equipment_boxes(frame, last_eq_boxes, last_eq_labels, last_eq_scores)
        draw_tracked_persons(frame, tracks)
        draw_counters(frame, n_persons, n_infractions)
        _draw_fps(frame, fps)
        _draw_mode_indicator(frame, is_inf)

        t_now  = time.time()
        dt     = (t_now - t_prev) or 1e-6
        fps    = 0.85 * fps + 0.15 / dt
        t_prev = t_now

        cv2.imshow("SafeGuard AI — SSD + Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Log guardado en: {logger.csv_path}")
