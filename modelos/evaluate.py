"""
evaluate.py — Evaluación y comparación de SSD300 y YOLOv8.

Funciones exportadas:
    compute_iou(box_a, box_b)                          → float
    compute_metrics(predictions, targets, ...)          → dict
    evaluate_ssd(model, dataset, device, ...)           → (predictions, targets)
    evaluate_yolo(weights_path, data_yaml, split)       → dict de métricas
    draw_boxes_pil(image, boxes, labels, scores)        → PIL.Image
    print_metrics(metrics, model_name)                  → None (imprime tabla)
    compare_models(ssd_metrics, yolo_metrics)           → None (tabla comparativa)

Uso desde el notebook:
    from modelos.evaluate import evaluate_ssd, evaluate_yolo, compare_models
    ssd_m  = compute_metrics(preds, targets, ...)
    yolo_m = evaluate_yolo(YOLO_PRETRAINED, DATA_YAML, split="test")
    compare_models(ssd_m, yolo_m)
"""
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from modelos.config import (
    CLASS_COLORS_RGB,
    DATA_YAML,
    IOU_THRESHOLD,
    NUM_CLASSES_SSD,
    ORIGINAL_TO_SSD,
    OUTPUTS_DIR,
    SCORE_THRESHOLD,
    SSD_CLASS_NAMES,
    YOLO_PRETRAINED,
)

# Alias para simplificar (usamos SSD_CLASS_NAMES como referencia compartida)
CLASS_NAMES = SSD_CLASS_NAMES


# ──────────────────────────────────────────
# Utilidades geométricas
# ──────────────────────────────────────────

def compute_iou(
    box_a: list | torch.Tensor,
    box_b: list | torch.Tensor,
) -> float:
    """
    Calcula IoU entre dos cajas [x1, y1, x2, y2].

    Args:
        box_a, box_b: listas o tensores con formato [xmin, ymin, xmax, ymax]

    Returns:
        Valor de IoU en [0, 1].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# ──────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────

def compute_metrics(
    all_predictions: list[dict],
    all_targets: list[dict],
    num_classes: int = NUM_CLASSES_SSD,
    score_threshold: float = SCORE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, dict[str, float]]:
    """
    Calcula Precision y Recall por clase sobre el conjunto de test.

    Una predicción es True Positive si su IoU con el GT más cercano
    (misma clase, sin asignar) es >= iou_threshold.

    Args:
        all_predictions: lista de dicts {"boxes", "labels", "scores"} (tensores)
        all_targets:     lista de dicts {"boxes", "labels"} (tensores)
        num_classes:     número total de clases (incluye background=0)
        score_threshold: umbral de confianza mínimo
        iou_threshold:   IoU mínimo para True Positive

    Returns:
        Dict  clase_nombre → {"precision", "recall", "tp", "fp", "fn"}
    """
    tp = {c: 0 for c in range(1, num_classes)}
    fp = {c: 0 for c in range(1, num_classes)}
    fn = {c: 0 for c in range(1, num_classes)}

    for preds, target in zip(all_predictions, all_targets):
        gt_boxes  = target["boxes"]
        gt_labels = target["labels"]
        pred_boxes  = preds["boxes"]
        pred_labels = preds["labels"]
        pred_scores = preds["scores"]

        # Filtrar por score
        mask = pred_scores >= score_threshold
        pred_boxes  = pred_boxes[mask]
        pred_labels = pred_labels[mask]

        for cls in range(1, num_classes):
            cls_pred_idx = (pred_labels == cls).nonzero(as_tuple=True)[0]
            cls_gt_idx   = (gt_labels == cls).nonzero(as_tuple=True)[0]
            matched: set[int] = set()

            for pi in cls_pred_idx:
                best_iou, best_gi = 0.0, -1
                for gi in cls_gt_idx:
                    if gi.item() in matched:
                        continue
                    iou = compute_iou(pred_boxes[pi].tolist(), gt_boxes[gi].tolist())
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi.item()
                if best_iou >= iou_threshold and best_gi >= 0:
                    tp[cls] += 1
                    matched.add(best_gi)
                else:
                    fp[cls] += 1

            fn[cls] += len(cls_gt_idx) - len(matched)

    metrics: dict[str, dict[str, float]] = {}
    for cls in range(1, num_classes):
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        recall    = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        metrics[CLASS_NAMES[cls]] = {
            "precision": precision,
            "recall":    recall,
            "tp": tp[cls],
            "fp": fp[cls],
            "fn": fn[cls],
        }
    return metrics


# ──────────────────────────────────────────
# Visualización
# ──────────────────────────────────────────

def draw_boxes_pil(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> Image.Image:
    """
    Dibuja bounding boxes con etiqueta y score sobre una imagen PIL.

    Args:
        image:  imagen PIL (se modifica una copia)
        boxes:  tensor (N, 4) en formato [x1, y1, x2, y2]
        labels: tensor (N,) con IDs de clase SSD
        scores: tensor (N,) con scores de confianza

    Returns:
        Nueva imagen PIL con las anotaciones dibujadas.
    """
    image = image.copy()
    draw  = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        cls   = label.item()
        color = CLASS_COLORS_RGB.get(cls, (200, 200, 200))
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{CLASS_NAMES.get(cls, '?')} {score:.2f}"
        draw.text((x1, max(0, y1 - 16)), text, fill=color, font=font)

    return image


# ──────────────────────────────────────────
# Evaluación SSD
# ──────────────────────────────────────────

def evaluate_ssd(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    save_images: bool = True,
    outputs_dir: Path = OUTPUTS_DIR,
) -> tuple[list[dict], list[dict]]:
    """
    Ejecuta inferencia SSD sobre todo el dataset de test.

    Args:
        model:       modelo SSD cargado y en modo eval
        dataset:     ConstructionSafetyDataset("test")
        device:      dispositivo de inferencia
        save_images: si True, guarda imágenes anotadas en outputs_dir
        outputs_dir: directorio de salida para las imágenes

    Returns:
        (all_predictions, all_targets) — listas de dicts con tensores en CPU

    Ejemplo:
        preds, targets = evaluate_ssd(model, test_ds, device)
        metrics = compute_metrics(preds, targets)
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            image_tensor, target = dataset[idx]
            img_path = dataset.samples[idx][0]

            output = model([image_tensor.to(device)])[0]
            output = {k: v.cpu() for k, v in output.items()}
            all_predictions.append(output)

            if save_images:
                original = Image.open(img_path).convert("RGB")
                annotated = draw_boxes_pil(
                    original,
                    output["boxes"],
                    output["labels"],
                    output["scores"],
                )
                annotated.save(outputs_dir / f"ssd_{img_path.name}")

    all_targets = [dataset[i][1] for i in range(len(dataset))]
    print(f"Inferencia SSD completada — {len(dataset)} imágenes")
    return all_predictions, all_targets


# ──────────────────────────────────────────
# Evaluación YOLO
# ──────────────────────────────────────────

def evaluate_yolo(
    weights_path: Path = YOLO_PRETRAINED,
    data_yaml: Path = DATA_YAML,
    split: str = "test",
    score_threshold: float = SCORE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, dict[str, float]]:
    """
    Evalúa un modelo YOLOv8 y devuelve métricas en el mismo formato que SSD.

    Mapea las 10 clases YOLO a las 5 clases SSD relevantes para comparar
    de forma equitativa entre modelos.

    Args:
        weights_path:    path al .pt del modelo YOLO
        data_yaml:       path al .yaml del dataset
        split:           "test", "val", o "train"
        score_threshold: umbral de confianza
        iou_threshold:   IoU mínimo para TP

    Returns:
        Dict clase_nombre → {"precision", "recall", "tp", "fp", "fn"}

    Ejemplo:
        from modelos.config import YOLO_PRETRAINED, DATA_YAML
        metrics = evaluate_yolo(YOLO_PRETRAINED, DATA_YAML)
        print_metrics(metrics, "YOLOv8n")
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Instala ultralytics: pip install ultralytics")

    from modelos.config import DATA_DIR

    model = YOLO(str(weights_path))

    # Inferencia sobre el split indicado
    split_dir = DATA_DIR / split / "images"
    results_list = model.predict(
        source=str(split_dir),
        conf=score_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    # Construir predictions en formato SSD (solo clases mapeadas)
    all_predictions = []
    all_targets = []

    labels_dir = DATA_DIR / split / "labels"

    for result in results_list:
        img_path = Path(result.path)
        h, w = result.orig_shape

        # Predicciones: filtrar y mapear a clases SSD
        boxes_list, labels_list, scores_list = [], [], []
        if result.boxes is not None:
            for box_data in result.boxes:
                orig_cls = int(box_data.cls.item())
                if orig_cls not in ORIGINAL_TO_SSD:
                    continue
                ssd_cls = ORIGINAL_TO_SSD[orig_cls]
                xyxy = box_data.xyxy[0].cpu().tolist()
                conf = float(box_data.conf.item())
                boxes_list.append(xyxy)
                labels_list.append(ssd_cls)
                scores_list.append(conf)

        all_predictions.append({
            "boxes":  torch.tensor(boxes_list, dtype=torch.float32) if boxes_list else torch.zeros((0, 4)),
            "labels": torch.tensor(labels_list, dtype=torch.int64),
            "scores": torch.tensor(scores_list, dtype=torch.float32),
        })

        # Ground truth: leer label correspondiente
        label_path = labels_dir / img_path.with_suffix(".txt").name
        gt_boxes_list, gt_labels_list = [], []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    orig_cls = int(parts[0])
                    if orig_cls not in ORIGINAL_TO_SSD:
                        continue
                    xc, yc, bw, bh = map(float, parts[1:5])
                    xmin = max(0.0, (xc - bw / 2) * w)
                    ymin = max(0.0, (yc - bh / 2) * h)
                    xmax = min(float(w), (xc + bw / 2) * w)
                    ymax = min(float(h), (yc + bh / 2) * h)
                    if xmax > xmin and ymax > ymin:
                        gt_boxes_list.append([xmin, ymin, xmax, ymax])
                        gt_labels_list.append(ORIGINAL_TO_SSD[orig_cls])

        all_targets.append({
            "boxes":  torch.tensor(gt_boxes_list, dtype=torch.float32) if gt_boxes_list else torch.zeros((0, 4)),
            "labels": torch.tensor(gt_labels_list, dtype=torch.int64),
        })

    metrics = compute_metrics(
        all_predictions, all_targets,
        num_classes=NUM_CLASSES_SSD,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    print(f"Evaluación YOLO completada — {len(results_list)} imágenes")
    return metrics


# ──────────────────────────────────────────
# Impresión de resultados
# ──────────────────────────────────────────

def print_metrics(
    metrics: dict[str, dict[str, float]],
    model_name: str = "Modelo",
) -> None:
    """
    Imprime una tabla formateada de métricas por clase.

    Args:
        metrics:    salida de compute_metrics o evaluate_yolo
        model_name: nombre para la cabecera de la tabla
    """
    print(f"\n{'─'*55}")
    print(f"  {model_name} — Precision / Recall por clase")
    print(f"{'─'*55}")
    header = f"{'Clase':<12} {'Precision':>10} {'Recall':>10} {'TP':>6} {'FP':>6} {'FN':>6}"
    print(header)
    print("─" * len(header))
    for cls_name, m in metrics.items():
        print(
            f"{cls_name:<12} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{int(m['tp']):>6} "
            f"{int(m['fp']):>6} "
            f"{int(m['fn']):>6}"
        )
    # Macro-average
    if metrics:
        avg_p = sum(m["precision"] for m in metrics.values()) / len(metrics)
        avg_r = sum(m["recall"]    for m in metrics.values()) / len(metrics)
        print("─" * len(header))
        print(f"{'MEDIA':<12} {avg_p:>10.4f} {avg_r:>10.4f}")


def compare_models(
    ssd_metrics: dict[str, dict[str, float]],
    yolo_metrics: dict[str, dict[str, float]],
) -> None:
    """
    Imprime una tabla comparativa side-by-side entre SSD y YOLO.

    Args:
        ssd_metrics:  salida de compute_metrics para SSD
        yolo_metrics: salida de evaluate_yolo

    Ejemplo:
        compare_models(ssd_m, yolo_m)
    """
    print(f"\n{'═'*75}")
    print("  Comparación SSD300 vs YOLOv8n")
    print(f"{'═'*75}")
    header = (
        f"{'Clase':<12} "
        f"{'SSD Prec':>9} {'SSD Rec':>8} "
        f"{'YOLOPrec':>9} {'YOLORec':>8} "
        f"{'ΔPrec':>7} {'ΔRec':>7}"
    )
    print(header)
    print("─" * len(header))

    all_classes = sorted(set(list(ssd_metrics.keys()) + list(yolo_metrics.keys())))
    for cls in all_classes:
        s = ssd_metrics.get(cls, {"precision": 0.0, "recall": 0.0})
        y = yolo_metrics.get(cls, {"precision": 0.0, "recall": 0.0})
        dp = y["precision"] - s["precision"]
        dr = y["recall"]    - s["recall"]
        dp_str = f"{dp:+.3f}"
        dr_str = f"{dr:+.3f}"
        print(
            f"{cls:<12} "
            f"{s['precision']:>9.4f} {s['recall']:>8.4f} "
            f"{y['precision']:>9.4f} {y['recall']:>8.4f} "
            f"{dp_str:>7} {dr_str:>7}"
        )

    print("─" * len(header))
    # Macro-average
    classes = [c for c in all_classes if c in ssd_metrics and c in yolo_metrics]
    if classes:
        avg_sp = sum(ssd_metrics[c]["precision"]  for c in classes) / len(classes)
        avg_sr = sum(ssd_metrics[c]["recall"]     for c in classes) / len(classes)
        avg_yp = sum(yolo_metrics[c]["precision"] for c in classes) / len(classes)
        avg_yr = sum(yolo_metrics[c]["recall"]    for c in classes) / len(classes)
        print(
            f"{'MEDIA':<12} "
            f"{avg_sp:>9.4f} {avg_sr:>8.4f} "
            f"{avg_yp:>9.4f} {avg_yr:>8.4f} "
            f"{avg_yp-avg_sp:>+7.3f} {avg_yr-avg_sr:>+7.3f}"
        )
    print(f"{'═'*75}")
