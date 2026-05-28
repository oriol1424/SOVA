"""
dataset.py — Carga y preprocesamiento del dataset Construction Site Safety para SSD300.

Preprocesamiento aplicado
─────────────────────────
Train (augmentación):
  1. RandomHorizontalFlip (p=0.5)
       Trabajadores y EPIs aparecen en cualquier orientación horizontal.
       Las cajas se invierten simétricamente.
  2. ColorJitter (brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
       Obras con iluminación muy variable: amanecer, mediodía, nublado, focos.
       Parámetros moderados para no distorsionar colores diagnósticos
       (casco amarillo, chaleco naranja).
  3. GaussianBlur (kernel 3×3, p=0.2)
       Simula cámaras de seguridad de baja resolución y desenfoque por movimiento
       de trabajadores. p=0.2 para que solo afecte a 1 de cada 5 imágenes.
  4. ToTensor → escala [0,255] a [0,1]

Val / Test (sin augmentación):
  Solo ToTensor.

Normalización: la aplica internamente el modelo SSD300-VGG16 de torchvision
(GeneralizedRCNNTransform) con los estadísticos ImageNet
(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
No se aplica aquí para evitar doble normalización.

Funciones exportadas:
    build_dataloaders(batch_size)    → (train_loader, val_loader)
    get_transforms(training)         → función (image, boxes) → (tensor, boxes)
    ConstructionSafetyDataset        → clase Dataset de PyTorch
    collate_fn                       → función de collate para DataLoader

Uso desde el notebook:
    from modelos.dataset import build_dataloaders
    train_loader, val_loader = build_dataloaders(batch_size=8)
"""
import random
from pathlib import Path
from typing import Callable

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from modelos.config import DATA_DIR, ORIGINAL_TO_SSD, SSD_BATCH_SIZE


# ──────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────

class ConstructionSafetyDataset(Dataset):
    """
    Dataset YOLO → SSD.

    Lee imágenes (.jpg) y sus labels (.txt en formato YOLO normalizado),
    filtra a las 5 clases relevantes y convierte las coords a xyxy absoluto.

    Args:
        split:      "train", "valid" o "test"
        transforms: función (image, boxes) → (tensor, boxes).
                    Si None, solo aplica ToTensor.
    """

    def __init__(self, split: str, transforms: Callable | None = None) -> None:
        self.images_dir = DATA_DIR / split / "images"
        self.labels_dir = DATA_DIR / split / "labels"
        self.transforms = transforms
        self.samples = self._collect_samples()

    def _collect_samples(self) -> list[tuple[Path, Path]]:
        """Empareja cada imagen con su archivo de labels correspondiente."""
        pairs = []
        for img_path in sorted(self.images_dir.glob("*.jpg")):
            label_path = self.labels_dir / img_path.with_suffix(".txt").name
            if label_path.exists():
                pairs.append((img_path, label_path))
        return pairs

    def _parse_labels(
        self, label_path: Path, img_w: int, img_h: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parsea un .txt YOLO → tensores SSD.

        Filtra clases irrelevantes, remapea IDs y convierte
        de (cx, cy, w, h) normalizado a (x1, y1, x2, y2) absoluto.
        """
        boxes: list[list[float]] = []
        labels: list[int] = []

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                orig_cls = int(parts[0])
                if orig_cls not in ORIGINAL_TO_SSD:
                    continue
                xc, yc, w, h = map(float, parts[1:5])
                xmin = max(0.0, (xc - w / 2) * img_w)
                ymin = max(0.0, (yc - h / 2) * img_h)
                xmax = min(float(img_w), (xc + w / 2) * img_w)
                ymax = min(float(img_h), (yc + h / 2) * img_h)
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(ORIGINAL_TO_SSD[orig_cls])

        if boxes:
            return (
                torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.int64),
            )
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.int64)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        boxes, labels = self._parse_labels(label_path, img_w, img_h)

        if self.transforms is not None:
            image, boxes = self.transforms(image, boxes)
        else:
            image = TF.to_tensor(image)

        return image, {"boxes": boxes, "labels": labels}

    def __len__(self) -> int:
        return len(self.samples)


# ──────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────

# Instancias reutilizables (evita crear un objeto nuevo en cada llamada)
_COLOR_JITTER = T.ColorJitter(
    brightness=0.3,   # variaciones de exposición en obra (sombras, focos)
    contrast=0.3,     # diferencias entre zonas iluminadas y en sombra
    saturation=0.2,   # cámaras con distintas calibraciones de color
    hue=0.05,         # mínimo, para no distorsionar colores diagnósticos
)
_GAUSSIAN_BLUR = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))


def get_transforms(training: bool) -> Callable:
    """
    Devuelve la función de transformación adecuada para cada split.

    training=True  → flip horizontal + color jitter + gaussian blur + to_tensor
    training=False → solo to_tensor
                     (la normalización ImageNet la aplica el modelo internamente)

    Args:
        training: True para train, False para valid/test.

    Returns:
        Función con firma (image: PIL.Image, boxes: Tensor) → (Tensor, Tensor).
    """
    if training:
        def transform(image: Image.Image, boxes: torch.Tensor):
            # 1. Flip horizontal aleatorio (p=0.5)
            #    Las cajas se invierten simétricamente respecto al eje vertical.
            if random.random() < 0.5:
                w = image.width
                image = TF.hflip(image)
                if boxes.shape[0] > 0:
                    flipped = boxes.clone()
                    flipped[:, 0] = w - boxes[:, 2]   # x1_new = w - x2_old
                    flipped[:, 2] = w - boxes[:, 0]   # x2_new = w - x1_old
                    boxes = flipped

            # 2. Color jitter (brightness, contrast, saturation, hue)
            image = _COLOR_JITTER(image)

            # 3. Gaussian blur suave (p=0.2)
            #    Simula desenfoque de cámaras de baja resolución y movimiento.
            if random.random() < 0.2:
                image = _GAUSSIAN_BLUR(image)

            # 4. PIL → tensor [0, 1]
            return TF.to_tensor(image), boxes

    else:
        def transform(image: Image.Image, boxes: torch.Tensor):
            # Solo conversión a tensor — sin augmentación para evaluación reproducible
            return TF.to_tensor(image), boxes

    return transform


# ──────────────────────────────────────────
# Collate y DataLoaders
# ──────────────────────────────────────────

def collate_fn(batch: list) -> tuple:
    """Collate personalizado para targets de tamaño variable (detección)."""
    return tuple(zip(*batch))


def build_dataloaders(
    batch_size: int = SSD_BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """
    Crea DataLoaders de entrenamiento y validación con los transforms apropiados.

    Args:
        batch_size: tamaño de batch (default: SSD_BATCH_SIZE de config.py)

    Returns:
        (train_loader, val_loader)

    Ejemplo:
        train_loader, val_loader = build_dataloaders(batch_size=8)
        images, targets = next(iter(train_loader))
    """
    import os
    # num_workers=0 bloqueaba la GPU esperando I/O de Drive.
    # Con datos en disco local y workers paralelos, la carga es ~100x más rápida.
    # En Windows (local) se mantiene 0 para evitar problemas con multiprocessing.
    _num_workers = min(4, os.cpu_count() or 0) if os.name != "nt" else 0

    train_ds = ConstructionSafetyDataset("train", transforms=get_transforms(training=True))
    val_ds   = ConstructionSafetyDataset("valid", transforms=get_transforms(training=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=_num_workers,
        pin_memory=(_num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=_num_workers,
        pin_memory=(_num_workers > 0),
    )
    print(f"Dataset — train: {len(train_ds)} imgs | valid: {len(val_ds)} imgs")
    print(f"Augmentacion train: HFlip(p=0.5) + ColorJitter + GaussianBlur(p=0.2)")
    return train_loader, val_loader


def get_test_dataset() -> ConstructionSafetyDataset:
    """Devuelve el dataset de test (sin augmentacion, para evaluacion reproducible)."""
    return ConstructionSafetyDataset("test", transforms=get_transforms(training=False))
