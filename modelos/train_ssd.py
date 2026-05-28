"""
train_ssd.py — Fine-tuning SSD300-VGG16 con backbone ImageNet preentrenado.

El backbone VGG16 parte de pesos ImageNet (transfer learning).
La cabeza de deteccion SSD se inicializa aleatoriamente y se adapta a nuestras clases.
Se usan learning rates diferenciados: LR bajo para el backbone, LR normal para la cabeza.

Al finalizar el entrenamiento se guarda un JSON con metricas completas
(tiempos, GPU, parametros, curvas de perdida, hiperparametros) util para
incluir en la memoria del proyecto.

Funciones exportadas:
    build_ssd_model()                           -> nn.Module
    get_gpu_info()                              -> dict
    save_checkpoint(model, optimizer, epoch, loss, path)
    load_checkpoint(model, path, optimizer, device) -> (epoch, loss)
    train_one_epoch(model, optimizer, loader, device) -> float
    evaluate_val_loss(model, loader, device)    -> float
    train_ssd(num_epochs, batch_size, lr, ...)  -> (model, history)

Uso desde el notebook:
    from modelos.train_ssd import build_ssd_model, train_ssd, load_checkpoint
    model, history = train_ssd(num_epochs=100, patience=10)
"""
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights
from torchvision.models.detection import ssd300_vgg16

from modelos.config import (
    CHECKPOINTS_DIR,
    NUM_CLASSES_SSD,
    SSD_BACKBONE_LR_FACTOR,
    SSD_BATCH_SIZE,
    SSD_GRAD_CLIP_NORM,
    SSD_LEARNING_RATE,
    SSD_LR_GAMMA,
    SSD_LR_MILESTONES,
    SSD_MOMENTUM,
    SSD_NUM_EPOCHS,
    SSD_WEIGHT_DECAY,
)
from modelos.dataset import build_dataloaders


# ──────────────────────────────────────────
# Modelo
# ──────────────────────────────────────────

def build_ssd_model() -> nn.Module:
    """
    Construye SSD300-VGG16 con fine-tuning desde backbone ImageNet preentrenado.

    - backbone VGG16: carga pesos ImageNet (transfer learning)
    - cabeza SSD:     inicializacion aleatoria
                      (COCO tiene 91 clases != NUM_CLASSES_SSD=6, incompatible)
    """
    return ssd300_vgg16(
        weights=None,
        weights_backbone=VGG16_Weights.DEFAULT,
        num_classes=NUM_CLASSES_SSD,
    )


# ──────────────────────────────────────────
# Info de hardware
# ──────────────────────────────────────────

def get_gpu_info() -> dict:
    """
    Devuelve informacion del dispositivo de entrenamiento.
    Util para incluir en el log y en la memoria del proyecto.

    Returns:
        dict con device, gpu_name, vram_total_gb, vram_available_gb
        (campos GPU son None si se entrena en CPU).
    """
    info: dict = {"device": "cpu", "gpu_name": None, "vram_total_gb": None}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["device"]         = "cuda"
        info["gpu_name"]       = props.name
        info["vram_total_gb"]  = round(props.total_memory / 1e9, 2)
        info["cuda_version"]   = torch.version.cuda
    return info


# ──────────────────────────────────────────
# Checkpoints
# ──────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
) -> None:
    """Guarda estado del modelo y optimizador en un archivo .pth."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss":                 loss,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: Path,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> tuple[int, float]:
    """
    Carga pesos desde un checkpoint .pth.

    Returns:
        (epoch, val_loss) almacenados en el checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    loss  = checkpoint.get("loss",  float("inf"))
    print(f"Checkpoint cargado: epoca {epoch}, val_loss={loss:.4f}")
    return epoch, loss


# ──────────────────────────────────────────
# Log de entrenamiento
# ──────────────────────────────────────────

def _fmt_seconds(s: float) -> str:
    """Convierte segundos a string legible: '1h 23m 45s'."""
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m > 0:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def _count_params(model: nn.Module) -> dict:
    """Cuenta parametros del backbone y la cabeza por separado."""
    backbone = sum(p.numel() for p in model.backbone.parameters())
    head     = sum(p.numel() for p in model.head.parameters())
    return {
        "backbone":    backbone,
        "head":        head,
        "total":       backbone + head,
        "trainable":   sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def save_training_log(log: dict, path: Path) -> None:
    """Guarda el log de entrenamiento como JSON formateado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"Log de entrenamiento guardado: {path}")


def print_training_summary(log: dict) -> None:
    """Imprime un resumen del log al finalizar el entrenamiento."""
    t  = log["training"]
    hw = log["metadata"]
    m  = log["model"]
    print("\n" + "=" * 60)
    print("  RESUMEN DEL ENTRENAMIENTO")
    print("=" * 60)
    print(f"  GPU              : {hw['gpu_name'] or 'CPU'}")
    if hw.get("vram_total_gb"):
        print(f"  VRAM total       : {hw['vram_total_gb']} GB")
    print(f"  Parametros total : {m['params']['total']:,}")
    print(f"    - Backbone     : {m['params']['backbone']:,}")
    print(f"    - Cabeza SSD   : {m['params']['head']:,}")
    print(f"  Epocas ejecutadas: {t['epochs_run']}")
    print(f"  Mejor epoca      : {t['best_epoch']}")
    print(f"  Mejor val_loss   : {t['best_val_loss']:.4f}")
    print(f"  Early stopping   : {'Si' if t['early_stopping_triggered'] else 'No'}")
    print(f"  Tiempo total     : {t['total_time_formatted']}")
    print(f"  Tiempo por epoca : {t['avg_epoch_time_s']:.1f} s")
    if t.get("peak_gpu_memory_gb"):
        print(f"  Pico VRAM usado  : {t['peak_gpu_memory_gb']} GB")
    print("=" * 60)


# ──────────────────────────────────────────
# Bucle de entrenamiento
# ──────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Ejecuta una epoca completa de entrenamiento.

    Returns:
        Perdida media sobre todos los batches.
    """
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), SSD_GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate_val_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Calcula la perdida de validacion sin actualizar pesos.

    Returns:
        Perdida media sobre el conjunto de validacion.
    """
    model.train()   # SSD necesita modo train para calcular la loss
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in data_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total_loss += sum(loss_dict.values()).item()
    return total_loss / len(data_loader)


# ──────────────────────────────────────────
# Funcion de alto nivel
# ──────────────────────────────────────────

def train_ssd(
    num_epochs:      int   = SSD_NUM_EPOCHS,
    batch_size:      int   = SSD_BATCH_SIZE,
    lr:              float = SSD_LEARNING_RATE,
    checkpoints_dir: Path  = CHECKPOINTS_DIR,
    save_every:      int   = 5,
    patience:        int   = 10,
) -> tuple[nn.Module, dict]:
    """
    Fine-tuning SSD300-VGG16 con early stopping, checkpoints y log de metricas.

    Args:
        num_epochs:       maximo de epocas (el early stopping puede parar antes)
        batch_size:       tamano de batch
        lr:               learning rate para la cabeza SSD
                          (backbone recibe lr * SSD_BACKBONE_LR_FACTOR)
        checkpoints_dir:  directorio donde guardar .pth y ssd_training_log.json
        save_every:       checkpoint periodico cada N epocas
        patience:         epocas sin mejora de val_loss antes de parar (0=off)

    Returns:
        (model, history) donde history contiene:
            train_loss      : lista de losses de entrenamiento por epoca
            val_loss        : lista de losses de validacion por epoca
            stopped_epoch   : epoca real en que paro el entrenamiento
            training_log    : dict completo guardado en ssd_training_log.json
    """
    # ── Hardware ──────────────────────────────────────────────────────────────
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = get_gpu_info()

    print(f"Dispositivo      : {device}")
    if gpu_info["gpu_name"]:
        print(f"GPU              : {gpu_info['gpu_name']}  ({gpu_info['vram_total_gb']} GB VRAM)")
    print(f"Fine-tuning SSD300-VGG16  (backbone ImageNet preentrenado)")
    print(f"  LR cabeza      : {lr:.2e}")
    print(f"  LR backbone    : {lr * SSD_BACKBONE_LR_FACTOR:.2e}  (factor={SSD_BACKBONE_LR_FACTOR})")
    print(f"  Batch size     : {batch_size}")
    print(f"  Epocas max     : {num_epochs}  |  Early stopping patience={patience}")
    print(f"  Preprocesamiento: HFlip(p=0.5) + ColorJitter + GaussianBlur(p=0.2)")

    # ── Datos y modelo ────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(batch_size)
    model = build_ssd_model().to(device)
    params = _count_params(model)

    print(f"\nParametros del modelo:")
    print(f"  Backbone (VGG16) : {params['backbone']:>12,}")
    print(f"  Cabeza SSD       : {params['head']:>12,}")
    print(f"  Total            : {params['total']:>12,}")

    # ── Optimizador con LR diferenciado ──────────────────────────────────────
    optimizer = torch.optim.SGD(
        [
            {"params": model.backbone.parameters(), "lr": lr * SSD_BACKBONE_LR_FACTOR},
            {"params": model.head.parameters(),     "lr": lr},
        ],
        momentum=SSD_MOMENTUM,
        weight_decay=SSD_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=SSD_LR_MILESTONES, gamma=SSD_LR_GAMMA
    )

    # Limpiar estadisticas de memoria GPU para medir el pico real
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ── Estructuras de datos ──────────────────────────────────────────────────
    history: dict = {"train_loss": [], "val_loss": [], "stopped_epoch": 0}
    per_epoch_log: list[dict] = []

    best_val_loss     = float("inf")
    epochs_no_improve = 0
    train_start       = time.time()

    # ── Bucle de entrenamiento ────────────────────────────────────────────────
    print(f"\n{'Epoca':>6} | {'Train loss':>10} | {'Val loss':>10} | {'LR':>9} | {'Tiempo':>8}")
    print("-" * 55)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss   = evaluate_val_loss(model, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]   # LR de la cabeza (grupo 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        per_epoch_log.append({
            "epoch":       epoch,
            "train_loss":  round(train_loss, 6),
            "val_loss":    round(val_loss,   6),
            "lr_head":     round(current_lr, 8),
            "lr_backbone": round(current_lr * SSD_BACKBONE_LR_FACTOR, 8),
            "epoch_time_s": round(epoch_time, 2),
        })

        no_improve_str = (
            f"  sin mejora {epochs_no_improve}/{patience}"
            if patience > 0 and epochs_no_improve > 0 else ""
        )
        print(
            f"{epoch:>6d} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
            f"{current_lr:>9.2e} | {_fmt_seconds(epoch_time):>8}"
            f"{no_improve_str}"
        )

        # ── Checkpoint si mejora ───────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_loss,
                            checkpoints_dir / "ssd_best.pth")
            print(f"         -> Mejor val_loss={val_loss:.4f}  (checkpoint guardado)")
        else:
            epochs_no_improve += 1

        # ── Checkpoint periodico ──────────────────────────────────────────
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,
                            checkpoints_dir / f"ssd_epoch_{epoch:03d}.pth")

        # ── Early stopping ─────────────────────────────────────────────────
        if patience > 0 and epochs_no_improve >= patience:
            print(f"\nEarly stopping: {patience} epocas sin mejora. Parando en epoca {epoch}.")
            history["stopped_epoch"] = epoch
            break
    else:
        history["stopped_epoch"] = num_epochs

    # ── Metricas finales ──────────────────────────────────────────────────────
    total_time       = time.time() - train_start
    best_epoch       = history["val_loss"].index(min(history["val_loss"])) + 1
    early_stopped    = epochs_no_improve >= patience if patience > 0 else False

    peak_gpu_mem = None
    if device.type == "cuda":
        peak_gpu_mem = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    ckpt_path  = checkpoints_dir / "ssd_best.pth"
    ckpt_size  = round(ckpt_path.stat().st_size / 1e6, 1) if ckpt_path.exists() else None

    # ── Construir log completo ────────────────────────────────────────────────
    training_log = {
        "metadata": {
            "date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device":        gpu_info["device"],
            "gpu_name":      gpu_info["gpu_name"],
            "vram_total_gb": gpu_info["vram_total_gb"],
            "cuda_version":  gpu_info.get("cuda_version"),
            "torch_version": torch.__version__,
        },
        "model": {
            "architecture":  "SSD300-VGG16",
            "backbone":      "VGG16 (ImageNet pretrained)",
            "num_classes":   NUM_CLASSES_SSD,
            "params":        params,
            "checkpoint_mb": ckpt_size,
        },
        "hyperparameters": {
            "lr_head":             lr,
            "lr_backbone":         round(lr * SSD_BACKBONE_LR_FACTOR, 8),
            "backbone_lr_factor":  SSD_BACKBONE_LR_FACTOR,
            "batch_size":          batch_size,
            "optimizer":           "SGD",
            "momentum":            SSD_MOMENTUM,
            "weight_decay":        SSD_WEIGHT_DECAY,
            "scheduler":           "MultiStepLR",
            "lr_milestones":       SSD_LR_MILESTONES,
            "lr_gamma":            SSD_LR_GAMMA,
            "grad_clip_norm":      SSD_GRAD_CLIP_NORM,
            "max_epochs":          num_epochs,
            "patience":            patience,
        },
        "preprocessing": {
            "train":         [
                "RandomHorizontalFlip(p=0.5)",
                "ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)",
                "GaussianBlur(kernel=3x3, p=0.2)",
                "ToTensor [0,255]->[0,1]",
            ],
            "val_test":      ["ToTensor [0,255]->[0,1]"],
            "normalization": "ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) — interna al modelo",
        },
        "training": {
            "epochs_run":              history["stopped_epoch"],
            "best_epoch":              best_epoch,
            "best_val_loss":           round(best_val_loss, 6),
            "early_stopping_triggered": early_stopped,
            "total_time_s":            round(total_time, 2),
            "total_time_formatted":    _fmt_seconds(total_time),
            "avg_epoch_time_s":        round(total_time / history["stopped_epoch"], 2),
            "peak_gpu_memory_gb":      peak_gpu_mem,
        },
        "per_epoch": per_epoch_log,
    }

    # ── Guardar log y mostrar resumen ─────────────────────────────────────────
    log_path = checkpoints_dir / "ssd_training_log.json"
    save_training_log(training_log, log_path)
    print_training_summary(training_log)

    history["training_log"] = training_log
    return model, history
