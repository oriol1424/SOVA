import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('colab_train.ipynb', encoding='utf-8'))

# ─── CELL 10: check JSON al inicio, study en else ────────────────────────────
new_cell10 = (
    "# ─── Parámetros de la búsqueda ────────────────────────────────────────────\n"
    "HPO_TRIALS  = 12    # combinaciones a probar (suficiente para que TPE aprenda)\n"
    "HPO_EPOCHS  = 4     # épocas por trial — calibrado para T4 con ≤3.5 h GPU disponibles\n"
    "HPO_TIMEOUT = 2700  # 45 min máximo → deja ~2.5 h para el entrenamiento final\n"
    "\n"
    "# Espacio de búsqueda\n"
    "#   lr:         log-uniforme [1e-4, 5e-2]  (rango habitual para SGD)\n"
    "#   batch_size: {4, 8, 16}  (limitado por VRAM de T4 ~15 GB)\n"
    "#\n"
    "# Estimación de tiempos con T4 y 2605 imágenes:\n"
    "#   batch_size=4  → ~8 min/trial  |  batch_size=8 → ~5 min  |  batch_size=16 → ~3 min\n"
    "#   Promedio: ~5 min/trial × 12 trials ≈ 45–50 min total HPO ✓\n"
    "# ─────────────────────────────────────────────────────────────────────────────\n"
    "\n"
    "import json\n"
    "import optuna\n"
    "import logging\n"
    "from pathlib import Path\n"
    "from modelos.train_ssd import build_ssd_model, train_ssd\n"
    "from modelos.config import CHECKPOINTS_DIR\n"
    "\n"
    "# Silenciar logs internos de optuna para no saturar la salida\n"
    "optuna.logging.set_verbosity(optuna.logging.WARNING)\n"
    "\n"
    'HPO_CKPT_DIR     = CHECKPOINTS_DIR / "hpo_tmp"               # dir temporal (no sobreescribe ssd_best.pth)\n'
    'HPO_RESULTS_PATH = CHECKPOINTS_DIR / "hpo_best_params.json"  # resultados persistentes en Drive\n'
    "HPO_CKPT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "# ─── Si ya existe el resultado guardado, cargarlo y saltarse el HPO ─────────\n"
    "if HPO_RESULTS_PATH.exists():\n"
    "    with open(HPO_RESULTS_PATH) as f:\n"
    "        _saved = json.load(f)\n"
    '    best_lr         = _saved["lr"]\n'
    '    best_batch_size = _saved["batch_size"]\n'
    "    study           = None   # las celdas de visualización comprobarán esto\n"
    '    print("✓ Resultados HPO ya guardados — saltando búsqueda:")\n'
    "    print(f\"  lr={best_lr:.2e} | batch_size={best_batch_size} | val_loss={_saved['best_val_loss']:.4f}\")\n"
    '    print("\\n  Borra hpo_best_params.json si quieres repetir la búsqueda.")\n'
    "\n"
    "else:\n"
    "    def objective(trial: optuna.Trial) -> float:\n"
    '        """\n'
    "        Función objetivo para Optuna.\n"
    "        Cada trial entrena SSD durante HPO_EPOCHS épocas y devuelve el mejor val_loss.\n"
    '        """\n'
    '        lr         = trial.suggest_float("lr", 1e-4, 5e-2, log=True)\n'
    '        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])\n'
    "\n"
    '        print(f"\\nTrial {trial.number:>3} | lr={lr:.2e} | batch_size={batch_size}")\n'
    "\n"
    "        _, history = train_ssd(\n"
    "            num_epochs      = HPO_EPOCHS,\n"
    "            batch_size      = batch_size,\n"
    "            lr              = lr,\n"
    "            checkpoints_dir = HPO_CKPT_DIR,  # dir separado → no toca ssd_best.pth\n"
    "            save_every      = 9999,           # no guardar checkpoints periódicos\n"
    "            patience        = 0,              # sin early stopping: todas las épocas aportan señal\n"
    "        )\n"
    "\n"
    '        best_val = min(history["val_loss"])\n'
    '        print(f"         → mejor val_loss={best_val:.4f}")\n'
    "        return best_val\n"
    "\n"
    "\n"
    "    study = optuna.create_study(\n"
    '        direction  = "minimize",\n'
    "        sampler    = optuna.samplers.TPESampler(seed=42),\n"
    '        study_name = "ssd_hpo",\n'
    "    )\n"
    "\n"
    '    print(f"Iniciando HPO: {HPO_TRIALS} trials × {HPO_EPOCHS} épocas cada uno")\n'
    "    print(f\"Espacio: lr ∈ [1e-4, 5e-2]  |  batch_size ∈ {4, 8, 16}\")\n"
    '    print("─" * 55)\n'
    "\n"
    "    study.optimize(objective, n_trials=HPO_TRIALS, timeout=HPO_TIMEOUT)\n"
    "\n"
    '    print("\\n" + "═" * 55)\n'
    '    print("RESULTADOS HPO")\n'
    '    print("═" * 55)\n'
    '    print(f"Mejor val_loss : {study.best_value:.4f}")\n'
    "    print(f\"Mejor lr       : {study.best_params['lr']:.2e}\")\n"
    "    print(f\"Mejor batch    : {study.best_params['batch_size']}\")\n"
    '    print("═" * 55)\n'
)

# ─── CELL 11: guard if study is None ─────────────────────────────────────────
new_cell11 = (
    "# ─── Visualizar evolución del estudio ───────────────────────────────────────\n"
    "if study is None:\n"
    '    print("ℹ️  HPO cargado desde archivo — visualización no disponible (no hay objeto study).")\n'
    "else:\n"
    "    import matplotlib.pyplot as plt\n"
    "\n"
    '    trials_df = study.trials_dataframe()\n'
    '    trials_df = trials_df.sort_values("number")\n'
    "\n"
    "    fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
    "\n"
    "    # val_loss por trial\n"
    '    axes[0].plot(trials_df["number"], trials_df["value"], "o-", color="steelblue")\n'
    "    axes[0].axhline(study.best_value, color=\"green\", linestyle=\"--\", alpha=0.7,\n"
    "                    label=f\"Mejor: {study.best_value:.4f}\")\n"
    '    axes[0].set_title("Val loss por trial")\n'
    '    axes[0].set_xlabel("Trial"); axes[0].set_ylabel("Val loss")\n'
    "    axes[0].legend(); axes[0].grid(True, alpha=0.3)\n"
    "\n"
    "    # lr vs val_loss\n"
    '    lrs = [t.params["lr"] for t in study.trials]\n'
    "    vals = [t.value for t in study.trials]\n"
    '    axes[1].scatter(lrs, vals, c="steelblue", s=60, alpha=0.8)\n'
    '    axes[1].scatter([study.best_params["lr"]], [study.best_value],\n'
    '                    c="green", s=120, zorder=5, label="Mejor")\n'
    '    axes[1].set_xscale("log")\n'
    '    axes[1].set_title("lr vs val_loss")\n'
    '    axes[1].set_xlabel("lr (log)"); axes[1].set_ylabel("Val loss")\n'
    "    axes[1].legend(); axes[1].grid(True, alpha=0.3)\n"
    "\n"
    "    # batch_size vs val_loss (box plot)\n"
    "    bs_vals = {b: [] for b in [4, 8, 16]}\n"
    "    for t in study.trials:\n"
    '        bs_vals[t.params["batch_size"]].append(t.value)\n'
    '    axes[2].boxplot([bs_vals[b] for b in [4, 8, 16]], labels=["4", "8", "16"])\n'
    '    axes[2].set_title("batch_size vs val_loss")\n'
    '    axes[2].set_xlabel("batch_size"); axes[2].set_ylabel("Val loss")\n'
    '    axes[2].grid(True, alpha=0.3, axis="y")\n'
    "\n"
    "    plt.suptitle(f\"Optuna HPO — {len(study.trials)} trials\", fontsize=13)\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "\n"
    "    # Tabla de los 5 mejores trials\n"
    '    print("\\nTop-5 trials:")\n'
    '    top5 = trials_df.nsmallest(5, "value")[["number", "params_lr", "params_batch_size", "value"]]\n'
    '    top5.columns = ["trial", "lr", "batch_size", "val_loss"]\n'
    "    print(top5.to_string(index=False))\n"
)

# ─── CELL 12: guardar a JSON además de a variables ───────────────────────────
new_cell12 = (
    "# ─── Guardar mejores hiperparámetros (variable + disco) ────────────────────\n"
    "import json\n"
    "\n"
    "if study is not None:\n"
    '    best_lr         = study.best_params["lr"]\n'
    '    best_batch_size = study.best_params["batch_size"]\n'
    "\n"
    "    # Guardar en disco → las siguientes ejecuciones del notebook lo leerán\n"
    '    _params = {"lr": best_lr, "batch_size": best_batch_size, "best_val_loss": study.best_value}\n'
    "    with open(HPO_RESULTS_PATH, \"w\") as f:\n"
    "        json.dump(_params, f, indent=2)\n"
    "    print(f\"✓ Hiperparámetros guardados en disco: lr={best_lr:.2e} | batch_size={best_batch_size}\")\n"
    "    print(f\"  → {HPO_RESULTS_PATH}\")\n"
    "else:\n"
    "    # Ya cargados en celda anterior desde hpo_best_params.json\n"
    "    print(f\"ℹ️  Hiperparámetros ya listos (cargados desde archivo): lr={best_lr:.2e} | batch_size={best_batch_size}\")\n"
)

# Aplicar cambios (source puede ser string o lista; usamos string)
nb['cells'][10]['source'] = new_cell10
nb['cells'][11]['source'] = new_cell11
nb['cells'][12]['source'] = new_cell12

with open('colab_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('✓ Notebook parcheado correctamente')
