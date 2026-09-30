"""
Generic Optuna hyperparameter tuner for Ultralytics models
(YOLO, YOLOWorld, RTDETR, and custom RTDETR YAMLs).

Every trial starts from a FRESH copy of the model. Training changes the weights
in place, so reusing one loaded model would make trial 2 start from trial 1's
result. You can pass either:
    - a loaded model:  YOLO("yolo12s.pt")           -> reloaded from its source each trial
    - a factory:       lambda: RTDETR("x.yaml").load("rtdetr-l.pt")
      (use a factory whenever the model needs extra steps after construction,
       e.g. .load() weight transfer, or it will be rebuilt without them)
"""

import gc
import json
import os
import random
from pathlib import Path

import optuna
import torch
import yaml
from ultralytics.engine.model import Model

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ------------------------------------------------------------------
# Train-subset YAML (works for every trainer, unlike `fraction=`)
# ------------------------------------------------------------------
def make_subset_yaml(data_yaml, fraction, out_dir, seed=0):
    """Write a data YAML whose train split is a fixed random `fraction` of the
    original train images. Val (and test) stay unchanged."""
    data_yaml = Path(data_yaml).resolve()
    d = yaml.safe_load(open(data_yaml))

    root = Path(d.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    imgs = []
    for t in d["train"] if isinstance(d["train"], list) else [d["train"]]:
        t = resolve(t)
        if t.is_dir():
            imgs += [str(f) for f in t.rglob("*") if f.suffix.lower() in IMG_EXT]
        else:  # .txt list of image paths
            for line in open(t):
                line = line.strip()
                if line:
                    p = Path(line)
                    imgs.append(str(p if p.is_absolute() else (t.parent / p).resolve()))

    imgs.sort()
    random.Random(seed).shuffle(imgs)
    subset = imgs[: max(1, int(len(imgs) * fraction))]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / "train_subset.txt"
    txt.write_text("\n".join(subset))

    d2 = dict(d)
    d2["path"] = str(root)
    d2["train"] = str(txt)
    for split in ("val", "test"):
        if d.get(split):
            v = d[split]
            d2[split] = [str(resolve(x)) for x in v] if isinstance(v, list) else str(resolve(v))

    out = out_dir / "subset_data.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(d2, f, sort_keys=False)
    print(f"[subset] {len(subset)}/{len(imgs)} train images -> {out}")
    return str(out)


def _as_factory(model):
    if isinstance(model, Model):
        cls, src = type(model), (model.ckpt_path or model.cfg)
        if not src:
            raise ValueError("Can't tell where this model was loaded from; pass a factory (lambda) instead.")
        return lambda: cls(src)
    if callable(model):
        return model
    raise TypeError("`model` must be an Ultralytics model or a zero-arg function returning one.")


# ------------------------------------------------------------------
# Main tuner
# ------------------------------------------------------------------
def tune_model(
    model,
    name,
    data_yaml,
    save_root,
    n_trials=30,
    epochs=10,
    fraction=0.2,
    imgsz=640,
    batch=16,
    optimizer_choices=("SGD", "Adam", "AdamW", "NAdam", "RAdam"),
    lr_range=(1e-4, 3e-3),
    train_kwargs=None,
    seed=0,
):
    """
    Tune one model with Optuna and return the study.

    model        : loaded Ultralytics model, or a zero-arg factory returning one
    name         : run name, e.g. "yolov12s" (one Optuna study + folder per name)
    data_yaml    : your normal data YAML (train = train, val = val)
    save_root    : folder for all tuning runs (e.g. on Google Drive)
    fraction     : share of TRAIN images used per trial (val is never trained on)
    optimizer_choices : optimizers Optuna picks from each trial (categorical param)
    lr_range     : lr0 search range; DETR-style models need lower than YOLO
    train_kwargs : extra args passed to .train(), e.g. {"trainer": MyTrainer}
    """
    factory = _as_factory(model)
    save_dir = os.path.join(save_root, name)
    os.makedirs(save_dir, exist_ok=True)
    subset_yaml = make_subset_yaml(data_yaml, fraction, save_dir, seed=seed)
    extra = dict(train_kwargs or {})

    def objective(trial):
        optimizer = trial.suggest_categorical("optimizer", list(optimizer_choices))
        params = dict(
            lr0=trial.suggest_float("lr0", *lr_range, log=True),
            lrf=trial.suggest_float("lrf", 0.01, 0.3, log=True),
            momentum=trial.suggest_float("momentum", 0.85, 0.95),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            warmup_epochs=trial.suggest_int("warmup_epochs", 1, 3),
            # box/cls/dfl loss gains. Ultralytics stock defaults are
            # box=7.5, cls=0.5, dfl=1.5 — ranges below bracket each default
            # rather than centering blindly on 0:
            #   box (5.0-10.0): box regression is the dominant loss term by
            #     design (default already 15x cls); this keeps it dominant
            #     while letting Optuna trade it off against cls/dfl instead
            #     of letting it run away and starve classification entirely.
            #   cls (0.3-1.0): lower bound stays above where cls signal gets
            #     drowned out by box+dfl; upper bound caps it at 2x default
            #     so classification doesn't dominate and hurt localization
            #     (matters more here since you're on a 2-class dataset where
            #     cls is already the easier sub-task).
            #   dfl (1.0-2.0): DFL sharpens the box-edge distribution; below
            #     ~1.0 boxes get blurrier, above ~2.0 it destabilizes early
            #     training — tighter range than box/cls because DFL is more
            #     sensitive to overweighting, and fraction=0.1 means each
            #     trial already trains on noisier, smaller batches.
            box=trial.suggest_float("box", 5.0, 10.0),
            cls=trial.suggest_float("cls", 0.3, 1.0),
            dfl=trial.suggest_float("dfl", 1.0, 2.0),
        )
        m = factory()

        def report(trainer):
            score = trainer.metrics.get("metrics/mAP50-95(B)", 0.0)
            trial.report(score, trainer.epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        m.add_callback("on_fit_epoch_end", report)

        try:
            r = m.train(
                data=subset_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                optimizer=optimizer, seed=seed, deterministic=False,
                project=save_dir, name=f"trial_{trial.number}", exist_ok=True,
                plots=False, verbose=False,
                **params, **extra,
            )
            metrics = r.results_dict
            with open(os.path.join(save_dir, f"trial_{trial.number}_metrics.json"), "w") as f:
                json.dump({"params": {**params, "optimizer": optimizer}, "metrics": metrics}, f, indent=4)
            return metrics["metrics/mAP50-95(B)"]
        finally:
            del m
            gc.collect()
            torch.cuda.empty_cache()

    study = optuna.create_study(
        direction="maximize",
        study_name=name,
        storage=f"sqlite:///{os.path.join(save_dir, 'optuna.db')}",
        load_if_exists=True,  # resumes after a Colab disconnect
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    done = len([t for t in study.trials if t.state.is_finished()])
    study.optimize(objective, n_trials=max(0, n_trials - done))

    best = {"model": name, "best_mAP50-95": study.best_value, "best_params": study.best_params}
    with open(os.path.join(save_dir, "best_params.json"), "w") as f:
        json.dump(best, f, indent=4)
    print(f"[{name}] best mAP50-95={study.best_value:.4f}  params={study.best_params}")
    return study


def summarize(save_root):
    """Print best result of every tuned model under save_root."""
    for p in sorted(Path(save_root).glob("*/best_params.json")):
        b = json.load(open(p))
        print(f"{b['model']:<20} {b['best_mAP50-95']:.4f}  {b['best_params']}")
