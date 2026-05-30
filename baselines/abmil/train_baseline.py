import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from models.mil_baselines import build_mil_baseline
from utils.bracs_dataset import BracsWSIFeatureDataset, collate_wsi_features

try:
    import wandb
except Exception:
    wandb = None


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        print(
            "[WARN] For stricter CUDA reproducibility, run with "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python. "
            "The included run scripts set this automatically.",
            flush=True,
        )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def optional_int(value):
    if value in (None, "", "null", "None"):
        return None
    return int(value)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def setup_wandb(cfg):
    if not bool(cfg.get("wandb", False)):
        return None
    if wandb is None:
        print("[WARN] wandb is enabled in config but not installed. Continuing without wandb.", flush=True)
        return None

    token = cfg.get("wandb_api_key") or os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_TOKEN")
    key_file = Path(cfg.get("wandb_api_key_file", "wandb_api_key.txt"))
    if not token and key_file.exists():
        token = key_file.read_text().strip()

    if token:
        wandb.login(key=token)
    return wandb.init(
        project=cfg.get("wandb_project", "bracs_wsi_classification"),
        name=cfg.get("wandb_run_name") or cfg.get("save_name", "bracs_run"),
        config=cfg,
    )


def wandb_log(run, metrics, prefix, epoch=None):
    if run is None:
        return
    payload = {
        f"{prefix}/loss": metrics.get("loss"),
        f"{prefix}/accuracy": metrics.get("accuracy"),
        f"{prefix}/f1_macro": metrics.get("f1_macro"),
        f"{prefix}/auc_macro": metrics.get("auc_macro"),
        f"{prefix}/precision_macro": metrics.get("precision_macro"),
        f"{prefix}/recall_macro": metrics.get("recall_macro"),
    }
    if "lr" in metrics:
        payload[f"{prefix}/lr"] = metrics.get("lr")
    for key, value in metrics.items():
        if key.startswith(("pred_count_", "target_count_", "mean_prob_", "auc_class_", "f1_class_", "precision_class_", "recall_class_")):
            payload[f"{prefix}/{key}"] = value
    if epoch is not None:
        payload["epoch"] = epoch
    run.log(payload)


def make_model(cfg, device):
    return build_mil_baseline(cfg).to(device)


def extract_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def forward_logits(model, feats, attn_mask, epoch=0, stride=1):
    return extract_logits(model(feats, attn_mask, epoch=epoch, stride=stride))


def forward_train_loss(model, feats, attn_mask, labels, loss_fn, epoch=0, stride=1):
    if hasattr(model, "forward_with_loss"):
        outputs = model.forward_with_loss(
            feats,
            attn_mask,
            labels,
            epoch=epoch,
            stride=stride,
            loss_fn=loss_fn,
        )
        if isinstance(outputs, dict):
            return outputs["logits"], outputs["loss"]
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            return outputs[0], outputs[1]
    logits = forward_logits(model, feats, attn_mask, epoch=epoch, stride=stride)
    return logits, loss_fn(logits, labels)


def make_loader(csv_path, cfg, training):
    df = pd.read_csv(csv_path)
    seed_offset = 0 if training else 10000
    generator = torch.Generator()
    generator.manual_seed(int(cfg["seed"]) + seed_offset)
    dataset = BracsWSIFeatureDataset(
        df,
        feats_dir="",
        max_patches=optional_int(cfg["max_patches"] if training else cfg["eval_max_patches"]),
        sampling=cfg["train_sampling"] if training else cfg["eval_sampling"],
        training=training,
        normalize=bool(cfg.get("normalize_features", False)),
        feature_dim=int(cfg["hidden_size"]),
    )
    sampler = None
    shuffle = training
    if training and bool(cfg.get("balanced_sampler", False)):
        counts = df["label_idx"].astype(int).value_counts().sort_index().to_dict()
        sampler_mode = cfg.get("sampler_weighting", "inverse")
        sample_weights = sampler_weights_from_csv(csv_path, mode=sampler_mode)
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
        print(f"Using balanced sampler mode={sampler_mode} counts={counts}", flush=True)
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg["num_workers"]) > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=collate_wsi_features,
    )


def class_weights_from_csv(csv_path, num_classes, mode="none"):
    if mode in (None, "", "none", "None"):
        return None
    df = pd.read_csv(csv_path)
    counts = np.bincount(df["label_idx"].astype(int).to_numpy(), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if mode == "inverse":
        weights = 1.0 / counts
    elif mode == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown class_weighting mode: {mode}")
    weights = weights * (num_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def sampler_weights_from_csv(csv_path, mode="inverse"):
    df = pd.read_csv(csv_path)
    labels = df["label_idx"].astype(int).to_numpy()
    counts = np.bincount(labels).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if mode == "inverse":
        class_weights = 1.0 / counts
    elif mode == "inverse_sqrt":
        class_weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown sampler weighting mode: {mode}")
    return [float(class_weights[int(label)]) for label in labels]


def class_counts_from_csv(csv_path, num_classes):
    df = pd.read_csv(csv_path)
    return np.bincount(df["label_idx"].astype(int).to_numpy(), minlength=num_classes)


def format_class_counts(counts):
    return " ".join(f"{idx}={int(count)}" for idx, count in enumerate(counts))


def print_split_summary(cfg):
    num_classes = int(cfg["num_classes"])
    for split, csv_key in (("train", "train_csv"), ("val", "val_csv"), ("test", "test_csv")):
        counts = class_counts_from_csv(cfg[csv_key], num_classes)
        print(f"{split.upper()} target_counts: {format_class_counts(counts)}", flush=True)


def resolve_class_weighting_mode(cfg):
    mode = cfg.get("class_weighting", "none")
    if mode in (None, "", "none", "None"):
        return "none"
    if bool(cfg.get("balanced_sampler", False)) and not bool(cfg.get("allow_sampler_and_class_weights", False)):
        print(
            "[WARN] balanced_sampler=true and class_weighting is also enabled. "
            "Disabling class-weighted loss for this run to avoid overcorrecting class priors. "
            "Set allow_sampler_and_class_weights=true only if you intentionally want both.",
            flush=True,
        )
        return "none"
    return mode


def metric_is_better(metric_name, value, best_value):
    if metric_name == "loss":
        return value < best_value
    return value > best_value


def initial_best_value(metric_name):
    if metric_name == "loss":
        return float("inf")
    return -float("inf")


def make_scheduler(optimizer, cfg):
    scheduler_name = str(cfg.get("scheduler", "none")).lower()
    if scheduler_name in ("", "none"):
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(cfg["max_epochs"])),
            eta_min=float(cfg.get("min_lr", 0.0)),
        )
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg.get("scheduler_factor", 0.5)),
            patience=int(cfg.get("scheduler_patience", 5)),
        )
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def cuda_memory_text(device):
    if device.type != "cuda":
        return ""
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    return f" gpu_alloc={allocated:.2f}GB gpu_reserved={reserved:.2f}GB gpu_peak={max_allocated:.2f}GB"


def compute_metrics(targets, probs, num_classes):
    preds = probs.argmax(axis=1)
    f1_per_class = f1_score(targets, preds, average=None, labels=list(range(num_classes)), zero_division=0)
    precision_per_class = precision_score(targets, preds, average=None, labels=list(range(num_classes)), zero_division=0)
    recall_per_class = recall_score(targets, preds, average=None, labels=list(range(num_classes)), zero_division=0)
    out = {
        "accuracy": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
        "precision_macro": precision_score(targets, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(targets, preds, average="macro", zero_division=0),
    }
    auc_values = []
    for idx in range(num_classes):
        binary_targets = (targets == idx).astype(np.int64)
        if binary_targets.min() == binary_targets.max():
            out[f"auc_class_{idx}"] = float("nan")
            continue
        try:
            auc_value = roc_auc_score(binary_targets, probs[:, idx])
            out[f"auc_class_{idx}"] = float(auc_value)
            auc_values.append(auc_value)
        except ValueError:
            out[f"auc_class_{idx}"] = float("nan")
    out["auc_macro"] = float(np.mean(auc_values)) if auc_values else float("nan")
    pred_counts = np.bincount(preds, minlength=num_classes)
    target_counts = np.bincount(targets, minlength=num_classes)
    mean_probs = probs.mean(axis=0)
    for idx in range(num_classes):
        out[f"pred_count_{idx}"] = int(pred_counts[idx])
        out[f"target_count_{idx}"] = int(target_counts[idx])
        out[f"mean_prob_{idx}"] = float(mean_probs[idx])
        out[f"f1_class_{idx}"] = float(f1_per_class[idx])
        out[f"precision_class_{idx}"] = float(precision_per_class[idx])
        out[f"recall_class_{idx}"] = float(recall_per_class[idx])
    return out


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, cfg=None, epoch=0, phase="train"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_targets = []
    all_probs = []
    amp_enabled = bool(cfg.get("amp", True)) and device.type == "cuda"
    start_time = time.time()
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"{phase} epoch {epoch}",
        dynamic_ncols=True,
        leave=True,
    )
    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, (feats, labels, attn_mask) in enumerate(progress, 1):
        batch_tokens = int((~attn_mask.bool()).sum().item())
        max_tokens = int((~attn_mask.bool()).sum(dim=1).max().item())
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                if training:
                    logits, loss = forward_train_loss(
                        model,
                        feats,
                        attn_mask,
                        labels,
                        loss_fn,
                        epoch=epoch,
                        stride=1,
                    )
                else:
                    logits = forward_logits(model, feats, attn_mask, epoch=0, stride=1)
                    loss = loss_fn(logits, labels)

            if training:
                scaled_loss = loss / grad_accum_steps
                scaler.scale(scaled_loss).backward()
                if step % grad_accum_steps == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())
        all_targets.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(torch.softmax(logits.detach(), dim=1).cpu().numpy())
        elapsed = time.time() - start_time
        avg_loss = total_loss / step
        postfix = {
            "loss": f"{avg_loss:.4f}",
            "tokens": max_tokens,
            "elapsed": f"{elapsed:.0f}s",
        }
        if device.type == "cuda":
            postfix.update({
                "alloc_gb": f"{torch.cuda.memory_allocated(device) / 1024**3:.2f}",
                "peak_gb": f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f}",
            })
        progress.set_postfix(postfix)

    probs = np.asarray(all_probs)
    metrics = compute_metrics(np.asarray(all_targets), probs, int(cfg["num_classes"]))
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def save_predictions(model, loader, cfg, device, output_csv):
    model.eval()
    rows = []
    class_names = cfg.get("class_names", [str(i) for i in range(int(cfg["num_classes"]))])
    with torch.no_grad():
        for feats, labels, attn_mask in loader:
            feats = feats.to(device)
            labels = labels.to(device)
            attn_mask = attn_mask.to(device)
            logits = forward_logits(model, feats, attn_mask, epoch=0, stride=1)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            for y, pred, prob in zip(labels.cpu().tolist(), preds.cpu().tolist(), probs.cpu().numpy()):
                row = {"target": y, "pred": pred, "target_name": class_names[y], "pred_name": class_names[pred]}
                for idx, name in enumerate(class_names):
                    row[f"prob_{str(name).lower()}"] = float(prob[idx])
                rows.append(row)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train a BRACS MIL baseline on WSI feature bags.")
    parser.add_argument("--config", default="configs/bracs_server.yaml")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    run = setup_wandb(cfg)
    print_split_summary(cfg)
    if bool(cfg.get("balanced_sampler", False)):
        print("Using balanced sampler for training batches.", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    results_dir = Path(cfg["results_dir"])
    (results_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (results_dir / "predictions").mkdir(parents=True, exist_ok=True)

    model = make_model(cfg, device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)

    train_loader = make_loader(cfg["train_csv"], cfg, training=True)
    val_loader = make_loader(cfg["val_csv"], cfg, training=False)

    if args.eval_only:
        test_loader = make_loader(cfg["test_csv"], cfg, training=False)
        metrics = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device, cfg=cfg, phase="test")
        print(f"TEST {metrics}", flush=True)
        wandb_log(run, metrics, "test")
        save_predictions(model, test_loader, cfg, device, results_dir / "predictions" / "test_predictions.csv")
        if run is not None:
            run.finish()
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = make_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(cfg.get("amp", True)))
    class_weighting_mode = resolve_class_weighting_mode(cfg)
    class_weights = class_weights_from_csv(
        cfg["train_csv"],
        int(cfg["num_classes"]),
        mode=class_weighting_mode,
    )
    if class_weights is not None:
        print(f"Using class weights: {class_weights.tolist()}", flush=True)
        class_weights = class_weights.to(device)
    train_loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    eval_loss_fn = nn.CrossEntropyLoss()

    selection_metric = str(cfg.get("selection_metric", "f1_macro"))
    if selection_metric not in ("loss", "accuracy", "f1_macro", "auc_macro"):
        raise ValueError(f"Unsupported selection_metric: {selection_metric}")
    best_main = initial_best_value(selection_metric)
    best_loss = float("inf")
    best_f1 = -float("inf")
    best_acc = -float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(int(cfg["max_epochs"])):
        print(f"Starting epoch {epoch}", flush=True)
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model, train_loader, train_loss_fn, device,
            optimizer=optimizer, scaler=scaler, cfg=cfg, epoch=epoch, phase="train"
        )
        val_metrics = run_epoch(model, val_loader, eval_loss_fn, device, cfg=cfg, epoch=epoch, phase="val")
        train_metrics["lr"] = current_lr
        val_metrics["lr"] = current_lr
        wandb_log(run, train_metrics, "train", epoch)
        wandb_log(run, val_metrics, "val", epoch)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(results_dir / "metrics.csv", index=False)
        print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_macro']:.4f}", flush=True)
        print(
            "VAL pred_counts: "
            f"0={val_metrics.get('pred_count_0')} "
            f"1={val_metrics.get('pred_count_1')} "
            f"2={val_metrics.get('pred_count_2')} | "
            "target_counts: "
            f"0={val_metrics.get('target_count_0')} "
            f"1={val_metrics.get('target_count_1')} "
            f"2={val_metrics.get('target_count_2')} | "
            "mean_probs: "
            f"0={val_metrics.get('mean_prob_0'):.3f} "
            f"1={val_metrics.get('mean_prob_1'):.3f} "
            f"2={val_metrics.get('mean_prob_2'):.3f}",
            flush=True,
        )

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_metrics": val_metrics}, results_dir / "checkpoints" / "best_loss.pth")
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_metrics": val_metrics}, results_dir / "checkpoints" / "best_f1.pth")
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_metrics": val_metrics}, results_dir / "checkpoints" / "best_acc.pth")

        main_value = val_metrics[selection_metric]
        if metric_is_better(selection_metric, main_value, best_main):
            best_main = main_value
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_metrics": val_metrics}, results_dir / "checkpoints" / "best.pth")
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["patience"]):
                print(f"Early stopping at epoch {epoch}", flush=True)
                break
        if scheduler is not None:
            if str(cfg.get("scheduler", "none")).lower() == "plateau":
                scheduler.step(val_metrics.get("f1_macro", 0.0))
            else:
                scheduler.step()

    best = torch.load(results_dir / "checkpoints" / "best.pth", map_location=device)
    print(
        f"Evaluating best checkpoint selected by val_{selection_metric}: "
        f"epoch={best.get('epoch')} val_metrics={best.get('val_metrics')}",
        flush=True,
    )
    model.load_state_dict(best["model"])
    if not bool(cfg.get("run_test_after_train", True)):
        print("Skipping test evaluation because run_test_after_train=false.", flush=True)
        if run is not None:
            run.finish()
        return
    test_loader = make_loader(cfg["test_csv"], cfg, training=False)
    test_metrics = run_epoch(model, test_loader, eval_loss_fn, device, cfg=cfg, phase="test")
    print(f"TEST {test_metrics}", flush=True)
    wandb_log(run, test_metrics, "test")
    save_predictions(model, test_loader, cfg, device, results_dir / "predictions" / "test_predictions.csv")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
