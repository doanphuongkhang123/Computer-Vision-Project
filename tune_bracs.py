import argparse
import copy
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dump_yaml(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def safe_name(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")


def merge_config(base, overrides):
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        merged[key] = value
    return merged


def grid_trials(grid):
    keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in keys)):
        overrides = dict(zip(keys, values))
        name = "__".join(f"{key}-{safe_name(value)}" for key, value in overrides.items())
        yield {"name": name, "overrides": overrides}


def expand_trials(tuning_cfg):
    trials = []
    for trial in tuning_cfg.get("trials", []):
        trials.append({
            "name": trial["name"],
            "overrides": trial.get("overrides", {}),
        })
    if tuning_cfg.get("grid"):
        trials.extend(grid_trials(tuning_cfg["grid"]))
    return trials


def read_best_metric(metrics_csv, metric_name, mode):
    if not metrics_csv.exists():
        return None
    df = pd.read_csv(metrics_csv)
    if df.empty or metric_name not in df.columns:
        return None
    values = df[metric_name].dropna()
    if values.empty:
        return None
    idx = values.idxmax() if mode == "max" else values.idxmin()
    row = df.loc[idx].to_dict()
    row["selected_epoch"] = int(df.loc[idx, "epoch"]) if "epoch" in df.columns else ""
    row["selected_metric"] = float(df.loc[idx, metric_name])
    return row


def build_trial_config(base_cfg, tuning_cfg, trial, seed, trial_index):
    overrides = dict(tuning_cfg.get("common_overrides", {}))
    overrides.update(trial.get("overrides", {}))
    overrides["seed"] = int(seed)
    overrides["run_test_after_train"] = False

    trial_name = safe_name(trial["name"])
    if len(tuning_cfg.get("seeds", [])) > 1:
        trial_name = f"{trial_name}__seed-{seed}"

    cfg = merge_config(base_cfg, overrides)
    output_root = Path(tuning_cfg.get("output_dir", "results/tuning"))
    cfg["results_dir"] = str(output_root / f"{trial_index:03d}_{trial_name}")
    cfg["save_name"] = trial_name
    cfg["wandb_run_name"] = f"{tuning_cfg.get('wandb_run_prefix', 'tune')}_{trial_name}"
    return cfg, trial_name


def main():
    parser = argparse.ArgumentParser(description="Sequential hyperparameter tuning for BRACS WSI training.")
    parser.add_argument("--tuning-config", default="configs/bracs_tuning.yaml")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--trial", action="append", default=[], help="Run only trial name(s) matching this value.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun trials even if metrics.csv exists.")
    args = parser.parse_args()

    tuning_path = Path(args.tuning_config)
    tuning_cfg = load_yaml(tuning_path)
    base_path = Path(tuning_cfg.get("base_config", "configs/bracs_server.yaml"))
    if not base_path.is_absolute():
        base_path = tuning_path.parent.parent / base_path if tuning_path.parent.name == "configs" else Path(base_path)
    base_cfg = load_yaml(base_path)

    metric = tuning_cfg.get("metric", "val_f1_macro")
    mode = tuning_cfg.get("mode", "max")
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'")

    seeds = tuning_cfg.get("seeds", [base_cfg.get("seed", 1027)])
    trials = expand_trials(tuning_cfg)
    if args.trial:
        selected = set(args.trial)
        trials = [trial for trial in trials if trial["name"] in selected]
    if args.max_trials is not None:
        trials = trials[: args.max_trials]
    if not trials:
        raise ValueError("No tuning trials selected.")

    output_root = Path(tuning_cfg.get("output_dir", "results/tuning"))
    summary_path = output_root / tuning_cfg.get("summary_name", "tuning_summary.csv")
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    trial_index = 0

    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    for trial in trials:
        for seed in seeds:
            trial_index += 1
            cfg, trial_name = build_trial_config(base_cfg, tuning_cfg, trial, seed, trial_index)
            result_dir = Path(cfg["results_dir"])
            trial_cfg_path = result_dir / "config.yaml"
            metrics_csv = result_dir / "metrics.csv"
            dump_yaml(cfg, trial_cfg_path)

            print(f"\n=== Trial {trial_index}: {trial_name} seed={seed} ===", flush=True)
            print(f"Config: {trial_cfg_path}", flush=True)
            if args.dry_run:
                continue
            if metrics_csv.exists() and not args.force:
                print(f"Skipping existing trial: {metrics_csv}", flush=True)
            else:
                subprocess.run(
                    [sys.executable, "train_bracs.py", "--config", str(trial_cfg_path)],
                    check=True,
                    env=env,
                )

            best = read_best_metric(metrics_csv, metric, mode)
            row = {
                "trial": trial_name,
                "seed": seed,
                "results_dir": str(result_dir),
                "config": str(trial_cfg_path),
                "metric": metric,
                "mode": mode,
            }
            row.update(trial.get("overrides", {}))
            if best:
                row.update({
                    "selected_epoch": best.get("selected_epoch"),
                    "selected_metric": best.get("selected_metric"),
                    "val_loss": best.get("val_loss"),
                    "val_accuracy": best.get("val_accuracy"),
                    "val_f1_macro": best.get("val_f1_macro"),
                    "val_auc_macro": best.get("val_auc_macro"),
                })
            summary_rows.append(row)
            summary = pd.DataFrame(summary_rows)
            ascending = mode == "min"
            if "selected_metric" in summary.columns:
                summary = summary.sort_values("selected_metric", ascending=ascending, na_position="last")
            summary.to_csv(summary_path, index=False)

    print(f"\nWrote tuning summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
