import argparse
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dump_yaml(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Create a full final-training config from a tuning trial config.")
    parser.add_argument("--trial-dir", required=True, help="Trial result directory containing config.yaml.")
    parser.add_argument("--output", default="configs/bracs_final_from_tuning.yaml")
    parser.add_argument("--results-dir", default="results/bracs_final_from_tuning")
    parser.add_argument("--run-name", default="bracs_final_from_tuning")
    parser.add_argument("--full-wsi", action="store_true", help="Set max_patches/eval_max_patches to null.")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    trial_dir = Path(args.trial_dir)
    cfg_path = trial_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Trial config not found: {cfg_path}")

    cfg = load_yaml(cfg_path)
    cfg["results_dir"] = args.results_dir
    cfg["save_name"] = args.run_name
    cfg["wandb_run_name"] = args.run_name
    cfg["run_test_after_train"] = True
    cfg["max_epochs"] = int(args.max_epochs)
    cfg["patience"] = int(args.patience)
    if args.full_wsi:
        cfg["max_patches"] = None
        cfg["eval_max_patches"] = None

    output = Path(args.output)
    dump_yaml(cfg, output)
    print(f"Wrote final config: {output}")


if __name__ == "__main__":
    main()
