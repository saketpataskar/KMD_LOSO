# run_experiment.py
import yaml
from cv_engine import run_cv
from baseline_engine import run_baseline


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("config.yaml")

    npz_path = cfg["dataset"]["npz_path"]
    model_name = cfg["experiment"]["model_name"]
    device = cfg["experiment"].get("device", "cuda")
    out_dir = cfg["output"]["out_dir"]

    mode = cfg["experiment"]["mode"]

    if mode == "baseline":
        run_baseline(
            npz_path,
            out_dir,
            model_name=model_name,
            device=device
        )

    elif mode == "cv":
        cv_type = cfg["experiment"]["cv_type"]
        run_cv(
            npz_path,
            out_dir,
            cv_type=cv_type,
            model_name=model_name,
            device=device
        )

    else:
        raise ValueError(f"Unknown experiment mode: {mode}")


if __name__ == "__main__":
    main()
