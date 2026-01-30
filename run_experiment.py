# run_experiment.py
import yaml
from cv_engine import run_cv
from baseline_engine import run_baseline


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("config.yaml")

    dataset = cfg["dataset"]["name"]
    npz_path = cfg["dataset"]["npz_path"]

    mode = cfg["experiment"]["mode"]          # baseline | cv
    model_name = cfg["experiment"]["model"]
    device = cfg["experiment"].get("device", "cuda")

    if mode == "baseline":
        out_dir = cfg["paths"]["baseline_root"][dataset] + f"/baseline_results_{model_name}_{dataset.lower()}"

        run_baseline(
            npz_path=npz_path,
            output_dir=out_dir,
            model_name=model_name,
            device=device
        )

    elif mode == "cv":
        cv_type = cfg["experiment"]["cv_type"]
        out_dir = cfg["paths"]["results_root"][dataset]

        run_cv(
            npz_path=npz_path,
            output_dir=out_dir,
            cv_type=cv_type,
            model_name=model_name,
            device=device
        )

    else:
        raise ValueError(f"Unknown experiment mode: {mode}")


if __name__ == "__main__":
    main()
