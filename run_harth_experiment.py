# run_harth_experiments.py
import yaml
from cv_engine import run_cv
from baseline_engine import run_baseline


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("config.yaml")

    dataset_name = cfg["dataset"]["name"]
    npz_path = cfg["dataset"]["npz_path"]

    device = cfg["experiment"].get("device", "cuda")
    models = cfg["experiment"]["models"]
    cv_types = cfg["experiment"]["cv_types"]

    results_root = cfg["paths"]["results_root"][dataset_name]
    baseline_root = cfg["paths"]["baseline_root"][dataset_name]

    baseline_enabled = cfg["baseline"].get("enabled", False)

    for model_name in models:
        print("=" * 80)
        print(f"Running {dataset_name} experiments for model: {model_name}")
        print("=" * 80)

        # 1) Baseline (unchanged logic)
        if baseline_enabled:
            baseline_dir = f"{baseline_root}/baseline_results_{model_name}_{dataset_name.lower()}"
            print(f"\n[Baseline] {model_name} → {baseline_dir}")
            run_baseline(
                npz_path,
                baseline_dir,
                model_name=model_name,
                device=device
            )

        # 2) Cross-validation
        for cv_type in cv_types:
            print(f"\n[CV] {cv_type} – {model_name}")
            run_cv(
                npz_path,
                results_root,
                cv_type=cv_type,
                model_name=model_name,
                device=device
            )


if __name__ == "__main__":
    main()
