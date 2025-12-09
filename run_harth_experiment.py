# run_harth_experiments.py
from cv_engine import run_cv
from baseline_engine import run_baseline

NPZ_PATH = r"D:/DKE/KMD/KMD_LOSO/harth_windows.npz"

# keep HARTH results separate from UCI HAR:
RESULTS_ROOT = r"D:/DKE/KMD/KMD_LOSO/results_harth"

# separate baseline dirs per MODEL for HARTH
BASELINE_ROOT = r"D:/DKE/KMD/KMD_LOSO"

DEVICE = "cuda"   # or "cpu" if you don’t have a GPU

CV_TYPES = ["LOSO"]
MODELS = ["LSTM"]
#"Stratified" "GroupKFold" "LOSO"


def main():
    for model_name in MODELS:
        print("=" * 80)
        print(f"Running HARTH experiments for model: {model_name}")
        print("=" * 80)

        # # 1) Baseline (simple train/test split) – like before
        # baseline_dir = f"{BASELINE_ROOT}/baseline_results_{model_name}_harth"
        # print(f"\n[Baseline] {model_name} → {baseline_dir}")
        # run_baseline(
        #     NPZ_PATH,
        #     baseline_dir,
        #     model_name=model_name,
        #     device=DEVICE
        # )

        # 2) Cross-validation runs
        for cv_type in CV_TYPES:
            print(f"\n[CV] {cv_type} – {model_name}")
            run_cv(
                NPZ_PATH,
                RESULTS_ROOT,
                cv_type=cv_type,
                model_name=model_name,
                device=DEVICE
            )


if __name__ == "__main__":
    main()
