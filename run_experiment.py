# run_experiment.py
from cv_engine import run_cv
from baseline_engine import run_baseline
# run_cv("D:/DKE/KMD/KMD_LOSO/ucihar_raw_merged.npz", "D:/DKE/KMD/results", cv_type="LOSO", model_name="CNN", device="cuda")

if __name__ == "__main__":
    # run_cv(
    #     "D:/DKE/KMD/KMD_LOSO/ucihar_raw_merged.npz",
    #     "D:/DKE/KMD/KMD_LOSO/results",
    #     cv_type="GroupKFold",
    #     model_name="CNNLSTM",
    #     device="cuda"
    # )
    npz_path = "D:/DKE/KMD/KMD_LOSO/ucihar_raw_merged.npz"
    run_baseline(
        npz_path,
        "D:/DKE/KMD/KMD_LOSO/baseline_results_LSTM",
        model_name="LSTM",
        device="cuda"
    )
