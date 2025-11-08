# # import pandas as pd
# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense, Dropout
# # from tensorflow.keras.callbacks import EarlyStopping
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.exceptions import DataConversionWarning
# # import warnings
# #
# # warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# #
# # def load_and_combine_data(base_path="UCI HAR Dataset/"):
# #     print("Loading and recombining data...")
# #     # 1. Load Feature Names
# #     # These will be the column headers for X_all
# #     features_df = pd.read_csv(
# #         base_path + 'features.txt',
# #         sep=' ',
# #         header=None,
# #         names=['id', 'name']
# #     )
# #     feature_names = features_df['name'].values
# #
# #     # 2. Load and Combine X (Features)
# #     X_train_df = pd.read_csv(
# #         base_path + 'train/X_train.txt',
# #         delim_whitespace=True,
# #         header=None,
# #         names=feature_names
# #     )
# #     X_test_df = pd.read_csv(
# #         base_path + 'test/X_test.txt',
# #         delim_whitespace=True,
# #         header=None,
# #         names=feature_names
# #     )
# #
# #     X_all = pd.concat([X_train_df, X_test_df]).reset_index(drop=True)
# #
# #     # 3. Load and Combine y (Labels)
# #     y_train_df = pd.read_csv(
# #         base_path + 'train/y_train.txt',
# #         header=None,
# #         names=['activity_id']
# #     )
# #     y_test_df = pd.read_csv(
# #         base_path + 'test/y_test.txt',
# #         header=None,
# #         names=['activity_id']
# #     )
# #     y_all = pd.concat([y_train_df, y_test_df]).reset_index(drop=True)
# #
# #     y_all['activity_id'] = y_all['activity_id'] - 1
# #
# #     subject_train_df = pd.read_csv(
# #         base_path + 'train/subject_train.txt',
# #         header=None,
# #         names=['subject_id']
# #     )
# #     subject_test_df = pd.read_csv(
# #         base_path + 'test/subject_test.txt',
# #         header=None,
# #         names=['subject_id']
# #     )
# #     subjects_all = pd.concat([subject_train_df, subject_test_df]).reset_index(drop=True)
# #
# #     print(f"Data combined. Total rows: {len(X_all)}")
# #     return X_all.values, y_all.values.ravel(), subjects_all.values.ravel()
# #
# #
# # def create_mlp_model(input_shape, num_classes=6):
# #     """
# #     Creates a simple Multi-Layer Perceptron (MLP) model.
# #     """
# #     model = Sequential([
# #         Dense(64, activation='relu', input_shape=(input_shape,)),
# #         Dropout(0.3),
# #         Dense(64, activation='relu'),
# #         Dropout(0.3),
# #         Dense(num_classes, activation='softmax')  # 6 classes
# #     ])
# #
# #     model.compile(
# #         optimizer='adam',
# #         loss='sparse_categorical_crossentropy',  # Use this loss for integer labels (0-5)
# #         metrics=['accuracy']
# #     )
# #     return model
# #
# #
# # def run_baseline_experiment(X_all, y_all, subjects_all):
# #     """
# #     Task 2: Runs the "unstable baseline" experiment.
# #     """
# #     print("Starting Task 2: Unstable Baseline Experiment...")
# #     all_subject_ids = np.unique(subjects_all)
# #     per_subject_test_scores = []
# #
# #     # 1. Outer Loop (LOSO Testing)
# #     for subject_i in all_subject_ids:
# #         print(f"\n--- Running Fold: TEST Subject {subject_i} ---")
# #
# #         # 2A. Outer Split (Create Test Set)
# #         test_indices = np.where(subjects_all == subject_i)[0]
# #         X_test, y_test = X_all[test_indices], y_all[test_indices]
# #
# #         # 2B. Create Training Pool (All other 29 subjects)
# #         training_pool_indices = np.where(subjects_all != subject_i)[0]
# #         subjects_in_pool = subjects_all[training_pool_indices]
# #
# #         # 2C. Inner "Problem" Split
# #         # Arbitrarily pick one subject from the pool for validation
# #         subject_j = subjects_in_pool[0]
# #         print(f"Using VALIDATION Subject {subject_j}")
# #
# #         validation_indices = training_pool_indices[np.where(subjects_in_pool == subject_j)[0]]
# #         X_val, y_val = X_all[validation_indices], y_all[validation_indices]
# #
# #         # The remaining 28 subjects are for training
# #         training_indices = training_pool_indices[np.where(subjects_in_pool != subject_j)[0]]
# #         X_train, y_train = X_all[training_indices], y_all[training_indices]
# #
# #         # 3. Scale and Train Model
# #         # Scale data based *only* on the training set
# #         scaler = StandardScaler()
# #         X_train = scaler.fit_transform(X_train)
# #         X_val = scaler.transform(X_val)
# #         X_test = scaler.transform(X_test)
# #
# #         model = create_mlp_model(input_shape=X_all.shape[1])
# #
# #         # This callback makes decisions based *only* on the single validation subject
# #         early_stopper = EarlyStopping(
# #             monitor='val_loss',
# #             patience=10,
# #             restore_best_weights=True
# #         )
# #
# #         model.fit(
# #             X_train, y_train,
# #             validation_data=(X_val, y_val),
# #             epochs=100,
# #             callbacks=[early_stopper],
# #             verbose=0  # Set to 1 if you want to see training progress per fold
# #         )
# #
# #         # 4. Evaluate and Store
# #         loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
# #         per_subject_test_scores.append(accuracy)
# #         print(f"Score for Test Subject {subject_i}: {accuracy:.4f}")
# #
# #     # 5. Analyze Final Results
# #     print("\n--- Experiment Complete ---")
# #     mean_accuracy = np.mean(per_subject_test_scores)
# #     std_accuracy = np.std(per_subject_test_scores)
# #
# #     print(f"Mean Test Accuracy: {mean_accuracy:.4f}")
# #     print(f"Std Dev of Accuracy: {std_accuracy:.4f}")
# #
# #     if std_accuracy > 0.1:  # You can set your own threshold
# #         print("\nConclusion: High standard deviation confirms the baseline is UNSTABLE.")
# #     else:
# #         print("\nConclusion: Low standard deviation. The problem may not be as severe as expected.")
# #
# #
# # def load_and_recombine_data(base_path='UCI HAR Dataset/'):
# #     """
# #     Task 1: Loads and recombines the UCI-HAR dataset.
# #
# #     --- UPDATED to fix duplicate column names and FutureWarning ---
# #     """
# #     print("Loading and recombining data...")
# #
# #     # 1. Load Feature Names
# #     features_df = pd.read_csv(
# #         base_path + 'features.txt',
# #         sep=' ',  # Use sep=' ' to read the file
# #         header=None,
# #         names=['id', 'name']
# #     )
# #
# #     # --- FIX for ValueError ---
# #     # Create unique names by prepending the ID, e.g., "1_tBodyAcc-mean()-X"
# #     feature_names = features_df['id'].astype(str) + "_" + features_df['name']
# #     feature_names = feature_names.values  # Get as a list/numpy array
# #
# #     # 2. Load and Combine X (Features)
# #     # --- FIX for FutureWarning: Use sep='\s+' ---
# #     X_train_df = pd.read_csv(
# #         base_path + 'train/X_train.txt',
# #         sep='\s+',  # Updated from delim_whitespace
# #         header=None,
# #         names=feature_names
# #     )
# #     X_test_df = pd.read_csv(
# #         base_path + 'test/X_test.txt',
# #         sep='\s+',  # Updated from delim_whitespace
# #         header=None,
# #         names=feature_names
# #     )
# #     X_all = pd.concat([X_train_df, X_test_df]).reset_index(drop=True)
# #
# #     # 3. Load and Combine y (Labels)
# #     y_train_df = pd.read_csv(
# #         base_path + 'train/y_train.txt',
# #         header=None,
# #         names=['activity_id']
# #     )
# #     y_test_df = pd.read_csv(
# #         base_path + 'test/y_test.txt',
# #         header=None,
# #         names=['activity_id']
# #     )
# #     y_all = pd.concat([y_train_df, y_test_df]).reset_index(drop=True)
# #
# #     # Keras/TensorFlow expects labels to be 0-indexed (0 to 5)
# #     y_all['activity_id'] = y_all['activity_id'] - 1
# #
# #     # 4. Load and Combine subjects
# #     subject_train_df = pd.read_csv(
# #         base_path + 'train/subject_train.txt',
# #         header=None,
# #         names=['subject_id']
# #     )
# #     subject_test_df = pd.read_csv(
# #         base_path + 'test/subject_test.txt',
# #         header=None,
# #         names=['subject_id']
# #     )
# #     subjects_all = pd.concat([subject_train_df, subject_test_df]).reset_index(drop=True)
# #
# #     print(f"Data combined. Total rows: {len(X_all)}")
# #     return X_all.values, y_all.values.ravel(), subjects_all.values.ravel()
# #
# # if __name__ == '__main__':
# #     # Task 1
# #     X_all, y_all, subjects_all = load_and_recombine_data()
# #
# #     # Task 2
# #     run_baseline_experiment(X_all, y_all, subjects_all)
#
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.preprocessing import StandardScaler
# from sklearn.exceptions import DataConversionWarning
# import warnings
# import random
#
# warnings.filterwarnings(action='ignore', category=DataConversionWarning)
#
#
# def load_and_recombine_data(base_path='UCI HAR Dataset/'):
#     """
#     Task 1: Loads, recombines, and saves a master CSV file.
#     """
#     print("Loading and recombining data...")
#
#     # 1. Load Feature Names
#     features_df = pd.read_csv(
#         base_path + 'features.txt',
#         sep=' ',
#         header=None,
#         names=['id', 'name']
#     )
#     # Create unique names, e.g., "1_tBodyAcc-mean()-X"
#     feature_names = features_df['id'].astype(str) + "_" + features_df['name']
#     feature_names = feature_names.values
#
#     # 2. Load and Combine X (Features)
#     X_train_df = pd.read_csv(
#         base_path + 'train/X_train.txt',
#         sep='\s+',
#         header=None,
#         names=feature_names
#     )
#     X_test_df = pd.read_csv(
#         base_path + 'test/X_test.txt',
#         sep='\s+',
#         header=None,
#         names=feature_names
#     )
#     X_all_df = pd.concat([X_train_df, X_test_df]).reset_index(drop=True)
#
#     # 3. Load and Combine y (Labels)
#     y_train_df = pd.read_csv(
#         base_path + 'train/y_train.txt',
#         header=None,
#         names=['activity_id']
#     )
#     y_test_df = pd.read_csv(
#         base_path + 'test/y_test.txt',
#         header=None,
#         names=['activity_id']
#     )
#     y_all_df = pd.concat([y_train_df, y_test_df]).reset_index(drop=True)
#
#     # Keras/TensorFlow expects labels to be 0-indexed (0 to 5)
#     y_all_df_keras = y_all_df.copy()
#     y_all_df_keras['activity_id'] = y_all_df_keras['activity_id'] - 1
#
#     # 4. Load and Combine subjects
#     subject_train_df = pd.read_csv(
#         base_path + 'train/subject_train.txt',
#         header=None,
#         names=['subject_id']
#     )
#     subject_test_df = pd.read_csv(
#         base_path + 'test/subject_test.txt',
#         header=None,
#         names=['subject_id']
#     )
#     subjects_all_df = pd.concat([subject_train_df, subject_test_df]).reset_index(drop=True)
#
#     # --- NEW: Create and save the master CSV file ---
#     print("Creating master_data.csv...")
#     # Use the original 1-6 labels for the master file, as they are more intuitive
#     master_df = pd.concat([subjects_all_df, y_all_df, X_all_df], axis=1)
#     master_df.to_csv('master_data.csv', index=False)
#     print("master_data.csv saved successfully.")
#     # --- End of new code ---
#
#     print(f"Data combined. Total rows: {len(X_all_df)}")
#     # Return the values needed for the experiment (with 0-5 labels)
#     return X_all_df.values, y_all_df_keras.values.ravel(), subjects_all_df.values.ravel()
#
#
# def create_mlp_model(input_shape, num_classes=6):
#     """
#     Creates a simple Multi-Layer Perceptron (MLP) model.
#     """
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(input_shape,)),
#         Dropout(0.3),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(num_classes, activation='softmax')  # 6 classes
#     ])
#
#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model
#
#
# def run_baseline_experiment(X_all, y_all, subjects_all):
#     """
#     Task 2: Runs the "unstable baseline" experiment (Random Validator)
#     and saves results to a file.
#     """
#     print("Starting Task 2: Unstable Baseline Experiment (Random Validator)...")
#     all_subject_ids = np.unique(subjects_all)
#     per_subject_test_scores = []
#
#     # --- NEW: Open a file to save results ---
#     with open('baseline_results.txt', 'w') as results_file:
#         results_file.write("--- Baseline Experiment Results (Unstable Validation) ---\n\n")
#
#         # 1. Outer Loop (LOSO Testing)
#         for subject_i in all_subject_ids:
#             fold_header = f"\n--- Running Fold: TEST Subject {subject_i} ---"
#             print(fold_header)
#             results_file.write(f"{fold_header}\n")
#
#             # 2A. Outer Split (Create Test Set)
#             test_indices = np.where(subjects_all == subject_i)[0]
#             X_test, y_test = X_all[test_indices], y_all[test_indices]
#
#             # 2B. Create Training Pool
#             training_pool_indices = np.where(subjects_all != subject_i)[0]
#             subjects_in_pool = subjects_all[training_pool_indices]
#
#             # 2C. Inner "Problem" Split
#             unique_subjects_in_pool = np.unique(subjects_in_pool)
#             subject_j = random.choice(unique_subjects_in_pool)
#             val_subject_str = f"Using RANDOM VALIDATION Subject {subject_j}"
#             print(val_subject_str)
#             results_file.write(f"{val_subject_str}\n")
#
#             validation_indices = training_pool_indices[np.where(subjects_in_pool == subject_j)[0]]
#             X_val, y_val = X_all[validation_indices], y_all[validation_indices]
#
#             training_indices = training_pool_indices[np.where(subjects_in_pool != subject_j)[0]]
#             X_train, y_train = X_all[training_indices], y_all[training_indices]
#
#             # 3. Scale and Train Model
#             scaler = StandardScaler()
#             X_train = scaler.fit_transform(X_train)
#             X_val = scaler.transform(X_val)
#             X_test = scaler.transform(X_test)
#
#             model = create_mlp_model(input_shape=X_all.shape[1])
#
#             early_stopper = EarlyStopping(
#                 monitor='val_loss',
#                 patience=10,
#                 restore_best_weights=True
#             )
#
#             model.fit(
#                 X_train, y_train,
#                 validation_data=(X_val, y_val),
#                 epochs=100,
#                 callbacks=[early_stopper],
#                 verbose=0
#             )
#
#             # 4. Evaluate and Store
#             loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
#             per_subject_test_scores.append(accuracy)
#             score_str = f"Score for Test Subject {subject_i}: {accuracy:.4f}"
#             print(score_str)
#             results_file.write(f"{score_str}\n")
#
#         # 5. Analyze Final Results
#         print("\n--- Experiment Complete ---")
#         results_file.write("\n\n--- Experiment Complete ---\n")
#
#         mean_accuracy = np.mean(per_subject_test_scores)
#         std_accuracy = np.std(per_subject_test_scores)
#
#         mean_str = f"Mean Test Accuracy: {mean_accuracy:.4f}"
#         std_str = f"Std Dev of Accuracy: {std_accuracy:.4f}"
#         print(mean_str)
#         print(std_str)
#         results_file.write(f"{mean_str}\n")
#         results_file.write(f"{std_str}\n")
#
#         if std_accuracy > 0.08:
#             conclusion = "\nConclusion: High standard deviation confirms the baseline is UNSTABLE."
#         else:
#             conclusion = "\nConclusion: Low standard deviation. The features are highly generalizable."
#
#         print(conclusion)
#         results_file.write(conclusion + "\n")
#
#     print("\nResults saved to baseline_results.txt")
#
#
# # --- Main execution ---
# if __name__ == '__main__':
#     # Task 1
#     X_all, y_all, subjects_all = load_and_recombine_data()
#
#     # Task 2
#     run_baseline_experiment(X_all, y_all, subjects_all)


# """
# End-to-end LOSO pipeline for UCI HAR
#
# What this script does
# - Loads and recombines UCI HAR data into arrays (and optionally writes a master CSV)
# - Runs an outer Leave-One-Subject-Out (LOSO) loop
# - Performs *group-aware* hyperparameter tuning on the training pool using GroupKFold (groups = subject IDs)
# - Supports multiple models via a model registry (LogReg, SVMs, RF, GBC, kNN, and a Keras MLP via SciKeras)
# - For the Keras MLP, compares validation strategies and training-stop choices (early stopping on val_loss/val_accuracy, or fixed epochs)
# - Computes robust metrics per test subject (accuracy, macro-F1, per-class F1) and writes per-fold results + summary CSVs
#
# Requirements
# - numpy, pandas, scikit-learn
# - tensorflow (for the MLP)
# - scikeras (pip install scikeras) for sklearn-style CV with Keras models
#
# Usage
# - Place this file at the project root and run: python l0so_har_pipeline.py
# - Adjust EXPERIMENTS at the bottom to select models/strategies
#
# Notes
# - Hyperparameter tuning uses GroupKFold on the training pool only (strictly LOSO-safe).
# - Early stopping / validation strategies are applied during the *final fit* for the MLP only.
# - Classical sklearn models are trained on the full training pool (they don't use early stopping).
# """
#
# import os
# import random
# from dataclasses import dataclass
# from typing import Dict, List, Tuple, Optional, Any
#
# import numpy as np
# import pandas as pd
# from sklearn.exceptions import DataConversionWarning
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# from sklearn.model_selection import GroupKFold, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# import warnings
#
# warnings.filterwarnings(action='ignore', category=DataConversionWarning)
#
# # TensorFlow / Keras
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
#
# # SciKeras (sklearn wrapper for Keras)
# try:
#     from scikeras.wrappers import KerasClassifier
#     HAS_SCIKERAS = True
# except Exception:
#     HAS_SCIKERAS = False
#
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)
#
# # Optional determinism (can reduce throughput)
# # try:
# #     tf.config.experimental.enable_op_determinism()
# # except Exception:
# #     pass
#
#
# # ------------------ Data Loading ------------------
#
# def load_and_recombine_data(base_path: str = "UCI HAR Dataset/",
#                             write_master_csv: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
#     """Load UCI HAR features, labels, subjects and return arrays.
#
#     Returns
#     -------
#     X_all : (n_samples, n_features)
#     y_all : (n_samples,) labels 0..5
#     subjects_all : (n_samples,) subject IDs
#     feature_names : list of str
#     """
#     print("Loading and recombining data...")
#
#     # 1. Features
#     features_df = pd.read_csv(
#         os.path.join(base_path, 'features.txt'),
#         sep=r'\s+', header=None, names=['id', 'name']
#     )
#     feature_names = (features_df['id'].astype(str) + '_' + features_df['name']).tolist()
#
#     # 2. X
#     X_train_df = pd.read_csv(os.path.join(base_path, 'train', 'X_train.txt'),
#                              sep=r'\s+', header=None, names=feature_names)
#     X_test_df = pd.read_csv(os.path.join(base_path, 'test', 'X_test.txt'),
#                             sep=r'\s+', header=None, names=feature_names)
#     X_all_df = pd.concat([X_train_df, X_test_df]).reset_index(drop=True)
#
#     # 3. y (1..6 in raw). Keras prefers 0-indexed, but we'll store both for convenience.
#     y_train_df = pd.read_csv(os.path.join(base_path, 'train', 'y_train.txt'), header=None, names=['activity_id'])
#     y_test_df  = pd.read_csv(os.path.join(base_path, 'test',  'y_test.txt'),  header=None, names=['activity_id'])
#     y_all_df = pd.concat([y_train_df, y_test_df]).reset_index(drop=True)
#     y_all_keras = (y_all_df['activity_id'] - 1).astype(int).values  # 0..5
#
#     # 4. subjects
#     subject_train_df = pd.read_csv(os.path.join(base_path, 'train', 'subject_train.txt'), header=None, names=['subject_id'])
#     subject_test_df  = pd.read_csv(os.path.join(base_path, 'test',  'subject_test.txt'),  header=None, names=['subject_id'])
#     subjects_all_df = pd.concat([subject_train_df, subject_test_df]).reset_index(drop=True)
#
#     if write_master_csv:
#         print("Creating master_data.csv...")
#         master_df = pd.concat([subjects_all_df, y_all_df, X_all_df], axis=1)
#         master_df.to_csv('master_data.csv', index=False)
#         print("master_data.csv saved successfully.")
#
#     print(f"Data combined. Total rows: {len(X_all_df)} | n_features: {X_all_df.shape[1]}")
#     return X_all_df.values.astype(np.float32), y_all_keras.ravel(), subjects_all_df.values.ravel(), feature_names
#
#
# # ------------------ Keras Model Factory ------------------
#
# def build_mlp(input_dim: int, units: int = 64, depth: int = 2, dropout: float = 0.3, lr: float = 1e-3,
#               num_classes: int = 6) -> tf.keras.Model:
#     layers = []
#     layers.append(Dense(units, activation='relu', input_shape=(input_dim,)))
#     layers.append(Dropout(dropout))
#     for _ in range(max(0, depth - 1)):
#         layers.append(Dense(units, activation='relu'))
#         layers.append(Dropout(dropout))
#     layers.append(Dense(num_classes, activation='softmax'))
#
#     model = Sequential(layers)
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model
#
#
# # ------------------ Validation Split Helpers ------------------
#
# def select_val_indices(subjects_pool: np.ndarray, pool_indices: np.ndarray,
#                         strategy: str = 'single_subject', k: int = 1,
#                         seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, List[int]]:
#     """Choose validation subset from the training pool for early stopping of the MLP.
#
#     Returns train_indices, val_indices (both are absolute indices into the full dataset), and the list of val subject IDs.
#     """
#     subj_pool = subjects_pool
#     unique_subj = np.unique(subj_pool)
#     rng = np.random.default_rng(seed)
#
#     if strategy == 'single_subject':
#         val_subj = int(rng.choice(unique_subj))
#         val_mask = (subj_pool == val_subj)
#         val_idx_abs = pool_indices[np.where(val_mask)[0]]
#         train_idx_abs = pool_indices[np.where(~val_mask)[0]]
#         return train_idx_abs, val_idx_abs, [val_subj]
#
#     if strategy == 'k_subjects':
#         k = max(1, min(k, len(unique_subj)))
#         rng.shuffle(unique_subj)
#         val_subj = list(map(int, unique_subj[:k]))
#         val_mask = np.isin(subj_pool, val_subj)
#         val_idx_abs = pool_indices[np.where(val_mask)[0]]
#         train_idx_abs = pool_indices[np.where(~val_mask)[0]]
#         return train_idx_abs, val_idx_abs, val_subj
#
#     raise ValueError("Unknown validation strategy: choose 'single_subject' or 'k_subjects'")
#
#
# # ------------------ Metrics ------------------
#
# def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
#     acc = accuracy_score(y_true, y_pred)
#     macro_f1 = f1_score(y_true, y_pred, average='macro')
#     per_class_f1 = f1_score(y_true, y_pred, average=None)
#     cm = confusion_matrix(y_true, y_pred)
#     return {
#         'accuracy': float(acc),
#         'macro_f1': float(macro_f1),
#         'per_class_f1': per_class_f1.tolist(),
#         'confusion_matrix': cm.tolist(),
#     }
#
#
# # ------------------ Model Registry ------------------
#
# def get_model_registry(n_features: int) -> Dict[str, Dict[str, Any]]:
#     """Return a registry mapping model name -> dict with estimator factory and param_grid.
#
#     For MLP (SciKeras), we tune hyperparams with GroupKFold but WITHOUT validation/early stopping.
#     We'll do early stopping in the final fit only (so we can swap strategies cleanly).
#     """
#     registry: Dict[str, Dict[str, Any]] = {}
#
#     # Logistic Regression (multinomial)
#     registry['logreg'] = {
#         'estimator': Pipeline([
#             ('scaler', StandardScaler()),
#             ('clf', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=5000, n_jobs=-1, random_state=SEED))
#         ]),
#         'param_grid': {
#             'clf__C': [0.1, 1, 3, 10],
#         }
#     }
#
#     # Linear SVM
#     registry['svm_linear'] = {
#         'estimator': Pipeline([
#             ('scaler', StandardScaler()),
#             ('clf', SVC(kernel='linear', probability=False, random_state=SEED))
#         ]),
#         'param_grid': {
#             'clf__C': [0.1, 1, 3, 10],
#         }
#     }
#
#     # RBF SVM
#     registry['svm_rbf'] = {
#         'estimator': Pipeline([
#             ('scaler', StandardScaler()),
#             ('clf', SVC(kernel='rbf', probability=False, random_state=SEED))
#         ]),
#         'param_grid': {
#             'clf__C': [1, 3, 10],
#             'clf__gamma': ['scale', 1e-3, 1e-2],
#         }
#     }
#
#     # Random Forest
#     registry['rf'] = {
#         'estimator': RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
#         'param_grid': {
#             'n_estimators': [300, 600],
#             'max_depth': [None, 20, 40],
#             'min_samples_leaf': [1, 2],
#         }
#     }
#
#     # Gradient Boosting
#     registry['gbc'] = {
#         'estimator': GradientBoostingClassifier(random_state=SEED),
#         'param_grid': {
#             'n_estimators': [150, 300],
#             'learning_rate': [0.05, 0.1],
#             'max_depth': [2, 3],
#         }
#     }
#
#     # kNN
#     registry['knn'] = {
#         'estimator': Pipeline([
#             ('scaler', StandardScaler()),
#             ('clf', KNeighborsClassifier())
#         ]),
#         'param_grid': {
#             'clf__n_neighbors': [1, 3, 5, 7],
#             'clf__metric': ['euclidean', 'manhattan'],
#         }
#     }
#
#     # Keras MLP via SciKeras
#     if HAS_SCIKERAS:
#         def scikeras_mlp(units=64, depth=2, dropout=0.3, lr=1e-3):
#             # SciKeras passes X with shape (n_samples, n_features)
#             # model__build_fn receives input_dim via model__input_dim param
#             return build_mlp(input_dim=n_features, units=units, depth=depth, dropout=dropout, lr=lr)
#
#         registry['mlp'] = {
#             'estimator': Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('clf', KerasClassifier(model=scikeras_mlp,
#                                         epochs=80, batch_size=64, verbose=0,
#                                         # No validation during CV
#                                         validation_split=0.0, shuffle=True, random_state=SEED))
#             ]),
#             'param_grid': {
#                 'clf__model__units': [64, 128],
#                 'clf__model__depth': [1, 2],
#                 'clf__model__dropout': [0.2, 0.4],
#                 'clf__model__lr': [1e-3, 3e-4],
#                 'clf__epochs': [60, 100],
#                 'clf__batch_size': [32, 128],
#             }
#         }
#     else:
#         print("[WARN] SciKeras not found. The 'mlp' model will be unavailable for GridSearchCV.\n"
#               "Install via: pip install scikeras")
#
#     return registry
#
#
# # ------------------ Training Utilities ------------------
#
# def group_grid_search(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
#                       estimator, param_grid: Dict[str, List[Any]],
#                       scoring: str = 'accuracy', n_splits: int = 5,
#                       n_jobs: int = -1, refit: bool = True, verbose: int = 0):
#     gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
#     gs = GridSearchCV(estimator, param_grid, cv=gkf.split(X, y, groups=groups),
#                       scoring=scoring, n_jobs=n_jobs, refit=refit, verbose=verbose)
#     gs.fit(X, y)
#     return gs
#
#
# def fit_mlp_with_early_stopping(X_train: np.ndarray, y_train: np.ndarray,
#                                  X_val: np.ndarray, y_val: np.ndarray,
#                                  best_params: Dict[str, Any], n_features: int,
#                                  stop_strategy: str = 'val_loss', patience: int = 10,
#                                  min_delta: float = 0.0, fixed_epochs: int = 100,
#                                  batch_size: Optional[int] = None) -> tf.keras.Model:
#     """Final MLP fit with explicit validation and selectable training-stop choice."""
#     units = best_params.get('clf__model__units', 64)
#     depth = best_params.get('clf__model__depth', 2)
#     dropout = best_params.get('clf__model__dropout', 0.3)
#     lr = best_params.get('clf__model__lr', 1e-3)
#     epochs = best_params.get('clf__epochs', fixed_epochs)
#     batch_size = batch_size or best_params.get('clf__batch_size', 64)
#
#     # Scale with training stats only
#     scaler = StandardScaler()
#     X_train_s = scaler.fit_transform(X_train)
#     X_val_s = scaler.transform(X_val)
#
#     model = build_mlp(input_dim=n_features, units=units, depth=depth, dropout=dropout, lr=lr)
#
#     callbacks = []
#     monitor = None
#
#     if stop_strategy in ('val_loss', 'val_accuracy'):
#         monitor = stop_strategy
#         callbacks.append(EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta,
#                                        restore_best_weights=True))
#     elif stop_strategy == 'none':
#         # No early stopping; override epochs to fixed_epochs if provided
#         epochs = fixed_epochs
#     else:
#         raise ValueError("stop_strategy must be one of {'val_loss','val_accuracy','none'}")
#
#     history = model.fit(
#         X_train_s, y_train,
#         validation_data=(X_val_s, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=callbacks,
#         verbose=0
#     )
#
#     # Track best epoch for logging
#     if stop_strategy in ('val_loss', 'val_accuracy'):
#         key = monitor
#         best_epoch = int(np.nanargmin(history.history[key]) if key == 'val_loss' else np.nanargmax(history.history[key])) + 1
#     else:
#         best_epoch = epochs
#
#     model._scaler = scaler  # attach for inference convenience
#     model._best_epoch = best_epoch
#     model._stop_strategy = stop_strategy
#     return model
#
#
# def predict_mlp(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
#     Xs = model._scaler.transform(X)
#     probs = model.predict(Xs, verbose=0)
#     return np.argmax(probs, axis=1)
#
#
# # ------------------ Experiment Config ------------------
#
# @dataclass
# class Experiment:
#     name: str
#     model_name: str  # key in registry
#     # For MLP only:
#     val_strategy: str = 'single_subject'  # 'single_subject' | 'k_subjects'
#     k_val_subjects: int = 1
#     stop_strategy: str = 'val_loss'       # 'val_loss' | 'val_accuracy' | 'none'
#     patience: int = 10
#     min_delta: float = 0.0
#     fixed_epochs: int = 100
#
#
# # ------------------ LOSO Runner ------------------
#
# def run_loso(X_all: np.ndarray, y_all: np.ndarray, subjects_all: np.ndarray,
#              experiments: List[Experiment], results_dir: str = 'results',
#              n_splits_inner: int = 5) -> pd.DataFrame:
#     os.makedirs(results_dir, exist_ok=True)
#
#     all_subjects = np.unique(subjects_all)
#     n_features = X_all.shape[1]
#
#     rows = []
#
#     for test_subj in all_subjects:
#         test_mask = (subjects_all == test_subj)
#         test_idx = np.where(test_mask)[0]
#         X_test, y_test = X_all[test_idx], y_all[test_idx]
#
#         pool_idx = np.where(~test_mask)[0]
#         X_pool, y_pool, subj_pool = X_all[pool_idx], y_all[pool_idx], subjects_all[pool_idx]
#
#         # Build registry (needs n_features for MLP)
#         registry = get_model_registry(n_features)
#
#         for exp in experiments:
#             if exp.model_name not in registry:
#                 print(f"[WARN] Skipping experiment '{exp.name}' because model '{exp.model_name}' is not available.")
#                 continue
#
#             print(f"\n=== Test Subject {test_subj} | Experiment: {exp.name} ({exp.model_name}) ===")
#
#             base_estimator = registry[exp.model_name]['estimator']
#             param_grid = registry[exp.model_name]['param_grid']
#
#             # ---- Hyperparameter tuning (group-aware) on the whole training pool ----
#             gs = group_grid_search(X_pool, y_pool, subj_pool, base_estimator, param_grid,
#                                    scoring='accuracy', n_splits=n_splits_inner, n_jobs=-1, refit=True)
#
#             # Retrieve best params for logging
#             best_params = gs.best_params_
#             best_cv_score = float(gs.best_score_)
#
#             # ---- Final fit + evaluation ----
#             if exp.model_name == 'mlp' and HAS_SCIKERAS:
#                 # Pick validation subset inside the training pool for early stopping
#                 tr_idx_abs, val_idx_abs, val_subjs = select_val_indices(subjects_pool=subj_pool,
#                                                                         pool_indices=pool_idx,
#                                                                         strategy=exp.val_strategy,
#                                                                         k=exp.k_val_subjects,
#                                                                         seed=SEED)
#                 X_tr, y_tr = X_all[tr_idx_abs], y_all[tr_idx_abs]
#                 X_val, y_val = X_all[val_idx_abs], y_all[val_idx_abs]
#
#                 mlp_model = fit_mlp_with_early_stopping(
#                     X_tr, y_tr, X_val, y_val,
#                     best_params=best_params, n_features=n_features,
#                     stop_strategy=exp.stop_strategy, patience=exp.patience,
#                     min_delta=exp.min_delta, fixed_epochs=exp.fixed_epochs,
#                 )
#                 y_pred = predict_mlp(mlp_model, X_test)
#                 best_epoch = getattr(mlp_model, '_best_epoch', None)
#             else:
#                 # Classical model: refit best estimator on full pool (standard practice)
#                 best_est = gs.best_estimator_
#                 best_est.fit(X_pool, y_pool)
#                 y_pred = best_est.predict(X_test)
#                 best_epoch = None
#                 val_subjs = []
#
#             m = compute_metrics(y_test, y_pred)
#
#             row = {
#                 'test_subject': int(test_subj),
#                 'experiment': exp.name,
#                 'model': exp.model_name,
#                 'best_cv_acc': best_cv_score,
#                 'test_accuracy': m['accuracy'],
#                 'test_macro_f1': m['macro_f1'],
#                 'per_class_f1': m['per_class_f1'],
#                 'confusion_matrix': m['confusion_matrix'],
#                 'best_epoch': best_epoch,
#                 'val_subjects': val_subjs,
#                 'best_params': best_params,
#             }
#             rows.append(row)
#
#         # Persist after each outer fold to avoid losing progress
#         df_partial = pd.DataFrame(rows)
#         df_partial.to_json(os.path.join(results_dir, 'per_fold_results.jsonl'), orient='records', lines=True)
#         df_partial.to_csv(os.path.join(results_dir, 'per_fold_results.csv'), index=False)
#
#     # Final summary
#     df = pd.DataFrame(rows)
#     if not df.empty:
#         summary = (df.groupby(['experiment', 'model'])
#                      .agg(mean_acc=('test_accuracy', 'mean'), std_acc=('test_accuracy', 'std'),
#                           mean_macro_f1=('test_macro_f1', 'mean'), std_macro_f1=('test_macro_f1', 'std'),
#                           n_folds=('test_accuracy', 'count'))
#                      .reset_index())
#         summary.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)
#         print("\n=== Summary ===")
#         print(summary)
#     else:
#         print("No results collected. Check your experiments configuration.")
#
#     return df if 'df' in locals() else pd.DataFrame()
#
#
# # ------------------ Main ------------------
#
# if __name__ == '__main__':
#     X_all, y_all, subjects_all, feature_names = load_and_recombine_data()
#
#     # Configure experiments
#
#     EXPERIMENTS: List[Experiment] = [
#         # Classical baselines (no early stopping)
#         #Experiment(name='LR_multinomial', model_name='logreg'),
#         #Experiment(name='SVM_linear', model_name='svm_linear'),
#         #Experiment(name='SVM_rbf', model_name='svm_rbf'),
#         Experiment(name='RandomForest', model_name='rf')
#         #Experiment(name='GradientBoosting', model_name='gbc'),
#         #Experiment(name='kNN', model_name='knn'),
#
#         # MLP variants: compare validation choice + training-stop choice
#         # Experiment(name='MLP_single_val_loss', model_name='mlp', val_strategy='single_subject', stop_strategy='val_loss', patience=10, min_delta=1e-4, fixed_epochs=120),
#         # Experiment(name='MLP_single_val_acc',  model_name='mlp', val_strategy='single_subject', stop_strategy='val_accuracy', patience=10, min_delta=1e-4, fixed_epochs=120),
#         # Experiment(name='MLP_k3_val_loss',     model_name='mlp', val_strategy='k_subjects',     k_val_subjects=3, stop_strategy='val_loss', patience=10, min_delta=1e-4, fixed_epochs=120),
#         # Experiment(name='MLP_fixed_epochs',    model_name='mlp', val_strategy='single_subject', stop_strategy='none', fixed_epochs=120),
#     ]
#     _ = run_loso(X_all, y_all, subjects_all, EXPERIMENTS, results_dir='results', n_splits_inner=5)



"""
Lightweight LOSO pipeline for UCI HAR (with comments & logging)

Goals (kept):
- Leave-One-Subject-Out (LOSO) evaluation (outer loop)
- Compare validation-choice + training-stop choices for a compact neural model (MLP)
- Do group-aware hyperparameter tuning (inner CV) with subject-disjoint folds
- Provide a strong classic baseline (Logistic Regression) with a tiny grid
- Keep it FAST: tiny grids, 3-fold GroupKFold, moderate epochs

Outputs: results/per_fold.csv and results/summary.csv

Usage
-----
python loso_har_simple_clean.py  # defaults to INFO logging

Set environment variable LOGLEVEL to DEBUG/INFO/WARNING/ERROR to control verbosity, e.g.:
LOGLEVEL=DEBUG python loso_har_simple_clean.py
"""

from __future__ import annotations

import os
import random
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------- Logging --------------------
# Configure a module-level logger. Users can control verbosity via LOGLEVEL env var.
LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOGLEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("LOSO")

# -------------------- Reproducibility --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# If you want to reduce GPU VRAM spikes, uncomment memory growth:
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#         logger.info("Enabled memory growth on %s", gpus[0].name)
#     except Exception as e:
#         logger.warning("Could not set memory growth: %s", e)

# -------------------- Data --------------------

def load_har(base_path: str = 'UCI HAR Dataset/') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load UCI HAR features (561), labels (0..5), and subject IDs.

    This merges the official train/test partitions into a single matrix, so we can
    perform subject-based LOSO splitting ourselves.
    """
    logger.info("Loading UCI HAR from %s", base_path)

    # Parse the feature list with whitespace regex to avoid irregular spacing issues
    feats = pd.read_csv(os.path.join(base_path, 'features.txt'), sep=r'\s+', header=None, names=['id', 'name'])
    names = (feats['id'].astype(str) + '_' + feats['name']).tolist()

    # Load features
    Xtr = pd.read_csv(os.path.join(base_path, 'train', 'X_train.txt'), sep=r'\s+', header=None, names=names)
    Xte = pd.read_csv(os.path.join(base_path, 'test', 'X_test.txt'), sep=r'\s+', header=None, names=names)
    X = pd.concat([Xtr, Xte]).reset_index(drop=True).values.astype(np.float32)

    # Load labels, convert to 0..5 for TF/sparse CE
    ytr = pd.read_csv(os.path.join(base_path, 'train', 'y_train.txt'), header=None)[0]
    yte = pd.read_csv(os.path.join(base_path, 'test', 'y_test.txt'), header=None)[0]
    y = pd.concat([ytr, yte]).reset_index(drop=True).values.astype(int) - 1

    # Load subject IDs
    str_tr = pd.read_csv(os.path.join(base_path, 'train', 'subject_train.txt'), header=None)[0]
    str_te = pd.read_csv(os.path.join(base_path, 'test', 'subject_test.txt'), header=None)[0]
    subjects = pd.concat([str_tr, str_te]).reset_index(drop=True).values

    logger.info(
        "Loaded %d samples with %d features across %d subjects",
        X.shape[0], X.shape[1], np.unique(subjects).size,
    )
    return X, y, subjects

# -------------------- Metrics --------------------

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute lightweight metrics for fast iteration."""
    return {
        'acc': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
    }

# -------------------- Logistic Regression (baseline) --------------------

def tune_logreg(X_pool: np.ndarray, y_pool: np.ndarray, subj_pool: np.ndarray) -> GridSearchCV:
    """Group-aware hyperparameter search for multinomial Logistic Regression.

    Uses a tiny grid and 3-fold GroupKFold (subjects as groups) for speed.
    """
    logger.debug("Tuning Logistic Regression on %d pool samples", X_pool.shape[0])

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        (
            'clf',
            LogisticRegression(
                multi_class='multinomial', solver='saga', max_iter=3000, n_jobs=-1, random_state=SEED
            ),
        ),
    ])
    grid = {'clf__C': [0.3, 1, 3]}

    gkf = GroupKFold(n_splits=min(3, len(np.unique(subj_pool))))
    gs = GridSearchCV(
        pipe,
        grid,
        cv=gkf.split(X_pool, y_pool, groups=subj_pool),
        scoring='accuracy',
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_pool, y_pool)

    logger.info("LR best params: %s | CV acc: %.4f", gs.best_params_, gs.best_score_)
    return gs

# -------------------- Simple MLP + tiny manual CV --------------------

def build_mlp(
    input_dim: int,
    units: int,
    dropout: float,
    lr: float,
    num_classes: int = 6,
) -> tf.keras.Model:
    """Tiny 2-hidden-layer MLP suitable for the engineered HAR features."""
    m = Sequential(
        [
            Dense(units, activation='relu', input_shape=(input_dim,)),
            Dropout(dropout),
            Dense(units, activation='relu'),
            Dropout(dropout),
            Dense(num_classes, activation='softmax'),
        ]
    )
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return m

# Very small hyperparam grid for speed
MLP_GRID: List[Dict[str, Any]] = [
    {'units': 64, 'dropout': 0.2, 'lr': 1e-3, 'epochs': 40, 'batch': 128},
    {'units': 128, 'dropout': 0.4, 'lr': 1e-3, 'epochs': 40, 'batch': 128},
]

def tune_mlp_tiny(
    X_pool: np.ndarray, y_pool: np.ndarray, subj_pool: np.ndarray
) -> Dict[str, Any]:
    """Subject-safe 3-fold CV over a tiny grid.

    We train fixed-epoch models per fold (no early stopping) to rank configs quickly.
    Returns the best config dictionary augmented with 'cv_acc'.
    """
    logger.debug("Tuning MLP on %d pool samples", X_pool.shape[0])

    gkf = GroupKFold(n_splits=min(3, len(np.unique(subj_pool))))
    n_feat = X_pool.shape[1]
    best_acc: float = -1.0
    best_cfg: Optional[Dict[str, Any]] = None

    for cfg in MLP_GRID:
        fold_accs: List[float] = []
        logger.debug("Testing MLP cfg: %s", cfg)
        for tr_idx, va_idx in gkf.split(X_pool, y_pool, groups=subj_pool):
            # Split subject-disjoint folds
            Xtr, Xva = X_pool[tr_idx], X_pool[va_idx]
            ytr, yva = y_pool[tr_idx], y_pool[va_idx]

            # Standardize features using training fold only
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xva_s = sc.transform(Xva)

            # Build and train model
            model = build_mlp(n_feat, cfg['units'], cfg['dropout'], cfg['lr'])
            model.fit(
                Xtr_s,
                ytr,
                validation_data=(Xva_s, yva),
                epochs=cfg['epochs'],
                batch_size=cfg['batch'],
                verbose=0,
            )

            # Evaluate on the validation fold
            preds = np.argmax(model.predict(Xva_s, verbose=0), axis=1)
            acc = accuracy_score(yva, preds)
            fold_accs.append(acc)

            # Free TF state (helps when looping many folds)
            tf.keras.backend.clear_session()

        mean_acc = float(np.mean(fold_accs))
        logger.info("MLP cfg %s -> CV acc: %.4f", cfg, mean_acc)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_cfg = cfg

    assert best_cfg is not None
    best_cfg = dict(best_cfg)
    best_cfg['cv_acc'] = best_acc
    logger.info("Best MLP cfg: %s | CV acc: %.4f", best_cfg, best_acc)
    return best_cfg

# Final fit with configurable validation-choice + stop-choice

def final_fit_mlp(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    subj_pool: np.ndarray,
    X_test: np.ndarray,
    stop_strategy: str = 'val_loss',
    k_val_subjects: int = 1,
    best_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Train the final MLP with a specific early-stopping/validation choice and predict on the test set.

    - Selects 1 or K validation subjects from the training pool (subject-pure)
    - Uses EarlyStopping on val_loss/val_accuracy or runs fixed epochs if 'none'
    - Returns test predictions and a small info dict for logging/reporting
    """
    n_feat = X_pool.shape[1]
    if best_cfg is None:
        best_cfg = MLP_GRID[0]

    # Pick validation subject(s) deterministically from the pool for reproducibility
    uniq = np.unique(subj_pool)
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    val_subjs = uniq[: max(1, k_val_subjects)]
    val_mask = np.isin(subj_pool, val_subjs)
    tr_mask = ~val_mask

    logger.debug("Final MLP fit | val_subjects=%s | stop=%s", val_subjs, stop_strategy)

    # Split into train/val using the chosen subjects
    Xtr, ytr = X_pool[tr_mask], y_pool[tr_mask]
    Xva, yva = X_pool[val_mask], y_pool[val_mask]

    # Standardize with training statistics only
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xva_s = sc.transform(Xva)
    Xte_s = sc.transform(X_test)

    # Build model from best CV config
    model = build_mlp(n_feat, best_cfg['units'], best_cfg['dropout'], best_cfg['lr'])

    # Configure training-stop policy
    callbacks: List[tf.keras.callbacks.Callback] = []
    epochs = best_cfg['epochs']
    if stop_strategy in ('val_loss', 'val_accuracy'):
        callbacks.append(
            EarlyStopping(
                monitor=stop_strategy, patience=8, min_delta=1e-4, restore_best_weights=True
            )
        )
    elif stop_strategy == 'none':
        # Fixed number of epochs
        pass
    else:
        raise ValueError("stop_strategy must be 'val_loss', 'val_accuracy', or 'none'")

    # Train
    hist = model.fit(
        Xtr_s,
        ytr,
        validation_data=(Xva_s, yva),
        epochs=epochs,
        batch_size=best_cfg['batch'],
        callbacks=callbacks,
        verbose=0,
    )

    # Predict on test subject
    preds = np.argmax(model.predict(Xte_s, verbose=0), axis=1)

    # Derive the best epoch (if early stopping was enabled)
    if stop_strategy == 'val_loss' and 'val_loss' in hist.history:
        best_epoch = int(np.argmin(hist.history['val_loss'])) + 1
    elif stop_strategy == 'val_accuracy' and 'val_accuracy' in hist.history:
        best_epoch = int(np.argmax(hist.history['val_accuracy'])) + 1
    else:
        best_epoch = epochs

    info = {
        'val_subjects': list(map(int, val_subjs)),
        'best_epoch': int(best_epoch),
        'stop_strategy': stop_strategy,
        'units': best_cfg['units'],
        'dropout': best_cfg['dropout'],
        'lr': best_cfg['lr'],
        'epochs': epochs,
        'batch': best_cfg['batch'],
    }

    tf.keras.backend.clear_session()
    return preds, info

# -------------------- Experiment config --------------------

@dataclass
class Experiment:
    """Simple experiment record defining a model and (for MLP) its validation/stop choices."""
    name: str
    model: str  # 'logreg' or 'mlp'
    stop_strategy: str = 'val_loss'  # MLP only: 'val_loss' | 'val_accuracy' | 'none'
    k_val_subjects: int = 1  # MLP only

# -------------------- LOSO Runner --------------------

def run_loso(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    experiments: List[Experiment],
    results_dir: str = 'results',
) -> pd.DataFrame:
    """Run a LOSO outer loop across all subjects and evaluate the configured experiments.

    For each test subject:
      - Build the training pool (all other subjects)
      - Tune each model with GroupKFold on the pool (subject-safe)
      - Fit final model (MLP uses a subject-held-out val set for early stopping)
      - Evaluate on the held-out test subject and log metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    all_subj = np.unique(subjects)
    rows: List[Dict[str, Any]] = []

    for test_subj in all_subj:
        # Partition into test vs training pool by subject ID
        test_mask = subjects == test_subj
        Xte, yte = X[test_mask], y[test_mask]
        Xpool, ypool, spool = X[~test_mask], y[~test_mask], subjects[~test_mask]

        logger.info(
            "=== Test Subject %s | pool=%d | test=%d ===",
            str(test_subj),
            Xpool.shape[0],
            Xte.shape[0],
        )

        # Cache tuning results to reuse for multiple experiments on the same subject
        lr_gs: Optional[GridSearchCV] = None  # Logistic Regression GridSearch result
        mlp_best: Optional[Dict[str, Any]] = None  # Best MLP config from tiny manual CV

        for exp in experiments:
            logger.info("Running experiment: %s (%s)", exp.name, exp.model)

            if exp.model == 'logreg':
                if lr_gs is None:
                    lr_gs = tune_logreg(Xpool, ypool, spool)
                best_est = lr_gs.best_estimator_.fit(Xpool, ypool)
                yhat = best_est.predict(Xte)
                info = {'best_params': lr_gs.best_params_, 'cv_acc': float(lr_gs.best_score_)}

            elif exp.model == 'mlp':
                if mlp_best is None:
                    mlp_best = tune_mlp_tiny(Xpool, ypool, spool)  # quick subject-safe search
                yhat, info = final_fit_mlp(
                    Xpool,
                    ypool,
                    spool,
                    Xte,
                    stop_strategy=exp.stop_strategy,
                    k_val_subjects=exp.k_val_subjects,
                    best_cfg=mlp_best,
                )
                info['cv_acc'] = float(mlp_best['cv_acc'])
                info['best_params'] = {k: mlp_best[k] for k in ['units', 'dropout', 'lr', 'epochs', 'batch']}

            else:
                raise ValueError('Unknown model in experiment')

            # Compute metrics on the held-out test subject
            m = metrics(yte, yhat)
            logger.info("%s | acc=%.4f | macro_f1=%.4f", exp.name, m['acc'], m['macro_f1'])

            rows.append(
                {
                    'test_subject': int(test_subj),
                    'experiment': exp.name,
                    'model': exp.model,
                    'test_acc': m['acc'],
                    'test_macro_f1': m['macro_f1'],
                    **info,
                }
            )

        # Save partial results after each subject (robust to interruptions)
        pd.DataFrame(rows).to_csv(os.path.join(results_dir, 'per_fold.csv'), index=False)
        logger.debug("Wrote partial per_fold.csv with %d rows", len(rows))

    # Aggregate across subjects and write summary
    df = pd.DataFrame(rows)
    if not df.empty:
        summary = (
            df.groupby(['experiment', 'model'])
            .agg(
                mean_acc=('test_acc', 'mean'),
                std_acc=('test_acc', 'std'),
                mean_macro_f1=('test_macro_f1', 'mean'),
                std_macro_f1=('test_macro_f1', 'std'),
                n=('test_acc', 'count'),
            )
            .reset_index()
        )
        summary.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)
        logger.info("\nSummary:\n%s", summary)
    else:
        logger.warning("No results collected. Did the dataset load correctly?")

    return df

# -------------------- Main --------------------

if __name__ == '__main__':
    X, y, subjects = load_har()

    # Define a few simple experiments covering validation & training-stop choices on the MLP
    EXPS: List[Experiment] = [
        #Experiment(name='LR_baseline', model='logreg'),
        Experiment(name='MLP_single_val_loss', model='mlp', stop_strategy='val_loss', k_val_subjects=1),
        Experiment(name='MLP_k2_val_accuracy', model='mlp', stop_strategy='val_accuracy', k_val_subjects=2),
        Experiment(name='MLP_fixed_epochs', model='mlp', stop_strategy='none', k_val_subjects=1),
    ]

    _ = run_loso(X, y, subjects, EXPS, results_dir='results')
