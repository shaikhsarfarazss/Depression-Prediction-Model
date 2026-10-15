import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.logger import get_logger
from src.utils.seed_utils import set_seed
from src.data_loader.synthetic_generator import create_sample_dataset, generate_eeg_epoch, STANDARD_19_CHANNELS
from src.preprocessing.pipeline import PreprocessingPipeline
from src.feature_extraction.pipeline import FeatureExtractionPipeline
from src.models.model_trainer import ModelTrainer
from src.evaluation.metrics import calculate_clinical_metrics
from src.evaluation.benchmark_comparison import run_benchmark_comparison

logger = get_logger("TrainPipeline")

def extract_split_features(file_records, preprocessor, feat_extractor):
    features_3d, features_2d, labels = [], [], []
    for fpath, class_idx in file_records:
        df = pd.read_csv(fpath)
        eeg_data = df[STANDARD_19_CHANNELS].values.T
        clean_res = preprocessor.process(eeg_data)
        epochs = clean_res["epochs"]
        feats = feat_extractor.extract_batch(epochs)
        f3d, f2d = feats["features_3d"], feats["features_2d"]
        for i in range(f3d.shape[0]):
            features_3d.append(f3d[i])
            features_2d.append(f2d[i])
            labels.append(class_idx)
    return (
        np.array(features_3d, dtype=np.float32),
        np.array(features_2d, dtype=np.float32),
        np.array(labels, dtype=np.int64)
    )

def main():
    set_seed(42)
    logger.info("Initializing EEG Alzheimer's & Depression Prediction Pipeline...")

    raw_dir = os.path.join(ROOT_DIR, "data", "raw")
    samples_dir = os.path.join(ROOT_DIR, "data", "samples")
    saved_models_dir = os.path.join(ROOT_DIR, "saved_models")
    results_dir = os.path.join(ROOT_DIR, "results")

    for d in [raw_dir, samples_dir, saved_models_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Dataset Generation
    logger.info("Step 1: Generating benchmark multi-channel EEG recordings...")
    dataset_files = create_sample_dataset(
        output_dir=raw_dir,
        n_samples_per_class=35,
        duration=6.0,
        sfreq=256
    )

    for cond_name, cond_key in [("Normal", "normal"), ("Alzheimer's Disease", "alzheimer"), ("Depression", "depression")]:
        eeg_sig, t = generate_eeg_epoch(condition=cond_key, duration=6.0, sfreq=256)
        df = pd.DataFrame(eeg_sig.T, columns=STANDARD_19_CHANNELS)
        df.insert(0, "time", t)
        df.to_csv(os.path.join(samples_dir, f"patient_{cond_key}_sample.csv"), index=False)

    # 2. Subject-Level Stratified Split (Prevents Overlapping Epoch Leakage!)
    label_mapping = {"Normal": 0, "Alzheimer's Disease": 1, "Depression": 2}
    all_recordings = []
    for class_name, file_list in dataset_files.items():
        class_idx = label_mapping[class_name]
        for fpath in file_list:
            all_recordings.append((fpath, class_idx))

    rec_labels = [r[1] for r in all_recordings]
    train_recs, temp_recs = train_test_split(
        all_recordings, test_size=0.30, random_state=42, stratify=rec_labels
    )
    temp_labels = [r[1] for r in temp_recs]
    val_recs, test_recs = train_test_split(
        temp_recs, test_size=0.50, random_state=42, stratify=temp_labels
    )

    logger.info(f"Subject Split -> Train: {len(train_recs)} | Val: {len(val_recs)} | Test: {len(test_recs)} patients")

    # 3. Preprocessing & Feature Extraction per Subject Split
    preprocessor = PreprocessingPipeline(sfreq=256, epoch_duration=2.0, overlap=0.5)
    feat_extractor = FeatureExtractionPipeline(sfreq=256)

    X_train_3d, X_train_2d, y_train = extract_split_features(train_recs, preprocessor, feat_extractor)
    X_val_3d, X_val_2d, y_val = extract_split_features(val_recs, preprocessor, feat_extractor)
    X_test_3d, X_test_2d, y_test = extract_split_features(test_recs, preprocessor, feat_extractor)

    # 4. Fit StandardScaler on Training Set Only and Transform 2D & 3D Tensors
    scaler = StandardScaler()
    X_train_2d_scaled = scaler.fit_transform(X_train_2d)
    X_val_2d_scaled = scaler.transform(X_val_2d)
    X_test_2d_scaled = scaler.transform(X_test_2d)
    joblib.dump(scaler, os.path.join(saved_models_dir, "scaler.pkl"))

    n_channels, n_feats = X_train_3d.shape[1], X_train_3d.shape[2]
    X_train_3d_scaled = X_train_2d_scaled.reshape(-1, n_channels, n_feats).astype(np.float32)
    X_val_3d_scaled = X_val_2d_scaled.reshape(-1, n_channels, n_feats).astype(np.float32)
    X_test_3d_scaled = X_test_2d_scaled.reshape(-1, n_channels, n_feats).astype(np.float32)

    # 5. Train Proposed Hybrid Model
    logger.info("Step 3: Training Hybrid CNN-SVM-KNN Framework...")
    trainer = ModelTrainer(
        n_channels=n_channels,
        n_features=n_feats,
        n_classes=3,
        epochs=30,
        batch_size=32,
        lr=0.001
    )
    ensemble, history = trainer.train_hybrid_framework(
        X_train=X_train_3d_scaled,
        y_train=y_train,
        X_val=X_val_3d_scaled,
        y_val=y_val,
        fusion_strategy="weighted_voting"
    )

    ensemble.save(saved_models_dir)
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # 6. Clinical Evaluation
    y_pred = ensemble.predict(X_test_3d_scaled)
    y_prob = ensemble.predict_proba(X_test_3d_scaled)
    metrics = calculate_clinical_metrics(y_test, y_pred, y_prob)

    with open(os.path.join(results_dir, "clinical_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 7. Benchmark Comparison
    comparison_df = run_benchmark_comparison(
        ensemble=ensemble,
        X_test_3d=X_test_3d_scaled,
        X_test_2d=X_test_2d_scaled,
        y_test=y_test,
        X_train_2d=X_train_2d_scaled,
        y_train=y_train
    )
    comparison_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    print("\n--- BENCHMARK COMPARISON TABLE ---")
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()
