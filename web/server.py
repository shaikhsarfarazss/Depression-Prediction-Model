import os
import sys
import json
import joblib
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader.eeg_loader import EEGLoader
from src.preprocessing.pipeline import PreprocessingPipeline
from src.feature_extraction.pipeline import FeatureExtractionPipeline
from src.models.hybrid_ensemble import HybridCNNSVMKNNEnsemble

app = FastAPI(title="NeuroPulse AI | EEG Neurological Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = os.path.join(ROOT_DIR, "web")
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

ensemble = HybridCNNSVMKNNEnsemble.load(SAVED_MODELS_DIR)
scaler_path = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

preprocessor = PreprocessingPipeline(sfreq=256, epoch_duration=2.0, overlap=0.5)
feat_extractor = FeatureExtractionPipeline(sfreq=256)
eeg_loader = EEGLoader(sfreq=256)

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "system": "NeuroPulse AI",
        "institution": "University of Mumbai, YTIET",
        "degree": "M.E. Computer Engineering",
        "model": "Hybrid CNN-SVM-KNN Framework",
        "channels": 19,
        "sampling_rate": 256
    }

@app.get("/api/metrics")
async def get_metrics():
    metrics_path = os.path.join(RESULTS_DIR, "clinical_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {"error": "Metrics not found"}

@app.get("/api/comparison")
async def get_comparison():
    comp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(comp_path):
        df = pd.read_csv(comp_path)
        return df.to_dict(orient="records")
    return {"error": "Comparison data not found"}

@app.post("/api/predict")
async def predict_eeg(file: UploadFile = File(...)):
    os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
    temp_path = os.path.join(ROOT_DIR, "data", f"upload_{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        data, ch_names, sfreq = eeg_loader.load(temp_path)
        clean_res = preprocessor.process(data)
        epochs = clean_res["epochs"]
        feats = feat_extractor.extract_batch(epochs)

        f3d = feats["features_3d"]
        f2d = feats["features_2d"]

        # Extract raw biomarkers before scaling:
        # In each channel's 28-feature vector:
        # idx 7 = Hjorth Complexity, idx 10 = rel_alpha, idx 14 = TAR, idx 15 = Slowing Ratio
        mean_tar = float(np.mean(f3d[:, :, 14]))
        mean_slowing = float(np.mean(f3d[:, :, 15]))
        mean_complexity = float(np.mean(f3d[:, :, 7]))
        # Frontal Alpha Asymmetry: ln(Right F4[idx 5]) - ln(Left F3[idx 3])
        alpha_f3 = float(np.mean(f3d[:, 3, 10])) + 1e-6
        alpha_f4 = float(np.mean(f3d[:, 5, 10])) + 1e-6
        faa_val = float(np.log(alpha_f4 / alpha_f3))

        if scaler is not None:
            try:
                f2d_scaled = scaler.transform(f2d)
                f3d_input = f2d_scaled.reshape(f3d.shape).astype(np.float32)
            except Exception:
                f3d_input = f3d
        else:
            f3d_input = f3d

        detailed = ensemble.predict_detailed(f3d_input)
        avg_probs = {
            "Normal": float(np.mean([d["probabilities"]["Normal"] for d in detailed])),
            "Alzheimer's Disease": float(np.mean([d["probabilities"]["Alzheimer's Disease"] for d in detailed])),
            "Depression": float(np.mean([d["probabilities"]["Depression"] for d in detailed]))
        }
        pred_label = max(avg_probs, key=avg_probs.get)

        cnn_avg = np.mean([d["component_predictions"]["cnn_probs"] for d in detailed], axis=0) * 100.0
        svm_avg = np.mean([d["component_predictions"]["svm_probs"] for d in detailed], axis=0) * 100.0
        knn_avg = np.mean([d["component_predictions"]["knn_probs"] for d in detailed], axis=0) * 100.0

        return {
            "prediction": pred_label,
            "confidence": round(avg_probs[pred_label] * 100.0, 2),
            "probabilities": avg_probs,
            "classifiers": {
                "cnn": [round(float(x), 1) for x in cnn_avg],
                "svm": [round(float(x), 1) for x in svm_avg],
                "knn": [round(float(x), 1) for x in knn_avg]
            },
            "biomarkers": {
                "tar": f"{mean_tar:.2f} ({'Elevated' if mean_tar > 1.8 else 'Normal'})",
                "slowingIndex": f"{mean_slowing:.2f} ({'Pathological' if mean_slowing > 1.5 else 'Optimal'})",
                "faa": f"{faa_val:.2f} ({'Asymmetric' if abs(faa_val) > 0.25 else 'Symmetrical'})",
                "hjorthComplexity": f"{mean_complexity:.2f}"
            },
            "epochs_analyzed": len(detailed),
            "channels": ch_names
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

app.mount("/css", StaticFiles(directory=os.path.join(WEB_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(WEB_DIR, "js")), name="js")
if os.path.exists(os.path.join(WEB_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIR, "assets")), name="assets")

if __name__ == "__main__":
    uvicorn.run("web.server:app", host="127.0.0.1", port=8080, reload=False)
