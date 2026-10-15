import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader.eeg_loader import EEGLoader
from src.data_loader.synthetic_generator import generate_eeg_epoch, STANDARD_19_CHANNELS
from src.preprocessing.pipeline import PreprocessingPipeline
from src.feature_extraction.pipeline import FeatureExtractionPipeline
from src.models.hybrid_ensemble import HybridCNNSVMKNNEnsemble
from src.visualization import (
    plot_eeg_signals,
    plot_brain_topomap,
    plot_confusion_matrix_heatmap,
    plot_training_curves
)

st.set_page_config(
    page_title="NeuroPulse AI | Clinical EEG Decision Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber-Neuro Dark CSS
st.markdown("""
<style>
    .stApp { background-color: #06080d; color: #f8fafc; }
    div[data-testid="stMetricValue"] { color: #00f0ff; font-family: monospace; font-weight: 800; }
    .status-card {
        padding: 18px; border-radius: 12px; border: 1px solid rgba(0, 240, 255, 0.3);
        background: rgba(16, 24, 38, 0.8); margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

SAVED_MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
SAMPLES_DIR = os.path.join(ROOT_DIR, "data", "samples")

@st.cache_resource
def load_diagnostic_system():
    ensemble = HybridCNNSVMKNNEnsemble.load(SAVED_MODELS_DIR)
    scaler_path = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    preprocessor = PreprocessingPipeline(sfreq=256, epoch_duration=2.0, overlap=0.5)
    feat_extractor = FeatureExtractionPipeline(sfreq=256)
    loader = EEGLoader(sfreq=256)
    return ensemble, scaler, preprocessor, feat_extractor, loader

ensemble, scaler, preprocessor, feat_extractor, eeg_loader = load_diagnostic_system()

# Sidebar Controls
st.sidebar.title("🧠 NeuroPulse AI")
st.sidebar.caption("Hybrid CNN-SVM-KNN Clinical Decision Support • University of Mumbai")

input_mode = st.sidebar.radio(
    "Select EEG Input Source:",
    ["Clinical Case Presets", "Upload Patient File (.CSV / .EDF / .NPY)"]
)

raw_eeg = None
ch_names = STANDARD_19_CHANNELS
sfreq = 256
case_label = "Normal Subject"

if input_mode == "Clinical Case Presets":
    preset_choice = st.sidebar.selectbox(
        "Choose Patient Profile:",
        [
            "Profile A — Healthy Control (Normal)",
            "Profile B — Early Alzheimer's Disease (AD)",
            "Profile C — Major Depressive Disorder (MDD)"
        ]
    )
    cond_map = {
        "Profile A — Healthy Control (Normal)": ("normal", "patient_normal_sample.csv"),
        "Profile B — Early Alzheimer's Disease (AD)": ("alzheimer", "patient_alzheimer_sample.csv"),
        "Profile C — Major Depressive Disorder (MDD)": ("depression", "patient_depression_sample.csv")
    }
    cond_key, sample_fname = cond_map[preset_choice]
    case_label = preset_choice
    sample_path = os.path.join(SAMPLES_DIR, sample_fname)

    if os.path.exists(sample_path):
        raw_eeg, ch_names, sfreq = eeg_loader.load(sample_path)
    else:
        raw_eeg, _ = generate_eeg_epoch(condition=cond_key, duration=6.0, sfreq=256)
else:
    uploaded = st.sidebar.file_uploader("Upload EEG Recording", type=["csv", "edf", "bdf", "npy"])
    if uploaded is not None:
        os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
        temp_path = os.path.join(ROOT_DIR, "data", f"st_upload_{uploaded.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        try:
            raw_eeg, ch_names, sfreq = eeg_loader.load(temp_path)
            case_label = f"Uploaded: {uploaded.name}"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

max_ch_plot = st.sidebar.slider("Oscilloscope Channels to Display", 4, 19, 8)
duration_plot = st.sidebar.slider("Signal Window Duration (sec)", 2.0, 6.0, 4.0, 0.5)

# Main Header
st.title("⚡ NeuroPulse AI — Clinical EEG Diagnostic Workstation")
st.markdown("**Hybrid 1D-CNN + RBF-SVM + k-NN Ensemble** | 19-Channel 10–20 International Montage (256 Hz)")

tab_diag, tab_bench = st.tabs(["🔬 Live Patient Diagnosis & Topography", "📊 Model Benchmarks & Training Curves"])

with tab_diag:
    if raw_eeg is None:
        st.info("Please upload an EEG file in the sidebar or switch to Clinical Case Presets.")
    else:
        # Run Pipeline
        clean_res = preprocessor.process(raw_eeg)
        clean_eeg = clean_res["cleaned_continuous"]
        epochs = clean_res["epochs"]
        feats = feat_extractor.extract_batch(epochs)
        f3d, f2d = feats["features_3d"], feats["features_2d"]

        # Biomarkers from unscaled features
        mean_tar = float(np.mean(f3d[:, :, 14]))
        mean_slowing = float(np.mean(f3d[:, :, 15]))
        mean_complexity = float(np.mean(f3d[:, :, 7]))
        alpha_f3 = float(np.mean(f3d[:, 3, 10])) + 1e-6
        alpha_f4 = float(np.mean(f3d[:, 5, 10])) + 1e-6
        faa_val = float(np.log(alpha_f4 / alpha_f3))

        if scaler is not None:
            try:
                f3d_scaled = scaler.transform(f2d).reshape(f3d.shape).astype(np.float32)
            except Exception:
                f3d_scaled = f3d
        else:
            f3d_scaled = f3d

        detailed = ensemble.predict_detailed(f3d_scaled)
        avg_probs = {
            "Normal": float(np.mean([d["probabilities"]["Normal"] for d in detailed])) * 100,
            "Alzheimer's Disease": float(np.mean([d["probabilities"]["Alzheimer's Disease"] for d in detailed])) * 100,
            "Depression": float(np.mean([d["probabilities"]["Depression"] for d in detailed])) * 100
        }
        pred_label = max(avg_probs, key=avg_probs.get)
        confidence = avg_probs[pred_label]

        # Top KPI Row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Primary Diagnosis", pred_label)
        c2.metric("Ensemble Confidence", f"{confidence:.1f}%")
        c3.metric("Theta/Alpha Ratio (TAR)", f"{mean_tar:.2f}")
        c4.metric("Slowing Index", f"{mean_slowing:.2f}")
        c5.metric("Frontal Asymmetry (FAA)", f"{faa_val:.2f}")

        st.divider()

        # Probability Breakdown & Topomap
        col_left, col_right = st.columns([1.45, 0.85])

        with col_left:
            st.subheader("📈 Multi-Channel EEG Telemetry (Raw vs. FastICA Preprocessed)")
            fig_eeg = plot_eeg_signals(
                raw_eeg=raw_eeg,
                clean_eeg=clean_eeg,
                channel_names=ch_names,
                sfreq=sfreq,
                duration=duration_plot,
                max_channels=max_ch_plot
            )
            st.pyplot(fig_eeg)

        with col_right:
            st.subheader("🗺️ 2D Cortical Activity Topomap")
            # Map mean relative alpha power per channel
            ch_power_map = {
                ch: float(np.mean(f3d[:, idx, 10]))
                for idx, ch in enumerate(ch_names[:19])
            }
            fig_topo = plot_brain_topomap(
                channel_values=ch_power_map,
                title=f"Scalp Alpha Power Distribution ({pred_label})"
            )
            st.pyplot(fig_topo)

        st.subheader("🤖 Hybrid Ensemble Probability Consensus")
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.progress(int(avg_probs["Normal"]), text=f"Normal: {avg_probs['Normal']:.1f}%")
        p_col2.progress(int(avg_probs["Alzheimer's Disease"]), text=f"Alzheimer's Disease: {avg_probs['Alzheimer's Disease']:.1f}%")
        p_col3.progress(int(avg_probs["Depression"]), text=f"Major Depression: {avg_probs['Depression']:.1f}%")

with tab_bench:
    st.subheader("🏆 Empirical Model Benchmark Comparison")
    comp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        st.dataframe(df_comp, use_container_width=True)

    b_col1, b_col2 = st.columns(2)
    metrics_path = os.path.join(RESULTS_DIR, "clinical_metrics.json")
    history_path = os.path.join(RESULTS_DIR, "training_history.json")

    with b_col1:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                m_data = json.load(f)
            if "confusion_matrix" in m_data:
                fig_cm = plot_confusion_matrix_heatmap(m_data["confusion_matrix"])
                st.pyplot(fig_cm)

    with b_col2:
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                h_data = json.load(f)
            fig_curves = plot_training_curves(h_data)
            st.pyplot(fig_curves)
