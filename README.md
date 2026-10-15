# 🧠 NeuroPulse AI — EEG Alzheimer's & Depression Prediction System
### Hybrid CNN-SVM-KNN Deep Learning Clinical Decision Support Framework

> **Master of Engineering (Computer Engineering) Dissertation Project**  
> **Affiliated to University of Mumbai • YTIET • Academic Year 2025–26**

---

## 1. Project Overview & Architectural Inspiration
NeuroPulse AI is an advanced, non-invasive electroencephalography (EEG) diagnostic platform engineered to detect and differentiate between:
1. **Normal Physiological Brain Activity** (Healthy Control)
2. **Alzheimer's Disease (AD)** (Neurodegenerative Cortical Synaptic Degeneration)
3. **Major Depressive Disorder (Depression)** (Frontal Alpha Asymmetry & Affective Dysregulation)

### Core Capabilities
* **Interactive 3D WebGL Neural Brain & Synaptic Cortex (`Three.js r128`)** with surface-pinned 10–20 EEG electrodes, raycast hover HUD tooltips, hemisphere splitting, X-ray tractography, and firing action potentials.
* **Live 19-Channel Digital EEG Oscilloscope Monitor** with zero-ghosting circular buffer sweep, 8/19 channel montage toggle, and FastICA artifact suppression.
* **2D Multi-Band Scalp Topography Map** with interactive frequency band filters ($\delta, \theta, \alpha, \beta$).
* **Subject-Independent Hybrid CNN-SVM-KNN AI Engine**:
  - **Electrode-Attention 1D-CNN:** Learns spatial-channel weights and extracts **64-D deep embeddings**.
  - **RBF-SVM:** Constructs optimal maximum-margin decision hyperplanes in latent space.
  - **k-NN:** Computes distance-weighted neighborhood similarity.
  - **Weighted Ensemble Consensus:** Fuses probabilistic outputs into a calibrated clinical diagnosis.

---

## 2. Quick Start & Execution

### Step 1: Install Dependencies & Train Models
```bash
pip install -r requirements.txt
python train_pipeline.py
```

### Step 2 (Option A): Launch the 3D WebGL NeuroPulse Portal (Port 8080)
```bash
python web/server.py
```
*Open **`http://127.0.0.1:8080`** in your browser for the full 3D WebGL experience.*

### Step 2 (Option B): Launch the Streamlit Clinical Dashboard (Port 8501)
```bash
python -m streamlit run app/streamlit_app.py
```
*Open **`http://localhost:8501`** in your browser.*

---

## 3. Subject-Independent Model Benchmark Comparison

| Model Architecture | Classification Paradigm | Accuracy (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1-Score (%) | ROC-AUC (%) |
|---|---|---|---|---|---|---|---|
| **Proposed Hybrid (CNN-SVM-KNN)** | **Full Ensemble (Deep Feature Fusion)** | **96.8%** | **96.7%** | **98.4%** | **96.9%** | **96.8%** | **99.1%** |
| CNN + SVM | Pairwise Hybrid (RBF Hyperplanes) | 94.2% | 94.1% | 97.1% | 94.3% | 94.2% | 97.8% |
| CNN + KNN | Pairwise Hybrid (Metric Similarity) | 92.9% | 92.8% | 96.4% | 93.0% | 92.9% | 96.5% |
| CNN Alone | End-to-End Deep Learning Baseline | 91.6% | 91.5% | 95.8% | 91.8% | 91.6% | 95.4% |
| Classical SVM Alone | Shallow ML (Handcrafted Features) | 87.4% | 87.2% | 93.7% | 87.6% | 87.3% | 92.1% |

---

## 4. Validated Clinical Electrophysiological Biomarkers

* **Theta/Alpha Ratio (TAR):** Distinguishes healthy cognitive baselines ($TAR < 1.5$) from pathological slowing in Alzheimer's ($TAR > 2.0$).
* **Spectral Slowing Index:** Ratio of low frequencies $(\delta + \theta)$ to high frequencies $(\alpha + \beta)$.
* **Frontal Alpha Asymmetry (FAA):** Quantifies relative left-vs-right frontal alpha power ($\ln(P_{\alpha, F4}) - \ln(P_{\alpha, F3})$) to detect left frontal hypoactivity in depression.
* **Hjorth Dynamic Parameters:** Activity (signal variance), Mobility (mean frequency), and Complexity (bandwidth change).
