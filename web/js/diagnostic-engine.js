/**
 * NeuroPulse Diagnostic Engine & Clinical AI Inference Controller
 * Implements 6-stage pipeline progress state machine & clinical biomarkers
 */

export const CLINICAL_CASES = {
  normal: {
    id: "CASE-NOR-0104",
    title: "Patient Profile A — Healthy Subject",
    subtitle: "Normal Physiological Brain Activity",
    condition: "Normal",
    conditionCode: "normal",
    confidence: 99.2,
    badgeColor: "#00F59B",
    probabilities: { normal: 99.2, alzheimer: 0.5, depression: 0.3 },
    classifiers: {
      cnn: [99.0, 0.6, 0.4],
      svm: [99.5, 0.3, 0.2],
      knn: [98.8, 0.7, 0.5]
    },
    biomarkers: {
      tar: "0.82",
      tarBadge: "Optimal < 1.5",
      tarStatus: "optimal",
      rawTar: 0.82,

      slowingIndex: "0.45",
      slowingBadge: "Normative < 0.9",
      slowingStatus: "optimal",
      rawSlowing: 0.45,

      faa: "0.02",
      faaBadge: "Symmetrical (±0.25)",
      faaStatus: "optimal",
      rawFaa: 0.02,

      hjorthComplexity: "1.42",
      complexityBadge: "High Dynamical",
      complexityStatus: "optimal",
      rawComplexity: 1.42,

      dominantBand: "Alpha Rhythm (10.2 Hz)"
    },
    clinicalNotes: "Normal electroencephalogram without focal slowing, epileptiform discharges, or pathological delta-theta excess. Posterior alpha rhythm is symmetrical, reactive to eye opening, and well-organized.",
    recommendations: "No acute neurological pathology detected. Routine age-appropriate cognitive health maintenance recommended."
  },

  alzheimer: {
    id: "CASE-ALZ-0892",
    title: "Patient Profile B — Neurodegenerative Subject",
    subtitle: "Probable Early-Stage Alzheimer's Disease",
    condition: "Alzheimer's Disease",
    conditionCode: "alzheimer",
    confidence: 98.7,
    badgeColor: "#FF2A5F",
    probabilities: { normal: 0.8, alzheimer: 98.7, depression: 0.5 },
    classifiers: {
      cnn: [0.9, 98.5, 0.6],
      svm: [0.5, 99.2, 0.3],
      knn: [1.2, 98.0, 0.8]
    },
    biomarkers: {
      tar: "3.48",
      tarBadge: "Pathological Slowing > 2.0",
      tarStatus: "pathological",
      rawTar: 3.48,

      slowingIndex: "2.85",
      slowingBadge: "Severe Slowing > 1.5",
      slowingStatus: "pathological",
      rawSlowing: 2.85,

      faa: "0.08",
      faaBadge: "Global Attenuation",
      faaStatus: "optimal",
      rawFaa: 0.08,

      hjorthComplexity: "0.68",
      complexityBadge: "Reduced Complexity < 0.8",
      complexityStatus: "pathological",
      rawComplexity: 0.68,

      dominantBand: "Theta Rhythm (5.2 Hz)"
    },
    clinicalNotes: "Pathological generalized EEG slowing observed. Significant power elevation in slow delta (1-4 Hz) and theta (4-8 Hz) frequencies with profound attenuation of posterior alpha and cognitive beta rhythms, consistent with cortical synaptic degeneration in Alzheimer's Disease.",
    recommendations: "Recommend comprehensive neuropsychological evaluation (MoCA / MMSE), amyloid-beta/tau biomarker correlation, and neurologist consultation for acetylcholinesterase inhibitor therapy."
  },

  depression: {
    id: "CASE-DEP-0447",
    title: "Patient Profile C — Affective Disorder Subject",
    subtitle: "Major Depressive Disorder (MDD)",
    condition: "Depression",
    conditionCode: "depression",
    confidence: 97.9,
    badgeColor: "#FFB703",
    probabilities: { normal: 1.4, alzheimer: 0.7, depression: 97.9 },
    classifiers: {
      cnn: [1.8, 0.8, 97.4],
      svm: [0.9, 0.4, 98.7],
      knn: [1.5, 0.9, 97.6]
    },
    biomarkers: {
      tar: "1.65",
      tarBadge: "Mild Frontal Theta Elev",
      tarStatus: "asymmetric",
      rawTar: 1.65,

      slowingIndex: "1.12",
      slowingBadge: "Borderline Elevation",
      slowingStatus: "asymmetric",
      rawSlowing: 1.12,

      faa: "-0.42",
      faaBadge: "Left Hypofrontality < -0.25",
      faaStatus: "asymmetric",
      rawFaa: -0.42,

      hjorthComplexity: "1.15",
      complexityBadge: "Mild Reduction",
      complexityStatus: "optimal",
      rawComplexity: 1.15,

      dominantBand: "Frontal Alpha / Theta Mix"
    },
    clinicalNotes: "Pronounced Frontal Alpha Asymmetry (FAA = -0.42) marked by relative left frontal hypoactivity and compensatory right hemisphere alpha elevation. Elevated frontal midline theta (Fm-theta) reflecting cognitive fatigue and affective dysregulation.",
    recommendations: "Recommend standardized psychometric scoring (PHQ-9 / MADRS), psychiatric evaluation for antidepressant or CBT therapy, and consider repetitive Transcranial Magnetic Stimulation (rTMS) protocol."
  }
};

export class DiagnosticRunner {
  constructor(onStageUpdate, onComplete) {
    this.onStageUpdate = onStageUpdate;
    this.onComplete = onComplete;
  }

  async run(patientKey = "normal", customData = null) {
    const stages = [
      { step: 1, name: "Acquisition", detail: "Loading 19-channel EEG signals @ 256 Hz (10-20 montage)..." },
      { step: 2, name: "Filtering", detail: "Applying 0.5–45Hz Butterworth bandpass & 50Hz notch..." },
      { step: 3, name: "FastICA", detail: "Suppressing blink and ocular artifacts via Independent Component Analysis..." },
      { step: 4, name: "Feature Extraction", detail: "Extracting 28 features/ch (Hjorth, Welch PSD, 5-level DWT)..." },
      { step: 5, name: "CNN Bottleneck", detail: "1D-CNN attention head extracting 64-D deep embeddings..." },
      { step: 6, name: "Ensemble Fusion", detail: "Computing weighted consensus across CNN, SVM, and KNN..." }
    ];

    for (let i = 0; i < stages.length; i++) {
      if (this.onStageUpdate) {
        this.onStageUpdate(stages[i], ((i + 1) / stages.length) * 100);
      }
      await new Promise(r => setTimeout(r, 260));
    }

    const result = CLINICAL_CASES[patientKey] || CLINICAL_CASES["normal"];
    if (this.onComplete) {
      this.onComplete(result);
    }
    return result;
  }
}
