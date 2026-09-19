# HARMONY-X

## Adaptive Multimodal Intelligence System

HARMONY-X is a research-oriented multimodal intelligence prototype that combines **computer vision, audio perception, multimodal fusion, contextual reasoning, and machine learning** into a unified system.

The system processes information from a camera and microphone, extracts visual and audio signals, combines them through a multimodal fusion layer, and provides a real-time representation of the interaction state through a Streamlit dashboard.

> **Project Type:** Research Prototype  
> **Author:** Mutyala Jayadeep  
> **Institution:** IIT (BHU), Varanasi  
> **Department:** Mechanical Engineering

---

## Research Question

> **Can a multimodal architecture combining visual perception, audio signals, contextual information, and adaptive reasoning provide a richer representation of human interaction than relying on a single sensory modality?**

---

## Overview

Human interaction contains information beyond spoken words.

HARMONY-X explores the combination of:

- Facial activity
- Hand activity
- Body pose
- Facial expression signals
- Speech activity
- Audio energy
- Multimodal confidence
- Context from recent observations
- Machine learning predictions

The goal is to create a modular architecture where different sensing and reasoning components can work together.

---

## System Architecture

```text
                    HARMONY-X
                        │
          ┌─────────────┴─────────────┐
          │                           │
       CAMERA                    MICROPHONE
          │                           │
          ▼                           ▼
   ┌──────────────┐            ┌──────────────┐
   │    VISION    │            │    AUDIO     │
   │  PERCEPTION  │            │  PERCEPTION  │
   ├──────────────┤            ├──────────────┤
   │ Face         │            │ Speech       │
   │ Hands        │            │ RMS          │
   │ Pose         │            │ Volume       │
   │ Expression   │            │ Confidence   │
   └──────┬───────┘            └──────┬───────┘
          │                           │
          └─────────────┬─────────────┘
                        ▼
              ┌───────────────────┐
              │ MULTIMODAL FUSION │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ CONTEXT /         │
              │ REASONING         │
              ├───────────────────┤
              │ Context Memory    │
              │ ML Prediction     │
              │ Decision Logic    │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ HARMONY-X         │
              │ DASHBOARD         │
              ├───────────────────┤
              │ Live Perception   │
              │ Fusion State      │
              │ Confidence        │
              │ Telemetry         │
              └───────────────────┘
```

---

# Key Components

## 1. Vision Perception

The vision subsystem uses **MediaPipe Tasks** and OpenCV.

It provides:

- Face detection
- Facial landmark analysis
- Facial blendshape information
- Hand tracking
- Pose detection
- Expression estimation

The expression component estimates observable facial expression signals.

> Expression estimation should not be interpreted as a definitive measurement of a person's internal emotional state.

---

## 2. Audio Perception

The audio subsystem processes microphone input in real time.

Current audio features include:

- RMS energy
- Volume estimation
- Speech activity estimation
- Audio confidence
- WebRTC callback telemetry

The microphone is configured as an **input-only sensor**.

HARMONY-X does not intentionally send microphone audio back through the speakers, preventing microphone-to-speaker loopback.

---

## 3. Multimodal Fusion

The fusion subsystem combines visual and audio information.

It calculates:

- Visual confidence
- Audio confidence
- Expression confidence
- Overall multimodal confidence

Conceptually:

```text
Visual Signals
      +
Audio Signals
      +
Expression Signals
      │
      ▼
Multimodal Fusion
      │
      ▼
Overall Confidence
```

---

## 4. Context Memory

HARMONY-X maintains information about recent observations.

Stored information can include:

- Timestamp
- Face presence
- Hand activity
- Pose activity
- Expression
- Expression confidence
- Speech activity
- Audio volume
- Multimodal confidence
- System action

This allows the system to consider temporal context instead of treating every observation independently.

---

## 5. Reasoning Layer

The reasoning layer combines:

- Current multimodal observations
- Contextual information
- Confidence levels
- Temporal statistics
- Machine learning predictions

---

# Machine Learning

HARMONY-X includes a **Random Forest baseline classifier**.

The model uses multimodal and contextual features such as:

```text
face_detected
hands_detected
pose_detected
visual_confidence
emotion_confidence
speech_detected
volume
audio_confidence
overall_confidence
speech_frequency
face_presence_frequency
average_volume
memory_size
```

The current baseline focuses on controlled **HAPPY vs NEUTRAL** observations.

---

## Experimental Results

| Metric | Result |
|---|---:|
| Accuracy | 94.19% |
| Macro F1 | 0.92 |
| Weighted F1 | 0.94 |

### Confusion Matrix

```text
                Predicted
              HAPPY  NEUTRAL

Actual HAPPY    574     39
Actual NEUTRAL    6    155
```

> These results represent a controlled experimental baseline and should not be interpreted as general real-world human emotion-recognition accuracy.

---

# Technology Stack

- Python
- OpenCV
- MediaPipe Tasks
- NumPy
- SoundDevice
- WebRTC
- Scikit-learn
- Random Forest
- Joblib
- Streamlit
- streamlit-webrtc
- Pandas
- Matplotlib
- Plotly
- VS Code
- Git / GitHub

---

# Project Structure

```text
HARMONY-X/
│
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── dashboard/
│   └── app.py
├── docs/
├── experiments/
│   ├── run_experiment.py
│   └── experiment_report.py
├── models/
│   ├── mediapipe/
│   └── harmony_x_emotion_model.joblib
├── notebooks/
├── src/
│   ├── perception/
│   ├── audio/
│   ├── fusion/
│   └── reasoning/
├── training/
├── tests/
├── main.py
├── repair_dataset.py
├── requirements.txt
└── README.md
```

---

# Installation

## Clone

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd HARMONY-X
```

## Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## Dependencies

```powershell
pip install -r requirements.txt
```

---

# Run the Dashboard

```powershell
streamlit run dashboard\app.py
```

The dashboard provides real-time monitoring of:

- Camera input
- Microphone input
- Face detection
- Hand detection
- Pose detection
- Expression estimation
- Speech activity
- Audio volume
- Multimodal fusion
- Confidence values
- System telemetry

---

# Run Experiments

```powershell
python experiments\run_experiment.py
python experiments\experiment_report.py
```

---

# Train the ML Model

```powershell
python training\train_emotion_model.py
```

The trained model is saved to:

```text
models/harmony_x_emotion_model.joblib
```

---

# Experimental Pipeline

```text
Multimodal Sensor Data
        │
        ▼
Feature Extraction
        │
        ▼
Processed Dataset
        │
        ▼
Train / Test Split
        │
        ▼
Random Forest
        │
        ▼
Predictions
        │
        ▼
Evaluation
        ├── Accuracy
        ├── Precision
        ├── Recall
        ├── F1 Score
        └── Confusion Matrix
```

---

# Current Status

```text
Vision Perception        [✓] Implemented
Face Detection           [✓] Implemented
Hand Tracking            [✓] Implemented
Pose Detection           [✓] Implemented
Expression Analysis      [✓] Implemented
Audio Processing         [✓] Implemented
Speech Detection         [✓] Implemented
Audio RMS                [✓] Implemented
Volume Estimation        [✓] Implemented
Multimodal Fusion        [✓] Implemented
Context Memory           [✓] Implemented
Reasoning Engine         [✓] Implemented
ML Baseline              [✓] Implemented
Experiment Pipeline      [✓] Implemented
Evaluation               [✓] Implemented
Streamlit Dashboard      [✓] Implemented
WebRTC Camera            [✓] Implemented
WebRTC Microphone        [✓] Implemented
Advanced Adaptive Learning [ ] Future Work
Session-Level Evaluation [ ] Future Work
Advanced Multimodal Fusion [ ] Future Work
```

---

# Limitations

The current version is a research prototype.

Important limitations include:

- The dataset is limited and contains controlled observations.
- The model has not been validated across a large population.
- Expression signals are not equivalent to internal emotional states.
- The current baseline uses frame-level observations.
- Adjacent observations may be correlated.
- Real-world performance can vary with lighting, cameras, microphones, noise, occlusion, users, and environments.

A stronger future evaluation should use independent recording sessions and subject/session-level splitting.

---

# Future Work

Potential directions include:

### Temporal Modeling

- LSTM
- GRU
- Temporal Transformers
- Temporal convolution

### Advanced Multimodal Fusion

- Early fusion
- Late fusion
- Attention-based fusion
- Cross-modal transformers

### Adaptive Learning

Develop mechanisms that allow the system to adapt to changing environments and interaction patterns.

### Better Uncertainty Modeling

Improve the system's ability to distinguish between high-confidence, moderate-confidence, and insufficient observations.

### Improved Evaluation

- Multiple participants
- Independent sessions
- Session-level train/test splitting
- Different environments
- Larger datasets
- Cross-subject evaluation

---

# Ethical Considerations

HARMONY-X processes visual and audio information from human interactions.

Important considerations:

- Obtain consent before recording people.
- Do not use the system for covert surveillance.
- Do not treat expression estimates as definitive psychological assessments.
- Do not make high-stakes decisions solely from model predictions.
- Protect recorded audio/video and derived datasets.
- Communicate model uncertainty clearly.

---

# Research Applications

HARMONY-X can serve as a foundation for research in:

- Multimodal Artificial Intelligence
- Human-Computer Interaction
- Adaptive AI
- Affective Computing
- Computer Vision
- Audio Intelligence
- Context-Aware Systems
- Human-Robot Interaction
- Intelligent Assistive Systems

---

# Author

## Mutyala Jayadeep

**B.Tech – Mechanical Engineering**  
**Indian Institute of Technology (BHU), Varanasi**

### Interests

- Artificial Intelligence
- Machine Learning
- Data Science
- Data Analytics
- AI Engineering
- Multimodal AI
- Computer Vision
- Intelligent Systems
- Mechanical Engineering

---

# Project Status

**HARMONY-X v0.1.0**

**Research Prototype**

The current implementation focuses on building and evaluating a modular multimodal intelligence pipeline. Advanced adaptive learning and larger-scale evaluation remain future research directions.

---

# License

This project is intended primarily for academic and research purposes.

Add an appropriate open-source license before publicly distributing the repository.
