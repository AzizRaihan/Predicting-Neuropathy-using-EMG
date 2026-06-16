# Neuropathy Risk Screening System

An AI-powered EMG (electromyography) sensor-based system for detecting peripheral neuropathy risk using machine learning. The system captures 60 seconds of muscle electrical signals and predicts whether a patient is at risk for neuropathy with 85.2% accuracy.

## Features

- **Machine Learning Model**: LightGBM classifier trained on 10,000 patient records
- **Three Input Modes**:
  - Manual input of EMG features
  - Simulated 60-second EMG recording
  - Live sensor integration (Arduino-ready)
- **Modern Web Interface**: Built with Streamlit, featuring Apple-inspired glassmorphism design
- **Cross-validated**: 5-fold stratified cross-validation for reliable performance
- **Production-ready**: Serialized model, clean prediction function, documented pipeline

## Quick Start

### Requirements

- Python 3.9+
- pip or conda

### Installation

```bash
git clone https://github.com/AzizRaihan/Predicting-Neuropathy-using-EMG.git
cd Predicting-Neuropathy-using-EMG

pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

## Usage

### Mode 1: Manual Input
Enter patient age, sex, EMG RMS, and median frequency to get instant predictions.

### Mode 2: Simulate Recording
Click "Start Simulation" to watch a synthetic 60-second EMG recording. The app automatically calculates features and makes a prediction.

### Mode 3: Live Sensor (Requires Arduino)
Connect an Arduino with EMG sensor and select the serial port. Records real EMG data for 60 seconds and makes a prediction.

## Project Structure

```
├── app.py                          # Streamlit web interface
├── predict.py                      # LightGBM inference function
├── sensor.py                       # EMG sensor interface (simulation + live)
├── neuropathy_clean_10k.csv        # Processed dataset (10,000 records)
├── sra499.ipynb                    # Original Jupyter notebook with analysis
├── train_models.py                 # Training script for initial models
├── train_more_models.py            # Training script for additional models
├── ensemble.py                     # Ensemble method implementations
├── PROJECT_REPORT.md               # Detailed project report
└── .streamlit/config.toml          # Streamlit configuration
```

## Model Performance

**Test Accuracy**: 85.2%  
**Recall (At-Risk Detection)**: 87% — catches most at-risk patients  
**Precision**: 88% — minimizes false alarms  

The model was trained and evaluated using 5-fold stratified cross-validation to prevent data leakage and ensure reliable performance.

## Dataset

**Source**: Clinical EMG data from neuropathy patients and controls  
**Size**: 10,000 patient records  
**Features**: Age, Sex, EMG RMS, EMG Median Frequency  
**Target**: Binary (Healthy vs. Neuropathy Risk)  
**Class Distribution**: 40% Healthy, 60% At-Risk  

## Signal Processing

### EMG RMS (Root Mean Square)
```
RMS = √(mean(signal²))
```
Measures signal amplitude/strength. Typical ranges:
- Healthy: 0.45–0.65 µV
- At-risk: 0.25–0.40 µV

### EMG Median Frequency
Calculated using Welch's Power Spectral Density method. Represents the frequency dividing the signal power in half.
- Healthy: 65–85 Hz
- At-risk: 40–60 Hz (frequency shift indicates nerve degradation)

## Simulation Logic

The simulation generates synthetic EMG data by combining:
1. **Broadband noise** (±15 µV) — natural muscle fiber activity
2. **60 Hz sine wave** (±8 µV) — primary contraction frequency
3. **120 Hz harmonic** (±4 µV) — frequency harmonic
4. **Low-frequency drift** (±5 µV) — baseline variations
5. **DC offset** (+100 µV) — physiological baseline

The signal is generated instantly but the app waits 60 real seconds to simulate actual recording time.

## Hardware Integration (Arduino)

For live sensor mode, use Arduino with an EMG sensor (e.g., RBD-221 from Robotics Bangladesh):

```cpp
// Arduino sketch template
void setup() {
  Serial.begin(9600);
}

void loop() {
  float voltage = analogRead(A0) * (5.0 / 1023.0);
  Serial.println(voltage);
  delay(1); // ~1000 Hz sampling rate
}
```

Connect via USB serial to the laptop and select the port in the app.

## Project Report

For a detailed explanation of:
- Data preprocessing
- Model training and evaluation
- Feature extraction methodology
- Simulation logic
- Complete performance metrics

See `PROJECT_REPORT.md`

## Key Files Explained

| File | Purpose |
|---|---|
| `app.py` | Streamlit web UI with 3 modes (manual, simulate, live) |
| `predict.py` | LightGBM inference engine; loads model and makes predictions |
| `sensor.py` | EMG signal processing; generates synthetic or reads live signals |
| `sra499.ipynb` | Jupyter notebook with original analysis, EDA, baseline, XGBoost training |
| `train_models.py` | Trains 4 baseline models (LogReg, RF, XGBoost, LightGBM) |
| `train_more_models.py` | Trains 5 additional models (SVM, KNN, ExtraTrees, GB, CatBoost) |
| `ensemble.py` | Builds ensemble methods (soft voting, stacking) |

## Model Selection

**LightGBM** was selected as the production model because:
- Highest cross-validation accuracy: 85.44%
- Best recall for at-risk detection: 87%
- Fast inference (milliseconds per prediction)
- Minimal overfitting (85.44% CV vs 85.20% test = 0.24% gap)
- Easy to serialize and integrate

## Training & Validation

All models were trained using **5-fold stratified cross-validation**:
- Prevents data leakage (scaler fit independently on each fold)
- Maintains class balance (40:60) across all folds
- Provides robust accuracy estimates

**Total models trained and compared**: 15 (linear, tree-based, ensemble methods)

## Future Work

1. **Arduino Integration**: Write and test Arduino EMG sensor sketch
2. **Clinical Validation**: Test with real patients and compare predictions to clinical diagnoses
3. **Feature Engineering**: Add additional EMG features (e.g., spectral entropy, peak frequency)
4. **Deployment**: Deploy Streamlit app to cloud platform for wider accessibility

## Installation Issues

### Missing libomp (XGBoost Error)
```bash
brew install libomp
```

### Streamlit Email Prompt
Create `~/.streamlit/credentials.toml`:
```toml
[general]
email = ""
```

## Contact & Questions

For questions or issues, open an issue on GitHub.

## License

This project is for educational and research purposes.

---

**Last Updated**: June 16, 2026  
**Status**: Production-ready (awaiting Arduino sketch for live sensor mode)
