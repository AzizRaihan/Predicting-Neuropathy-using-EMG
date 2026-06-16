import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = '/Users/azizraihan/Desktop/mom499/all_model_results.pkl'

_pipeline = None

def _load_model():
    global _pipeline
    if _pipeline is None:
        results = joblib.load(MODEL_PATH)
        _pipeline = results['LightGBM']['pipeline']
    return _pipeline


def predict_neuropathy(age: float, sex: int, emg_rms: float, emg_median_freq: float) -> dict:
    """
    Predict neuropathy risk from 4 inputs.

    Args:
        age             : age in years (e.g. 55.0)
        sex             : 0 = Male, 1 = Female
        emg_rms         : EMG RMS mean computed from 1-minute reading (µV)
        emg_median_freq : EMG median frequency computed from 1-minute reading (Hz)

    Returns:
        dict with keys:
            'prediction'   : 'Healthy' or 'Neuropathy Risk'
            'confidence'   : probability of the predicted class (0–1)
            'prob_healthy' : probability of Healthy
            'prob_risk'    : probability of Neuropathy Risk
    """
    pipeline = _load_model()

    features = pd.DataFrame([{
        'age_years':             age,
        'sex':                   sex,
        'EMG_RMS_mean':          emg_rms,
        'EMG_median_frequency':  emg_median_freq
    }])

    prediction   = pipeline.predict(features)[0]
    probabilities = pipeline.predict_proba(features)[0]

    label = 'Neuropathy Risk' if prediction == 1 else 'Healthy'

    return {
        'prediction':   label,
        'confidence':   round(float(probabilities[prediction]), 4),
        'prob_healthy': round(float(probabilities[0]), 4),
        'prob_risk':    round(float(probabilities[1]), 4),
    }


# ── Manual test cases ────────────────────────────────────────────────────────
if __name__ == '__main__':

    test_cases = [
        {
            'label':   'Young healthy male (low risk profile)',
            'age':     28, 'sex': 0,
            'emg_rms': 88.0, 'emg_median_freq': 92.0
        },
        {
            'label':   'Elderly female (high risk profile)',
            'age':     72, 'sex': 1,
            'emg_rms': 130.0, 'emg_median_freq': 52.0
        },
        {
            'label':   'Middle-aged male (borderline)',
            'age':     51, 'sex': 0,
            'emg_rms': 101.0, 'emg_median_freq': 76.0
        },
        {
            'label':   'Older male (high risk)',
            'age':     65, 'sex': 0,
            'emg_rms': 145.0, 'emg_median_freq': 48.0
        },
        {
            'label':   'Young female (low risk)',
            'age':     33, 'sex': 1,
            'emg_rms': 82.0, 'emg_median_freq': 98.0
        },
    ]

    print("=" * 62)
    print("  NEUROPATHY RISK PREDICT FUNCTION — TEST RUN")
    print("  Model: LightGBM (CV Acc: 85.44%)")
    print("=" * 62)

    for tc in test_cases:
        result = predict_neuropathy(
            age=tc['age'], sex=tc['sex'],
            emg_rms=tc['emg_rms'], emg_median_freq=tc['emg_median_freq']
        )
        print(f"\n  Case    : {tc['label']}")
        print(f"  Inputs  : Age={tc['age']}, Sex={'Male' if tc['sex']==0 else 'Female'}, "
              f"EMG RMS={tc['emg_rms']} µV, Median Freq={tc['emg_median_freq']} Hz")
        print(f"  Result  : {result['prediction']}")
        print(f"  Healthy : {result['prob_healthy']*100:.1f}%   "
              f"Neuropathy Risk: {result['prob_risk']*100:.1f}%   "
              f"Confidence: {result['confidence']*100:.1f}%")

    print("\n" + "=" * 62)
    print("  Predict function working correctly.")
    print("=" * 62)
