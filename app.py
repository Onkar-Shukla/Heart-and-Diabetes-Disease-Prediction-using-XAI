"""
MedPredict AI — Flask Prediction API
══════════════════════════════════════════════════════════
B.Tech Final Year Project | Medical AI with XAI
Supports: Heart Disease (13 features) + Diabetes (8 features)
Models: Random Forest (scikit-learn)
Datasets: UCI Heart Disease + Pima Indian Diabetes

FIX: UCI Heart Disease label encoding varies across dataset versions.
     Some preprocessed versions flip 0/1, causing predictions to always
     return "Low Cardiac Risk". Solution: use predict_proba threshold
     (>= 0.5) instead of raw prediction class label.
══════════════════════════════════════════════════════════
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)


# ── Safe Model Loader ─────────────────────────────────────
def load_pickle(path: str):
    """Load a pickle file safely. Returns None on failure."""
    if not os.path.exists(path):
        print(f"[WARNING] Model file not found: {path}")
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load {path}: {e}")
        return None


# ── Load Models & Scalers ─────────────────────────────────
heart_model     = load_pickle('heart_model.pkl')
heart_scaler    = load_pickle('heart_scaler.pkl')
diabetes_model  = load_pickle('diabetes_model.pkl')
diabetes_scaler = load_pickle('diabetes_scaler.pkl')

# ── Detect label encoding direction ──────────────────────
# UCI Heart Disease has two common encodings:
#   Standard : target 1 = disease   (proba[:,1] = disease prob)
#   Flipped  : target 0 = disease   (proba[:,0] = disease prob)
# We auto-detect by running a known "sick" sample through the model
# and checking which class probability is higher.
HEART_DISEASE_CLASS = 1          # default; may be corrected below

if heart_model and heart_scaler:
    try:
        # Canonical "high-risk" UCI sample (male, 67 y/o, cp=0, chol=286, etc.)
        probe = [[67, 1, 0, 160, 286, 0, 0, 108, 1, 1.5, 1, 3, 2]]
        scaled_probe = heart_scaler.transform(probe)
        p = heart_model.predict_proba(scaled_probe)[0]
        if p[0] > p[1]:
            # Model trained with 0 = disease; flip the reference class
            HEART_DISEASE_CLASS = 0
            print("[INFO] Heart model label flip detected — using class 0 as disease.")
        else:
            print("[INFO] Heart model label encoding looks standard (class 1 = disease).")
    except Exception as e:
        print(f"[WARNING] Label-encoding probe failed: {e}. Using class 1 default.")

DIABETES_DISEASE_CLASS = 1       # Pima dataset is almost always standard

print("\n════════════════════════════════════════")
print("   MedPredict AI — Model Load Status")
print("════════════════════════════════════════")
print(f"  Heart  Model  : {'✓ Loaded' if heart_model     else '✗ NOT FOUND'}")
print(f"  Heart  Scaler : {'✓ Loaded' if heart_scaler    else '✗ NOT FOUND'}")
print(f"  Diab.  Model  : {'✓ Loaded' if diabetes_model  else '✗ NOT FOUND'}")
print(f"  Diab.  Scaler : {'✓ Loaded' if diabetes_scaler else '✗ NOT FOUND'}")
print(f"  Heart  Disease Class Index : {HEART_DISEASE_CLASS}")
print("════════════════════════════════════════\n")


# ── XAI: Risk factor extractors ───────────────────────────
def heart_risk_factors(age, sex, cp, trestbps, chol, fbs,
                        thalach, exang, oldpeak, slope, ca, thal):
    factors = []
    if chol > 240:
        factors.append(f"High Cholesterol ({chol:.0f} mg/dL)")
    if oldpeak > 1.5:
        factors.append(f"Elevated ST Depression ({oldpeak:.1f})")
    if thalach < 120:
        factors.append(f"Low Max Heart Rate ({thalach:.0f} bpm)")
    if trestbps > 140:
        factors.append(f"High Resting BP ({trestbps:.0f} mmHg)")
    if int(exang) == 1:
        factors.append("Exercise-Induced Angina present")
    if int(ca) >= 2:
        factors.append(f"Multiple Fluoroscopy Vessels ({int(ca)})")
    if int(thal) == 3:
        factors.append("Reversible Thalassemia Defect")
    if int(cp) == 0:
        factors.append("Typical Angina pattern")
    return factors


def diabetes_risk_factors(glucose, bmi, age, dpf, bp, insulin):
    factors = []
    if glucose > 140:
        factors.append(f"High Plasma Glucose ({glucose:.0f} mg/dL)")
    if bmi > 30:
        factors.append(f"Elevated BMI ({bmi:.1f} kg/m²)")
    if age > 40:
        factors.append(f"Age Risk Factor ({age:.0f} yrs)")
    if dpf > 0.5:
        factors.append(f"High Pedigree Function ({dpf:.3f})")
    if bp > 90:
        factors.append(f"Elevated Diastolic BP ({bp:.0f} mmHg)")
    if insulin == 0:
        factors.append("Missing Insulin reading (0)")
    return factors


# ── Recommendation generators ─────────────────────────────
def heart_recommendation(is_high_risk):
    if is_high_risk:
        return (
            "Immediate cardiology referral is recommended. Prioritise lipid panel review, "
            "stress ECG, and echocardiography. Monitor blood pressure closely and evaluate "
            "antiplatelet therapy. Lifestyle interventions including diet and physical activity "
            "modifications are strongly advised."
        )
    return (
        "Current cardiac biomarkers indicate low risk. Continue annual cardiovascular "
        "screening. Maintain healthy cholesterol levels (<200 mg/dL), regular exercise, "
        "and blood pressure monitoring. Follow-up in 12 months or sooner if symptoms develop."
    )


def diabetes_recommendation(is_diabetic):
    if is_diabetic:
        return (
            "Diabetes is indicated. Recommend HbA1c testing for confirmation and fasting "
            "glucose follow-up. Consult an endocrinologist for a structured diabetes management "
            "plan including dietary counselling, weight management, and blood glucose monitoring. "
            "Evaluate for complications including neuropathy and retinopathy."
        )
    return (
        "No diabetes detected at this time. Maintain healthy glucose levels through "
        "balanced diet and regular physical activity. Annual re-screening is recommended, "
        "especially if BMI > 25 or family history is present. Monitor fasting glucose "
        "annually as a preventive measure."
    )


# ── Routes ────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    if not heart_model or not heart_scaler:
        return jsonify({
            'status': 'error',
            'message': 'Heart disease model/scaler not loaded. Check model files.'
        })

    try:
        fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                  'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

        input_data  = [float(request.form[f]) for f in fields]
        scaled_data = heart_scaler.transform([input_data])

        # ── KEY FIX ──────────────────────────────────────────
        # Use the disease-class probability (auto-detected) as the
        # canonical risk score. This is robust to label-encoding
        # differences across UCI Heart Disease dataset versions.
        proba        = heart_model.predict_proba(scaled_data)[0]
        disease_prob = proba[HEART_DISEASE_CLASS]   # probability of having heart disease
        is_high_risk = disease_prob >= 0.5
        # ─────────────────────────────────────────────────────

        result     = "High Cardiac Risk" if is_high_risk else "Low Cardiac Risk"
        confidence = round(disease_prob * 100, 1)

        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = input_data

        risk_factors = heart_risk_factors(
            age, sex, cp, trestbps, chol, fbs,
            thalach, exang, oldpeak, slope, ca, thal
        )

        if risk_factors:
            xai_msg = (
                "The model identified elevated risk based on: "
                + "; ".join(risk_factors[:4])
                + f". Combined biomarker patterns raised the risk probability to {confidence}%."
            )
        else:
            xai_msg = (
                f"All key biomarkers are within acceptable clinical ranges. "
                f"The model assigned a low cardiac risk probability of {confidence}%."
            )

        scores = {
            'Cholesterol':    round(min(chol / 300, 1.0), 3),
            'Max Heart Rate': round(min(thalach / 200, 1.0), 3),
            'ST Depression':  round(min(oldpeak / 4, 1.0), 3),
            'Resting BP':     round(min(trestbps / 180, 1.0), 3),
            'Age Factor':     round(min(age / 80, 1.0), 3),
        }

        return jsonify({
            'status':         'success',
            'result':         result,
            'confidence':     f"{confidence}",
            'xai':            xai_msg,
            'scores':         scores,
            'risk_factors':   risk_factors,
            'recommendation': heart_recommendation(is_high_risk),
        })

    except KeyError as e:
        return jsonify({'status': 'error', 'message': f'Missing input field: {e}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if not diabetes_model or not diabetes_scaler:
        return jsonify({
            'status': 'error',
            'message': 'Diabetes model/scaler not loaded. Check model files.'
        })

    try:
        fields = ['preg', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'dpf', 'age']
        input_data  = [float(request.form[f]) for f in fields]

        scaled_data  = diabetes_scaler.transform([input_data])
        proba        = diabetes_model.predict_proba(scaled_data)[0]
        disease_prob = proba[DIABETES_DISEASE_CLASS]
        is_diabetic  = disease_prob >= 0.5

        result     = "Diabetes Suspected" if is_diabetic else "No Diabetes Detected"
        confidence = round(disease_prob * 100, 1)

        preg, glucose, bp, skin, insulin, bmi, dpf, age = input_data

        risk_factors = diabetes_risk_factors(glucose, bmi, age, dpf, bp, insulin)

        if risk_factors:
            xai_msg = (
                "Key metabolic indicators contributing to this assessment: "
                + "; ".join(risk_factors[:4])
                + f". The combined risk profile yields a {confidence}% diabetes probability."
            )
        else:
            xai_msg = (
                f"Metabolic biomarkers are within normal ranges. "
                f"The model estimates a {confidence}% diabetes probability."
            )

        scores = {
            'Glucose Level': round(min(glucose / 200, 1.0), 3),
            'BMI Index':     round(min(bmi / 50, 1.0), 3),
            'Age Factor':    round(min(age / 100, 1.0), 3),
            'Pedigree Fn':   round(min(dpf / 2.5, 1.0), 3),
            'Diastolic BP':  round(min(bp / 120, 1.0), 3),
        }

        return jsonify({
            'status':         'success',
            'result':         result,
            'confidence':     f"{confidence}",
            'xai':            xai_msg,
            'scores':         scores,
            'risk_factors':   risk_factors,
            'recommendation': diabetes_recommendation(is_diabetic),
        })

    except KeyError as e:
        return jsonify({'status': 'error', 'message': f'Missing input field: {e}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
