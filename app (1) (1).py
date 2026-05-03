"""
MedPredict — Flask Prediction API
══════════════════════════════════════════════════════════
B.Tech Final Year Project | Medical AI with XAI
Supports: Heart Disease (13 features) + Diabetes (8 features)
══════════════════════════════════════════════════════════
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Helper: safe model loader ─────────────────────────────
def load_pickle(path: str):
    """Load a pickle file. Returns None on failure."""
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load {path}: {e}")
        return None


# ── Load Heart Disease Model & Scaler ─────────────────────
# The uploaded file is named 'heart_model__1_.pkl'
# Rename it to 'heart_model.pkl' in production, OR keep both paths below.
heart_model  = load_pickle('heart_model.pkl') or load_pickle('heart_model__1_.pkl')
heart_scaler = load_pickle('heart_scaler.pkl')

# ── Load Diabetes Model & Scaler ─────────────────────────
diabetes_model  = load_pickle('diabetes_model.pkl')
diabetes_scaler = load_pickle('diabetes_scaler.pkl')

# ── Startup status ────────────────────────────────────────
print("\n══════════════════════════════════════")
print("  MedPredict — Model Load Status")
print("══════════════════════════════════════")
print(f"  Heart  Model  : {'✓ Loaded' if heart_model  else '✗ NOT FOUND'}")
print(f"  Heart  Scaler : {'✓ Loaded' if heart_scaler else '✗ NOT FOUND'}")
print(f"  Diab.  Model  : {'✓ Loaded' if diabetes_model  else '✗ NOT FOUND'}")
print(f"  Diab.  Scaler : {'✓ Loaded' if diabetes_scaler else '✗ NOT FOUND'}")
print("══════════════════════════════════════\n")


# ── Routes ────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    """
    Heart disease prediction.
    Expected POST fields (13 features — UCI Cleveland Dataset):
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    """
    if not heart_model or not heart_scaler:
        return jsonify({
            'status':  'error',
            'message': 'Heart disease model / scaler not loaded. '
                       'Check that heart_model.pkl and heart_scaler.pkl '
                       'are present in the project directory.'
        })

    try:
        fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                  'restecg', 'thalach', 'exang', 'oldpeak',
                  'slope', 'ca', 'thal']

        input_data = [float(request.form[f]) for f in fields]

        scaled_data = heart_scaler.transform([input_data])
        prediction  = heart_model.predict(scaled_data)[0]
        probability = heart_model.predict_proba(scaled_data)[0][1]

        result     = "High Cardiac Risk" if int(prediction) == 1 else "Low Cardiac Risk"
        confidence = f"{round(probability * 100, 1)}%"

        # XAI explanation (rule-based; replace with SHAP/LIME for production)
        age       = int(input_data[0])
        chol      = input_data[4]
        oldpeak   = input_data[9]
        thalach   = input_data[7]
        cp        = int(input_data[2])

        xai_parts = []
        if chol > 240:
            xai_parts.append(f"elevated cholesterol ({chol:.0f} mg/dL, normal < 200)")
        if oldpeak > 1.5:
            xai_parts.append(f"significant ST depression (Oldpeak {oldpeak:.1f}, indicating ischaemia)")
        if thalach < 120:
            xai_parts.append(f"low maximum heart rate ({thalach:.0f} bpm, suggesting reduced cardiac reserve)")
        if cp == 0:
            xai_parts.append("typical angina chest pain type (CP = 0)")
        if age > 55:
            xai_parts.append(f"age ({age} years) as a non-modifiable risk factor")

        if not xai_parts:
            xai_msg = (f"No single dominant biomarker exceeded critical thresholds. "
                       f"The model assessed combined risk from cholesterol ({chol:.0f} mg/dL), "
                       f"max heart rate ({thalach:.0f} bpm), and ST depression ({oldpeak:.1f}).")
        else:
            xai_msg = ("Primary contributing factors identified by the model: "
                       + "; ".join(xai_parts) + ".")

        return jsonify({
            'status':     'success',
            'result':     result,
            'confidence': confidence,
            'xai':        xai_msg
        })

    except KeyError as e:
        return jsonify({'status': 'error', 'message': f'Missing field: {e}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    """
    Diabetes prediction.
    Expected POST fields (8 features — Pima Indian Diabetes Dataset):
        preg, glucose, bp, skin, insulin, bmi, dpf, age
    """
    if not diabetes_model or not diabetes_scaler:
        return jsonify({
            'status':  'error',
            'message': 'Diabetes model / scaler not loaded. '
                       'Check that diabetes_model.pkl and diabetes_scaler.pkl '
                       'are present in the project directory.'
        })

    try:
        fields = ['preg', 'glucose', 'bp', 'skin',
                  'insulin', 'bmi', 'dpf', 'age']

        input_data = [float(request.form[f]) for f in fields]

        scaled_data = diabetes_scaler.transform([input_data])
        prediction  = diabetes_model.predict(scaled_data)[0]
        probability = diabetes_model.predict_proba(scaled_data)[0][1]

        result     = "Diabetes Suspected" if int(prediction) == 1 else "No Diabetes Detected"
        confidence = f"{round(probability * 100, 1)}%"

        # XAI explanation
        preg    = int(input_data[0])
        glucose = input_data[1]
        bmi     = input_data[5]
        dpf     = input_data[6]
        age     = int(input_data[7])

        xai_parts = []
        if glucose > 140:
            xai_parts.append(f"high fasting glucose ({glucose:.0f} mg/dL, threshold > 140)")
        if bmi > 30:
            xai_parts.append(f"BMI indicating obesity ({bmi:.1f} kg/m², threshold > 30)")
        if dpf > 0.5:
            xai_parts.append(f"elevated Diabetes Pedigree Function ({dpf:.3f}, indicating family history influence)")
        if preg > 5:
            xai_parts.append(f"high number of pregnancies ({preg})")
        if age > 40:
            xai_parts.append(f"age ({age} years) as a contributing metabolic factor")

        if not xai_parts:
            xai_msg = (f"No single biomarker exceeded its critical threshold. "
                       f"The model considered glucose ({glucose:.0f} mg/dL), "
                       f"BMI ({bmi:.1f}), and DPF ({dpf:.3f}) in combination.")
        else:
            xai_msg = ("Primary contributing factors identified by the model: "
                       + "; ".join(xai_parts) + ".")

        return jsonify({
            'status':     'success',
            'result':     result,
            'confidence': confidence,
            'xai':        xai_msg
        })

    except KeyError as e:
        return jsonify({'status': 'error', 'message': f'Missing field: {e}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)