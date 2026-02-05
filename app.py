from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# --- Configuration and Model Loading ---
MODEL_PATH_HEART = 'heart_model.pkl'
MODEL_PATH_DIABETES = 'diabetes_model.pkl'
DATA_PATH_HEART = 'heart.csv'
DATA_PATH_DIABETES = 'diabetes.csv'
RANDOM_STATE = 42 # Ensure consistency with training split

# Load models
try:
    with open(MODEL_PATH_HEART, 'rb') as f:
        heart_model = pickle.load(f)
    print(f"Heart Disease model loaded from {MODEL_PATH_HEART}")

    with open(MODEL_PATH_DIABETES, 'rb') as f:
        diabetes_model = pickle.load(f)
    print(f"Pima Indians Diabetes model loaded from {MODEL_PATH_DIABETES}")
except FileNotFoundError as e:
    print(f"Error loading model: {e}. Make sure model files are in the correct directory.")
    exit() # Exit if models cannot be loaded

# Load original datasets for fitting preprocessors
try:
    heart_df_original = pd.read_csv(DATA_PATH_HEART)
    diabetes_df_original = pd.read_csv(DATA_PATH_DIABETES)
    print("Original datasets loaded for preprocessor fitting.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure data files are in the correct directory.")
    exit() # Exit if data cannot be loaded

# --- Heart Disease Preprocessing Setup ---
heart_numerical_cols = heart_df_original.select_dtypes(include=np.number).columns.tolist()
heart_features = [col for col in heart_numerical_cols if col != 'target']

# Split heart data to get training set for fitting imputer/scaler
X_heart_for_fit, _, _, _ = train_test_split(
    heart_df_original[heart_features],
    heart_df_original['target'],
    test_size=0.2,
    random_state=RANDOM_STATE
)

# Create and fit SimpleImputer for heart disease data
heart_imputer = SimpleImputer(strategy='mean')
heart_imputer.fit(X_heart_for_fit)
print("heart_imputer fitted successfully.")

# Create and fit StandardScaler for heart disease data
heart_scaler = StandardScaler()
heart_scaler.fit(heart_imputer.transform(X_heart_for_fit))
print("heart_scaler fitted successfully.")

# --- Pima Indians Diabetes Preprocessing Setup ---
dia_cols_to_impute_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] # Renamed for clarity with variable name guidelines

# Create a copy for preprocessing steps before splitting
diabetes_df_for_fit = diabetes_df_original.copy()
for col in dia_cols_to_impute_zeros:
    if col in diabetes_df_for_fit.columns:
        diabetes_df_for_fit[col] = diabetes_df_for_fit[col].replace(0, np.nan)

diabetes_numerical_cols = diabetes_df_for_fit.select_dtypes(include=np.number).columns.tolist()
diabetes_features = [col for col in diabetes_numerical_cols if col != 'Outcome']

# Split diabetes data to get training set for fitting imputer/scaler
X_diabetes_for_fit, _, _, _ = train_test_split(
    diabetes_df_for_fit[diabetes_features],
    diabetes_df_for_fit['Outcome'],
    test_size=0.2,
    random_state=RANDOM_STATE
)

# Create and fit SimpleImputer for diabetes data
diabetes_imputer = SimpleImputer(strategy='mean')
diabetes_imputer.fit(X_diabetes_for_fit)
print("diabetes_imputer fitted successfully.")

# Create and fit StandardScaler for diabetes data
diabetes_scaler = StandardScaler()
diabetes_scaler.fit(diabetes_imputer.transform(X_diabetes_for_fit))
print("diabetes_scaler fitted successfully.")


# --- Web Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    """
    Predicts heart disease based on input features from a form.
    """
    try:
        data = request.form.to_dict()
        # Convert form data to appropriate types
        input_data = {
            'age': float(data['age']),
            'sex': float(data['sex']),
            'cp': float(data['cp']),
            'trestbps': float(data['trestbps']),
            'chol': float(data['chol']),
            'fbs': float(data['fbs']),
            'restecg': float(data['restecg']),
            'thalach': float(data['thalach']),
            'exang': float(data['exang']),
            'oldpeak': float(data['oldpeak']),
            'slope': float(data['slope']),
            'ca': float(data['ca']),
            'thal': float(data['thal'])
        }

        input_df = pd.DataFrame([input_data], columns=heart_features)

        # Apply preprocessing steps (imputation and scaling)
        imputed_data = heart_imputer.transform(input_df)
        scaled_data = heart_scaler.transform(imputed_data)

        # Make prediction
        prediction = heart_model.predict(scaled_data)
        prediction_proba = heart_model.predict_proba(scaled_data)

        return render_template(
            'result.html',
            prediction_type='Heart Disease',
            prediction=int(prediction[0]),
            proba_0=float(prediction_proba[0][0]),
            proba_1=float(prediction_proba[0][1]),
            prediction_type_display='Heart Disease'
        )
    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    """
    Predicts diabetes based on input features from a form.
    """
    try:
        data = request.form.to_dict()
        # Convert form data to appropriate types, handling '0' as NaN if applicable
        input_data = {
            'Pregnancies': float(data['Pregnancies']),
            'Glucose': float(data['Glucose']),
            'BloodPressure': float(data['BloodPressure']),
            'SkinThickness': float(data['SkinThickness']),
            'Insulin': float(data['Insulin']),
            'BMI': float(data['BMI']),
            'DiabetesPedigreeFunction': float(data['DiabetesPedigreeFunction']),
            'Age': float(data['Age'])
        }
        input_df = pd.DataFrame([input_data], columns=diabetes_features)

        # Handle '0' values as NaN for specific diabetes features
        for col in dia_cols_to_impute_zeros:
            if col in input_df.columns:
                input_df[col] = input_df[col].replace(0, np.nan)

        # Apply preprocessing steps (imputation and scaling)
        imputed_data = diabetes_imputer.transform(input_df)
        scaled_data = diabetes_scaler.transform(imputed_data)

        # Make prediction
        prediction = diabetes_model.predict(scaled_data)
        prediction_proba = diabetes_model.predict_proba(scaled_data)

        return render_template(
            'result.html',
            prediction_type='Pima Indians Diabetes',
            prediction=int(prediction[0]),
            proba_0=float(prediction_proba[0][0]),
            proba_1=float(prediction_proba[0][1]),
            prediction_type_display='Pima Indians Diabetes'
        )
    except Exception as e:
        return render_template('result.html', error=str(e))

# --- Run Flask Application ---
if __name__ == '__main__':
    # For local development, remove debug=True and host='0.0.0.0' for production
    # host='0.0.0.0' makes the server accessible from any IP on the network
    # In Colab, you would typically use ngrok to expose this.
    app.run(debug=True, host='0.0.0.0', port=5000)
