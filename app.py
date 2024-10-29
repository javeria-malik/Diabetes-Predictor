from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Home route
@app.route('/')
def home():
    return render_template('home.html', title="Home - Diabetes Predictor")

# Prediction page route
@app.route('/predict_page')
def predict_page():
    return render_template('predict.html', title="Predict Diabetes")

# Prediction processing route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = [float(request.form[field]) for field in request.form]
    
    # Preprocess data
    scaled_data = scaler.transform([data])
    
    # Make prediction
    prediction = model.predict(scaled_data)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    
    return render_template('predict.html', prediction_text=f'Result: {result}', title="Predict Diabetes")

if __name__ == '__main__':
    app.run(debug=True)
