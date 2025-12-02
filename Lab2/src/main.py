# from flask import Flask, request, jsonify, render_template
# import tensorflow as tf
# import numpy as np

# app = Flask(__name__, static_folder='statics')

# # URL of your Flask API for making predictions
# api_url = 'http://0.0.0.0:4000/predict'  # Update with the actual URL

# # Load the TensorFlow model
# model = tf.keras.models.load_model('my_model.keras')  # Replace 'my_model.keras' with the actual model file
# class_labels = ['Setosa', 'Versicolor', 'Virginica']


# """Modern web apps use a technique named routing. This helps the user remember the URLs. 
# For instance, instead of having /booking.php they see /booking/. Instead of /account.asp?id=1234/ 
# they’d see /account/1234/."""

# @app.route('/')
# def home():
#     return "Welcome to the Iris Classifier API!"

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             data = request.form
#             sepal_length = float(data['sepal_length'])
#             sepal_width = float(data['sepal_width'])
#             petal_length = float(data['petal_length'])
#             petal_width = float(data['petal_width'])

#             # Perform the prediction
#             input_data = np.array([sepal_length, sepal_width, petal_length, petal_width])[np.newaxis, ]
#             prediction = model.predict(input_data)
#             predicted_class = class_labels[np.argmax(prediction)]

#             # Return the predicted class in the response
#             # Use jsonify() instead of json.dumps() in Flask
#             return jsonify({"predicted_class": predicted_class})
#         except Exception as e:
#             return jsonify({"error": str(e)})
#     elif request.method == 'GET':
#         return render_template('predict.html')
#     else:
#         return "Unsupported HTTP method"

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=4000)

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import joblib
import json

app = Flask(__name__, static_folder='statics', template_folder='templates')

# Load model metadata
print("="*60)
print("DIABETES PREDICTION SYSTEM - Starting...")
print("="*60)

with open('model_info.json', 'r') as f:
    model_info = json.load(f)

best_model_name = model_info['best_model_name']
best_accuracy = model_info['best_model_accuracy']
model_type = model_info['model_type']

print(f"\nLoading trained model...")
print(f"  Model: {best_model_name}")
print(f"  Accuracy: {best_accuracy*100:.2f}%")

# Load the best model
if model_type == 'neural_network':
    model = tf.keras.models.load_model('best_model.keras')
    print(f"  ✓ Neural Network model loaded")
else:
    model = joblib.load('best_model.pkl')
    print(f"  ✓ {best_model_name} model loaded")

# Load scaler
scaler = joblib.load('scaler.pkl')
print(f"  ✓ Feature scaler loaded")

print("\n" + "="*60)
print("System ready! Access at: http://0.0.0.0:4000/")
print("="*60 + "\n")

@app.route('/')
def home():
    return render_template('predict.html', 
                         model_name=best_model_name,
                         model_accuracy=round(best_accuracy*100, 2))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        
        # Get patient name
        patient_name = data.get('patient_name', 'Patient').strip()
        if not patient_name:
            patient_name = 'Patient'
        
        # Extract features
        features = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['blood_pressure']),
            float(data['skin_thickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetes_pedigree']),
            float(data['age'])
        ]
        
        # Prepare input
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        if model_type == 'neural_network':
            prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
            prediction = int(prediction_prob > 0.5)
        else:
            prediction = model.predict(input_scaled)[0]
            if hasattr(model, 'predict_proba'):
                prediction_prob = model.predict_proba(input_scaled)[0][1]
            else:
                prediction_prob = float(prediction)
        
        # Calculate risk percentage
        risk_percentage = float(prediction_prob * 100)
        
        # Determine risk level
        if risk_percentage < 30:
            risk_level = "Low"
            risk_category = "low"
        elif risk_percentage < 60:
            risk_level = "Moderate"
            risk_category = "moderate"
        else:
            risk_level = "High"
            risk_category = "high"
        
        # Risk factors analysis
        risk_factors = []
        if features[1] > 140:  # Glucose
            risk_factors.append("Elevated blood glucose levels")
        if features[5] > 30:  # BMI
            risk_factors.append("Body Mass Index indicates overweight")
        if features[2] > 90:  # Blood Pressure
            risk_factors.append("High blood pressure detected")
        if features[7] > 45:  # Age
            risk_factors.append("Age-related risk factor")
        if features[0] > 5:  # Pregnancies
            risk_factors.append("Multiple pregnancies")
        
        # Recommendations
        recommendations = []
        if risk_category == "high":
            recommendations = [
                "Schedule an appointment with your doctor as soon as possible",
                "Get a comprehensive diabetes screening test (HbA1c)",
                "Monitor your blood sugar levels regularly",
                "Adopt a low-sugar, balanced diet",
                "Exercise for at least 30 minutes daily",
                "Maintain a healthy weight"
            ]
        elif risk_category == "moderate":
            recommendations = [
                "Consult with your healthcare provider for proper evaluation",
                "Monitor your health metrics regularly",
                "Maintain a balanced, healthy diet",
                "Stay physically active",
                "Schedule regular check-ups"
            ]
        else:
            recommendations = [
                "Continue your healthy lifestyle",
                "Maintain regular physical activity",
                "Eat a balanced diet",
                "Get annual health check-ups",
                "Stay hydrated and get adequate sleep"
            ]
        
        return jsonify({
            "patient_name": patient_name,
            "prediction": int(prediction),
            "risk_level": risk_level,
            "risk_category": risk_category,
            "risk_percentage": round(risk_percentage, 1),
            "risk_factors": risk_factors,
            "recommendations": recommendations
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)