from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained SVM model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        inputs = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        # Convert inputs to numpy array
        input_data = np.array([inputs])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Convert prediction to human-readable output
        if prediction[0] == 1:
            result = "Positive (Diabetic)"
        else:
            result = "Negative (Not Diabetic)"
        
        return render_template('index.html', prediction_text=f'Diabetes Prediction: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
