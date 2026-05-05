import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

app = Flask(__name__)

# --- Logic from copy_of_welcome_to_colab.py ---
def initialize_model():
    # Loading dataset
    df = pd.read_csv('diabetes.csv')
    X = df.drop(columns='Outcome', axis=1)
    Y = df['Outcome']

    # Data Standardization
    scaler = StandardScaler()
    scaler.fit(X)
    X_standardized = scaler.transform(X)

    # Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_standardized, Y, test_size=0.2, stratify=Y, random_state=2
    )

    # Training the Model[cite: 1]
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)

    # Save components
    with open('model.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return classifier, scaler

# Load or Train
if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
else:
    model, scaler = initialize_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from HTML form
        input_values = [float(x) for x in request.form.values()]
        
        # Predictive System Logic[cite: 1]
        input_array = np.asarray(input_values).reshape(1, -1)
        std_data = scaler.transform(input_array)
        prediction = model.predict(std_data)

        if prediction[0] == 0:
            result = 'The person is not diabetic'
            res_class = 'success'
        else:
            result = 'The person is diabetic'
            res_class = 'danger'

        return render_template('index.html', prediction_text=result, res_class=res_class)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)