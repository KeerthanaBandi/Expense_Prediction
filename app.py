import numpy as np
from flask import Flask,jsonify, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    age = float(request.form['age'])
    income = float(request.form['income'])

    # Prepare input features for prediction
    features = np.array([[age, income]])
    prediction = float(model.predict(features)[0])

    # Render result back to the HTML template
    return render_template('index.html', prediction_text=f'Predicted Expense: ${prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)