import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model and the scaler
regmodel = pickle.load(open('regmodel_new.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))  # Load the scaler here

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    
    # Reshaping the input data and scaling it
    input_data = np.array(list(data.values())).reshape(1, -1)
    new_data = scalar.transform(input_data)  # Transform input data using the scaler
    
    # Make prediction using the model
    output = regmodel.predict(new_data)
    
    print(output[0])  # For debugging purposes
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text='The House price prediction is{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

    app.run(debug=True)