from flask import Flask, request, jsonify
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras


app = Flask(__name__)




@app.route('/api/v1/coverage', methods=['POST'])
def predict():
    # Load the trained model from a file
    insurance_model = tf.keras.models.load_model('insurance/coveraged_dec')
    # Get the input data from the request
    data = request.get_json(force=True)
    age = data['age']
    number_of_accidents = data['number_of_accidents']
    number_of_tickets = data['number_of_tickets']
    married = data['married']
    owns_car = data['owns_car']
    years_licensed = data['years_licensed']

    # Make a prediction using the loaded model
    input_data = np.array([[age, number_of_tickets, number_of_accidents, married, owns_car, years_licensed ]])
    prediction = cov_model.predict(input_data, verbose=0)

    response = jsonify({'prediction': prediction})
    return response



if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
