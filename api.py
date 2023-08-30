from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from waitress import serve
import tensorflow_decision_forests as tfdf

app = Flask(__name__)

# Load the trained model from a file
try:
    insurance_model = tf.keras.models.load_model('insurance/coverage_dec')
    print(insurance_model.summary())
except Exception as e:
    print("Error loading model", e)
    insurance_model = None


@app.route('/api/v1/coverage', methods=['POST'])
def predict():
    print("Predicting")
    if insurance_model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get the input data from the request
        data = request.get_json(force=True)
        age = data['age']
        number_of_accidents = data['number_of_accidents']
        number_of_tickets = data['number_of_tickets']
        married = int(data['married'])  # Convert boolean to int
        owns_car = int(data['owns_car'])  # Convert boolean to int
        years_licensed = data['years_licensed']

        # Convert booleans to int
        married_int = int(data['married'])
        owns_car_int = int(data['owns_car'])

        # Create a 1D NumPy array
        input_df = pd.DataFrame([{
            'age': age,
            'number_of_accidents': number_of_accidents,
            'number_of_tickets': number_of_tickets,
            'married': married,
            'owns_car': owns_car,
            'years_licensed': years_licensed
        }])
        tf_serving_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(input_df, label=None)

        prediction = insurance_model.predict(tf_serving_dataset,
                                             verbose=0).tolist()  # Converting numpy array to list for jsonify

        return jsonify({'prediction': prediction[0]})
    except Exception as er:
        print("Error during prediction", er)
        return jsonify({'error': 'Error during prediction'}), 500


if __name__ == '__main__':
    print("Starting")
    serve(app, host="0.0.0.0", port=8080)
