from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/api/v1/coverage', methods=['POST'])
def predict():
    # Load the trained model from a file
    loaded_model = pickle.load(open("auto_insurance_premiums_model.pkl.pkl", 'rb'))
    # Get the input data from the request
    data = request.get_json(force=True)
    Age = data['age']
    number_of_accidents = data['number_of_accidents']
    number_of_tickets = data['number_of_tickets']
    married = data['married']
    owns_car = data['owns_car']
    years_licensed = data['years_licensed']
    term = data['term']
    Gender = data["gender"]
    PaidOut = data["paid_out"]
    Coverage = data["coverage"]
    CarPrice = data["car_price"]
    Type = data["type"]

    # Make a prediction using the loaded model
    input_data = np.array([[Age, number_of_tickets, number_of_accidents, married, owns_car, years_licensed, term, Gender, PaidOut, Coverage, CarPrice, Type]])
    prediction = loaded_model.predict(input_data)

    # Return the prediction as a JSON response
    response = jsonify({'prediction': prediction.tolist()})

    return response


if __name__ == '__main__':
    app.run(debug=True)
