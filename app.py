from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import json

app = Flask(__name__)

# Load constants from JSON file
with open('constants.json', 'r') as json_file:
    constants_data = json.load(json_file)

NUMERIC_COLUMNS = constants_data['NUMERIC_COLUMNS']
CHAMPION_AMOUNT = constants_data['CHAMPION_AMOUNT']
TEAM_AMOUNT = constants_data['TEAM_AMOUNT']

# Load the trained model
model = tf.keras.models.load_model('LolModel') 

# Create feature columns
feature_columns = [tf.feature_column.numeric_column(feature_name, dtype=tf.float32) for feature_name in NUMERIC_COLUMNS]

# Create input function for prediction
def make_predict_input_fn(data_dict):
    if 'opponent_earned_gpm_rolling' in data_dict:
        print("ASDSADDASDASDDAS")

    data = {key: [value] for key, value in data_dict.items()}
    dataset = tf.data.Dataset.from_tensor_slices((data))
    dataset = dataset.batch(1)
    return dataset

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        input_data = {}
        NUMERIC_COLUMNS.pop(-1)
        for column in NUMERIC_COLUMNS:
            print(column)
            input_data[column] = float(request.form[column])

        # Make prediction
        input_fn = make_predict_input_fn(input_data)
        prediction = model.predict(input_fn)

        # Format the prediction result
        prediction_result = int(np.round(prediction[0][0]))

        return render_template('index.html', NUMERIC_COLUMNS=NUMERIC_COLUMNS, prediction_result=prediction_result)

    return render_template('index.html', NUMERIC_COLUMNS=NUMERIC_COLUMNS)

if __name__ == '__main__':
    app.run(debug=True)
