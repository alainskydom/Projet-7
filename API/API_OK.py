#import matplotlib
#import matplotlib.pyplot as plt
import json
#from typing import List
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import io
import os
#import uuid






# Charger le meilleur modèle
try:
    best_model = pickle.load(open(r"API/model.pkl", 'rb'))
except FileNotFoundError:
    print("Error: Model file not found.  Make sure 'best_model.pkl' is in the correct directory.")
    best_model = None # Handle the case where the model couldn't be loaded

# Charger les données
df_ = pd.read_csv(r"API/df_.csv")[0:100]
df_=df_.loc[:, ~df_.columns.str.match ('Unnamed')]
df_ = df_.drop(['TARGET', 'SK_ID_CURR'], axis=1)
# df.drop(columns='index', inplace=True)

# Define the threshold of for application.
threshold = 0.6


# --- Flask Setup ---
app = Flask(__name__)

# --- API Endpoints ---
@app.route('/predict/', methods=['POST'])
def predict():
    """
    Takes a string 'XXX' (renamed as index 'idx'), retrieves corresponding
    features from the dataframe, and makes a prediction using the
    pre-trained model.
    """

    class NumpyEncoder(json.JSONEncoder):
        """ Custom encoder for numpy data types """

    try:
        data = request.get_json()
        idx = data['client_id']
    except (KeyError, TypeError):
        return jsonify({"error": "Invalid input: 'client_id' key missing or incorrect JSON format."}), 400

    try:
        idx = int(idx) #convert to integer to match index
    except ValueError:
        return jsonify({"error": "Invalid input: 'client_id' must be an integer string."}), 400


    # Check if the index exists in the DataFrame
    if idx not in df_.index:
        try:
            df_.loc[int(idx)] # Try getting by int. Raises error if does not exist
        except KeyError:
            return jsonify({"error": f"Index '{idx}' not found in the DataFrame."}), 404

    # Extract features from the DataFrame row
    try:
        features = df_.loc[idx, df_.columns].to_numpy().reshape(1, -1)  # Or other feature columns. to_numpy is more robust for different pandas versions.
        features_1= np.array2string(features)
    except KeyError:
        return jsonify({"error": f"Index '{idx}' not found in DataFrame."}), 404

    # Make the prediction




    try:
        #prediction = best_model.predict(features)[0]
        score = best_model.predict_proba(features)[:, 1]
        df1 = df_.copy()
        #prediction_json=np.array2string(prediction)
        score_json = np.array2string(score)
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

    return jsonify({"prediction": score_json, "index": idx}), 200

# Create a DataFrame from the prediction result

    df_proba = pd.DataFrame(score, columns=['proba'],index=idx)

    #prediction_data = {'index': [idx], 'prediction': [prediction]}
    #prediction_df = pd.DataFrame(prediction_data)

    # for add prediction, used threshold value
    df_proba['Predict'] = np.where(df_proba['proba'] < threshold, 0, 1)

    df1['Proba_Score'] = df_proba['proba']
    df1['Predict'] = df_proba['Predict']

    #JSON format!
    df2 = df1.to_json()
    dict_result = {'Credit_score': score[0], "json_data": data, 'Total_score': df2}

    # for json format some values are categorical, however it is difficult to handle these values as float,
    # these values types are changed by using JSON encoder
    class NumpyFloatValuesEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)
            return json.JSONEncoder.default(self, obj)

    # JSON format dumps method for send the data to Dashboard
    dict_result = json.dumps(dict_result)

    # Each request of dashboard, df1 dataframe adding ['Proba_Score', 'Predict'] columns,
    # so It needs to drop these columns at the end of the API
    df1.drop(['Proba_Score', 'Predict'], axis=1)

    return dict_result

    # Create an in-memory CSV file
    csv_buffer = io.StringIO()
    #prediction_df.to_csv(csv_buffer, index=False)
    df_proba.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Generate a filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prediction_{idx}_{timestamp}.csv"

    # Return the CSV file as a downloadable attachment
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode()),  # Encode to bytes
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )


# --- Running the API ---
# To run this API:
# 1. Save the code as a Python file (e.g., main.py).
# 2. Install the necessary libraries
# 3. Run the API using: python main.py
# 4. Access the API by sending a POST request to http://127.0.0.1:5000/predict/
#    with a JSON body like {"XXX": "0"}

if __name__ == '__main__':
    app.run(debug=True) # For development purposes, use debug=True.  For production, configure a proper WSGI server.
