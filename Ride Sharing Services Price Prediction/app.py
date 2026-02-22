from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model and the unique values CSV
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
destination_encoder = joblib.load('destination_encoder.pkl')
source_encoder = joblib.load('source_encoder.pkl')
name_encoder = joblib.load('name_encoder.pkl')

unique_values = pd.read_csv('unique_values.csv')

for column in ['cab_type', 'surge_multiplier']:
    # Get the unique cleaned values as a list
    cleaned_values = unique_values[column].replace(['NaN', 'nan', ''], 'NA').dropna().unique().tolist()
    
    # Replace the entire column with the cleaned list
    unique_values[column] = cleaned_values + ['NA'] * (len(unique_values) - len(cleaned_values))



data = pd.read_csv('cab_rides.csv')

def preprocess_input_data(data):
    df = pd.DataFrame([data])
    df['cab_type'] = df['cab_type'].replace({'Lyft': 0, 'Uber': 1})
    
    df['destination'] = destination_encoder.transform(df['destination'])
    df['source'] = source_encoder.transform(df['source'])
    df['name'] = name_encoder.transform(df['name'])
    
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    
    return df

def get_distance(source, destination):
    # Extract the rows that match the given source and destination
    matching_rows = data[(data['source'] == source) & (data['destination'] == destination)]
    if matching_rows.shape[0] > 0:
        return matching_rows.iloc[0]['distance']
    else:
        return None

@app.route("/")
def index():
    print(unique_values.head(10))
    return render_template('index.html', data=unique_values)

@app.route("/get-distance", methods=["POST"])
def get_distance_endpoint():
    source = request.form['source']
    # print(source)
    destination = request.form['destination']
    distance = get_distance(source, destination)
    print(distance)
    return jsonify({'distance': distance})


@app.route("/predict", methods=["POST"])
def predict():
    # Extract the data from the POST request
    data = {
        'distance': float(request.form['distance']),
        'cab_type': request.form['cab_type'],
        'destination': request.form['destination'],
        'source': request.form['source'],
        'surge_multiplier': float(request.form['surge_multiplier']),
        'name': request.form['name'],
        'source_temp': float(request.form['source_temp']),
        'source_clouds': float(request.form['source_clouds']),
        'source_pressure': float(request.form['source_pressure']),
        'source_rain': float(request.form['source_rain']),
        'source_humidity': float(request.form['source_humidity']),
        'source_wind': float(request.form['source_wind']),
        'destination_temp': float(request.form['destination_temp']),
        'destination_clouds': float(request.form['destination_clouds']),
        'destination_pressure': float(request.form['destination_pressure']),
        'destination_rain': float(request.form['destination_rain']),
        'destination_humidity': float(request.form['destination_humidity']),
        'destination_wind': float(request.form['destination_wind'])
    }
    
    # Preprocess the data using the same preprocessing steps as before
    input_data = preprocess_input_data(data)
    
    # Make the prediction using the model
    prediction = model.predict(input_data)

    # Return the predicted price
    return jsonify({"price": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
