from flask import Flask, render_template, request, jsonify
import mlflow.pyfunc
import numpy as np

app = Flask(__name__)

# Load the model from MLflow Registry
try:
    model_name = "Species_Spotting_Project"
    
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Production")
    # Mapping for Iris classes
    class_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    
    # Generate prediction
    prediction = model.predict(features)######################################################
    species = class_map.get(int(prediction[0]), "Unknown")
    
    return jsonify({'result': species})

if __name__ == '__main__':
    app.run(port=5000, debug=True)