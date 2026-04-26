from flask import Flask, render_template, request, jsonify
import mlflow.pyfunc
import os

app = Flask(__name__)

# Load the PRODUCTION model from MLflow Registry
# Note: Ensure your MLflow server is running (mlflow ui)
try:
    model_name = "StudentDropoutModel"
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Successfully loaded {model_name} from Production stage.")
except Exception as e:
    print(f"Warning: Could not load production model from MLflow. Error: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    features = data.get('features')
    
    # MLflow models expect a specific format (usually 2D array or DataFrame)
    prediction = model.predict([features])
    
    return jsonify({'result': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, port=5000)