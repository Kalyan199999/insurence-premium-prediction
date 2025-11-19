from flask import Flask, request , jsonify
import pandas as pd
import joblib
import logging
import os

# Use absolute current directory
curr_dir = os.getcwd()

# Create the logs folder if it does not exist
logs_dir = os.path.join(curr_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure logging with absolute path
logging.basicConfig(
    filename=os.path.join(logs_dir, 'app.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='w'
)

logging.debug("Starting app and loading the models")

# Load models with absolute paths
model_path = os.path.join(curr_dir, 'saved_models', 'insurance_premium_model.pkl')
preprocessor_path = os.path.join(curr_dir, 'saved_models', 'preprocessor.pkl')

try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    logging.debug("Models are loaded successfully!")
except Exception as e:
    logging.exception(f"Failed to load models: {e}")
    raise

# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def home():
    return "Insurance Premium Prediction API is running."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.debug("Received a prediction request")

        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        logging.debug(f"Received data: {data}")

        # Preprocess the input data
        input_data = pd.DataFrame([data])

        logging.debug(f"Input data:\n{input_data}")

        logging.debug(f"Transforming the data")
        preprocessed_data = preprocessor.transform(input_data)
        logging.debug(f"Transforming the data is completed!")

        logging.debug(f"Applying the model to predict the output")
        prediction = model.predict(preprocessed_data)
        logging.debug(f"prediction is completed!")

        logging.debug(f"Prediction result: {prediction}")

        predicted_value = float(prediction[0])

        logging.debug('\n --------------------------------------------------------------------------------------------------')

        return jsonify({"predicted_insurance_premium": predicted_value})
    
    except Exception as e:

        logging.exception(f"Failed to preprocess data or predict: {e}")

        logging.debug('\n --------------------------------------------------------------------------------------------------')
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':

    print( os.listdir(curr_dir) )
    # http://127.0.0.1:5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
