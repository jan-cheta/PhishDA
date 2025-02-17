from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import joblib

# Load pre-trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorize.pkl')

app = Flask(__name__)
CORS(app)

@app.route('/')
def open_chu():
    return render_template('PhishDA.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data
        data = request.get_json()
        if not data or 'body' not in data:
            return jsonify({"error": "The 'body' field is required"}), 400
        
        # Get email body
        body = data['body']
        
        # Transform using the vectorizer
        email = vectorizer.transform([body])
        
        # Predict using the model
        prediction = model.predict(email)[0]
        
        # Return response as JSON
        return jsonify({'body': int(prediction)})
    
    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
