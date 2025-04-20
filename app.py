from flask import Flask, request, render_template, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Load the model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    print("Please run train_model.py first to generate the model files.")
    model = None
    vectorizer = None

# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        # Remove special characters and numbers
        text = re.sub('[^a-zA-Z]', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub('\s+', ' ', text).strip()
        return text
    return ''

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Please run train_model.py first.'})
    
    try:
        # Get the news article from the request
        news_text = request.form.get('news_text', '')
        
        if not news_text:
            return jsonify({'error': 'No text provided'})
        
        # Clean and transform the text
        cleaned_text = clean_text(news_text)
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(vectorized_text)[0]
        
        # Calculate confidence (distance to the decision boundary)
        confidence = abs(model.decision_function(vectorized_text)[0])
        confidence_percentage = min(confidence * 20, 99)  # Scale and cap at 99%
        
        return jsonify({
            'prediction': prediction,
            'confidence': f"{confidence_percentage:.2f}%",
            'text': news_text
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)