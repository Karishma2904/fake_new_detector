import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

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

# Load the dataset
# For this example, we'll use a sample dataset structure
# In a real scenario, you would use an actual fake news dataset
print("Loading and preparing dataset...")

# Sample data creation (replace with actual dataset loading)
# In a real scenario, you would use:
# df = pd.read_csv('path_to_your_dataset.csv')
# Creating a small sample dataset for demonstration
data = {
    'text': [
        "BREAKING: Scientists discover new planet that could support life",
        "President announces new economic policy to boost growth",
        "SHOCKING: Celebrity caught in scandal that will BLOW YOUR MIND",
        "Study shows correlation between exercise and mental health",
        "You won't BELIEVE what this politician said about taxes!",
        "New research suggests coffee may reduce risk of certain diseases",
        "URGENT: Government hiding the truth about alien contact",
        "Local community comes together to support family after house fire",
        "Secret document reveals conspiracy at highest levels of government",
        "New technology breakthrough could revolutionize renewable energy"
    ],
    'label': [
        'real',
        'real',
        'fake',
        'real',
        'fake',
        'real',
        'fake',
        'real',
        'fake',
        'real'
    ]
}

df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")

# Clean the text
df['cleaned_text'] = df['text'].apply(clean_text)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label'], test_size=0.2, random_state=42
)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set
y_pred = pac.predict(tfidf_test)

# Calculate accuracy
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {score*100:.2f}%")

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['real', 'fake'])
print("Confusion Matrix:")
print(cm)

# Save the model and vectorizer
print("Saving model and vectorizer...")
with open('model.pkl', 'wb') as model_file:
    pickle.dump(pac, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully!")