# 📰 Fake News Detector

A simple and interactive web application that uses machine learning to classify news articles as **real** or **fake**.

Built with **Flask**, **scikit-learn**, and **NLTK**, this app features a clean Bootstrap-powered UI for real-time news analysis.

## 🔍 Features

- Text preprocessing and cleaning using NLTK
- TF-IDF vectorization for feature extraction
- Fake news detection using a Passive Aggressive Classifier
- Web interface with confidence score display
- RESTful prediction API using Flask

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector

pip install -r requirements.txt

python train_model.py

python app.py
.
├── app.py               # Flask web server and API
├── train_model.py       # Script to train the ML model
├── requirements.txt     # Python dependencies
├── model.pkl            # Trained classifier (generated after training)
├── vectorizer.pkl       # TF-IDF vectorizer (generated after training)
└── templates/
    └── index.html       # Frontend UI (Bootstrap)

