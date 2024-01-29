from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer  # Use TfidfVectorizer instead of TfidfTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Download NLTK resources (run this once)
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Load the trained model and vectorizer
loaded_clf = load('dark_pattern_classifier.joblib')
loaded_vectorizer = load('dark_pattern_vectorizer.joblib')  # Assuming this is the TfidfVectorizer

# Define the preprocess_text function using NLTK
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Apply stemming (optional)
    words = [ps.stem(word) for word in words]

    # Join the words back into a single string
    preprocessed_text = ' '.join(words)

    return preprocessed_text

# Preprocess function for cookie information
def preprocess_cookies(cookies):
    # Combine cookie names and values into a single string
    cookie_text = ' '.join([f'{cookie.name}={cookie.value}' for cookie in cookies])
    # Apply the same preprocessing function used for training
    preprocessed_cookie_text = preprocess_text(cookie_text)
    return preprocessed_cookie_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_cookies', methods=['POST'])
def check_cookies():
    data = request.get_json()
    url = data['url']

    # Make a GET request to the URL
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify(result=f'Error: Unable to fetch website content. {e}')

    # Parse HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract and preprocess the cookies
    cookies = response.cookies
    preprocessed_cookies = preprocess_cookies(cookies)

    # Vectorize and transform the cookie data using the loaded vectorizer
    cookie_tfidf = loaded_vectorizer.transform([preprocessed_cookies])

    # Make a prediction
    prediction = loaded_clf.predict(cookie_tfidf)

    # Check if the predicted label is 1
    if prediction[0] == 1:
        return jsonify(result='Cookies found')
    else:
        return jsonify(result='Cookies not found')

if __name__ == '__main__':
    app.run(app.run(host='0.0.0.0', port=8080),debug=True)
