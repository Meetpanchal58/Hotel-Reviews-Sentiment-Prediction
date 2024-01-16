from flask import Flask, render_template, request
import pandas as pd
from pickle import load
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
wnl = WordNetLemmatizer()


# Creating the flask app
app = Flask(__name__)


def clean_text(text):
    # Remove numeric and alpha-numeric characters
    text = re.sub(r'\w*\d\w*', '', str(text)).strip()
    
    # Replace common contractions
    text = re.sub(r"won't", "will not", str(text))
    text = re.sub(r"can't", "can not", str(text))
    text = re.sub(r"ca n\'t", "can not", str(text))      
    text = re.sub(r"wo n\'t", "will not", str(text))     
    text = re.sub(r"\'t've", " not have", str(text))    
    text = re.sub(r"\'d've", " would have", str(text))
    text = re.sub(r"n\'t", " not", str(text))    
    text = re.sub(r"\'re", " are", str(text))     
    text = re.sub(r"\'s", " is", str(text))       
    text = re.sub(r"\'d", " would", str(text))   
    text = re.sub(r"\'ll", " will", str(text))   
    text = re.sub(r"\'t", " not", str(text))     
    text = re.sub(r"\'ve", " have", str(text))    
    text = re.sub(r"\'m", " am", str(text))      
    
    # Remove punctuation and symbols
    text = re.sub('[%s]' % re.escape(string.punctuation), '', str(text))
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and apply lemmatization
    tokens = [wnl.lemmatize(word, 'v') for word in tokens if word.lower().strip() not in stop_words]
    
    # Remove duplicate words
    tokens = list(dict.fromkeys(tokens))
    
    # Join tokens into a cleaned text
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def clean_text_frame(X):
    b = clean_text
    return X.applymap(b)


# Loading the model
model = load(open('HotelReviewSentimentAnalysis.joblib', 'rb'))

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Page
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        new = {'Review': review}
        features = pd.DataFrame(new, index=[0])

        # Model Prediction
        result = model.predict(features)

        # Convert result to string
        sentiment_result = "Positive" if result[0] == 1 else "Negative"

        # Rendering the result on HTML page
        return render_template('index.html', result=sentiment_result)

if __name__ == '__main__':
    app.run(debug=True)

