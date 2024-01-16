import streamlit as st
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



# Load the model
model = load(open('HotelReviewSentimentAnalysis.joblib', 'rb'))


def predict_sentiment(review):
    new = {'Review': review}
    features = pd.DataFrame(new, index=[0])
    result = model.predict(features)
    sentiment_result = "Positive" if result[0] == 1 else "Negative"
    return sentiment_result

# Streamlit App
def main():
    st.title("Hotel Review Sentiment Analyzer")

    # User input for review
    review = st.text_area("Enter your hotel review:", "")

    if st.button("Analyze Sentiment"):
        # Clean and analyze sentiment
        cleaned_review = clean_text(review)
        result = predict_sentiment(cleaned_review)

        # Display result
        st.subheader("Analysis Result:")
        if result == "Positive":
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😡")

if __name__ == "__main__":
    main()
