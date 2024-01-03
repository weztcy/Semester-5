# Filename: streamlit_app.py

import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk

# Load necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to preprocess text
def clean_and_lemmatize(text):
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    cleaned_text = re.sub('[^a-zA-Z]', ' ', text).lower()
    tokens = cleaned_text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in set(all_stopwords)]
    return " ".join(lemmatized_tokens)

# Load the model
with open("sentiment_analysis_model.pkl", "rb") as model_file:
    tfidf, classifier = pickle.load(model_file)

# Streamlit app
st.title('Sentiment Analysis App')

message = st.text_area('Enter your review:')
if st.button('Predict'):
    cleaned_message = clean_and_lemmatize(message)
    vect = tfidf.transform([cleaned_message]).toarray()
    prediction = classifier.predict(vect)
    st.write('Prediction:', prediction[0])
