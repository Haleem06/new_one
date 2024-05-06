import streamlit as st
import tensorflow as tf
import numpy as np
import re
from collections import defaultdict
import requests

# Function to extract English words from a text
def extract_english_words(text):
    english_words = []
    english_pattern = re.compile(r'\b[a-zA-Z]+\b')
    english_words = english_pattern.findall(text)
    return english_words

# Function to process the WhatsApp chat history file
def process_whatsapp_file(uploaded_file):
    user_messages = defaultdict(list)
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.splitlines()

    current_user = None
    for line in lines:
        parts = line.split(' - ')
        if len(parts) > 1:
            user_message = parts[1].strip()
            user_parts = user_message.split(': ')
            if len(user_parts) > 1:
                user = user_parts[0].strip()
                message = ': '.join(user_parts[1:]).strip()

                if user!= current_user:
                    current_user = user
                    if not user_messages[current_user]:
                        user_messages[current_user] = []

                english_words = extract_english_words(message)
                user_messages[current_user].extend(english_words)

    return user_messages

# Function to load the sentiment analysis model
def load_model():
    try:
        model = tf.keras.models.load_model('model.h5', custom_objects={'GlorotUniform': tf.keras.initializers.GlorotUniform()})
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to perform sentiment analysis
def perform_sentiment_analysis(model, messages):
    try:
        vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=9000,
            output_mode='int',
            output_sequence_length=200
        )
        vectorize_layer.adapt([" ".join(messages)])
        sequences = vectorize_layer([" ".join(messages)])
        if sequences.shape[1] < 200:
            sequences = tf.pad(sequences, [[0, 0], [0, 200 - sequences.shape[1]]])
        elif sequences.shape[1] > 200:
            sequences = sequences[:, :200]
        sentiment_label = np.argmax(model.predict(sequences), axis=1)
        return sentiment_label
    except Exception as e:
        st.error(f"Error performing sentiment analysis: {e}")
        return None

# Main function
def main():
    st.title('WhatsApp Chat Sentiment Analyzer')
    st.sidebar.title('Options')

    uploaded_file = st.sidebar.file_uploader("Upload WhatsApp Chat History File", type=['txt'])
    if uploaded_file:
        user_messages = process_whatsapp_file(uploaded_file)
        selected_user = st.sidebar.selectbox("Select a user", list(user_messages.keys()))

        if selected_user:
            st.subheader(f"Sentiment Analysis for {selected_user}'s messages")
            model = load_model()
            if model:
                messages = user_messages[selected_user]
                sentiment_label = perform_sentiment_analysis(model, messages)
                if sentiment_label:
                    st.write(f"Predicted sentiment label: {sent
