import streamlit as st
import tensorflow as tf
import numpy as np
import re
from collections import defaultdict
import requests
from pathlib import Path

# Function to extract English words from a text
def extract_english_words(text: str) -> list:
    """Extract English words from a text"""
    english_pattern = re.compile(r'\b[a-zA-Z]+\b')
    return english_pattern.findall(text)

# Function to process the WhatsApp chat history file
def process_whatsapp_file(uploaded_file: bytes) -> dict:
    """Process the WhatsApp chat history file"""
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

# Main function
def main():
    st.title('WhatsApp Chat Sentiment Analyzer')
    st.sidebar.title('Options')

    # Upload WhatsApp chat history file
    uploaded_file = st.sidebar.file_uploader("Upload WhatsApp Chat History File", type=['txt'])
    if uploaded_file:
        # Process the WhatsApp chat history file
        user_messages = process_whatsapp_file(uploaded_file)

        # Select a user from the chat history
        selected_user = st.sidebar.selectbox("Select a user", list(user_messages.keys()))

        if selected_user:
            st.subheader(f"Sentiment Analysis for {selected_user}'s messages")

            # Load model
            model_url = "https://github.com/Karth-i/New_One/raw/9ba3e1c71a83bf70df186c342b837a9745721849/model1.h5"
            response = requests.get(model_url)
            model_path = Path("model.h5")
            with model_path.open("wb") as f:
                f.write(response.content)
            model = tf.keras.models.load_model(str(model_path))

            # Fetch and preprocess messages
            messages = user_messages[selected_user]

            # Convert the messages into numerical vectors using the TextVectorization layer
            vectorize_layer = tf.keras.layers.TextVectorization(
                max_tokens=9000,
                output_mode='int',
                output_sequence_length=200
            )
            vectorize_layer.adapt([" ".join(messages)])

            # Predict sentiment
            sequences = vectorize_layer([" ".join(messages)])
            if sequences.shape[1] < 200:
                sequences = tf.pad(sequences, [[0, 0], [0, 200 - sequences.shape[1]]])
            elif sequences.shape[1] > 200:
                sequences = sequences[:, :200]
            sentiment_label = np.argmax(model.predict(sequences), axis=1)

            st.write(f"Predicted sentiment label: {sentiment_label[0]}")

if __name__ == '__main__':
    main()
