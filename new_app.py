import streamlit as st
import tensorflow as tf
import numpy as np
import re
from collections import defaultdict
import requests
import io
import tempfile

# Function to extract English words from a text
def extract_english_words(text):
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
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

                if user != current_user:
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
    
    uploaded_file = st.sidebar.file_uploader("Upload WhatsApp Chat History File", type=['txt'])
    if uploaded_file:
        user_messages = process_whatsapp_file(uploaded_file)

        selected_user = st.sidebar.selectbox("Select a user", list(user_messages.keys()))

        if selected_user:
            st.subheader(f"Sentiment Analysis for {selected_user}'s messages")

            model_url = "https://github.com/Karth-i/New_One/raw/9ba3e1c71a83bf70df186c342b837a9745721849/model1.h5"
            response = requests.get(model_url)
            model_file = io.BytesIO(response.content)
            
            # Save the model to a temporary file
            temp_model_file = tempfile.NamedTemporaryFile(delete=False)
            temp_model_file.write(model_file.read())
            temp_model_file.close()
            
            # Load model from the temporary file
            model = tf.keras.models.load_model(temp_model_file.name, compile=False)

            messages = user_messages[selected_user]

            vectorize_layer = tf.keras.layers.TextVectorization(
                max_tokens=9000,
                output_mode='int',
                output_sequence_length=200
            )
            vectorize_layer.adapt([" ".join(messages)])

            sequences = vectorize_layer([" ".join(messages)])
            sequences = tf.pad(sequences, [[0, 0], [0, 200 - sequences.shape[1]]]) if sequences.shape[1] < 200 else sequences[:, :200]
            sentiment_label = np.argmax(model.predict(sequences), axis=1)

            st.write(f"Predicted sentiment label: {sentiment_label[0]}")

# Run the main function
if __name__ == '__main__':
    main()
