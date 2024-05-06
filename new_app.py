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

        if user != current_user:
          current_user = user
          if not user_messages[current_user]:
            user_messages[current_user] = []

        english_words = extract_english_words(message)
        user_messages[current_user].extend(english_words)

  return user_messages

# Main function with enhanced debugging
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

      # **Error handling and model loading attempt:**
      try:
        # Load the model using a custom object dictionary (if necessary)
        model_url = "https://github.com/Karth-i/New_One/raw/9ba3e1c71a83bf70df186c342b837a9745721849/model1.h5"
        response = requests.get(model_url)
        with open("model.h5", "wb") as f:
          f.write(response.content)

        # **Attempt to load with default configuration:**
        model = tf.keras.models.load_model("model.h5")

      except ValueError as e:
        # If the default loading fails, attempt with potential custom objects
        if "recurrent_initializer" in str(e):  # Check for specific error related to recurrent initializer
          from tensorflow.keras.layers import GRU  # Import GRUCell if needed

          # **Load with custom object dictionary based on potential initializer issues:**
          custom_objects = {'GRUCell': GRU(recurrent_activation='sigmoid')}  # Example using sigmoid activation
          model = tf.keras.models.load_model("model.h5", custom_objects=custom_objects)

        else:
          # Handle other potential ValueErrors differently
          st.error(f"An error occurred while loading the model: {e}")
          # Print the full traceback and model summary for debugging
          print(traceback.format_exc())  # Print full traceback
          # st.write(model.summary())  # Uncomment to print model summary (if model is loaded successfully)
          return

      # **Additional debugging:**
      # Print model summary for inspection (if loaded successfully)
      # st.write(model.summary())  # Uncomment to print model summary

      # Fetch and preprocess messages
      messages = user_messages[selected_user]

      # Convert the messages into numerical vectors using the TextVectorization layer
      vectorize_layer = tf.keras.layers.TextVectorization(
          max_tokens=9000,
          output_mode='int
