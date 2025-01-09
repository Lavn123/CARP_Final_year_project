import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image 
import numpy as np
from PIL import Image

st.title("Image Classification: Dog, Cat, or Human")

try:
    model = keras.models.load_model("my_model.h5")
    print("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading the model: {e}. Please train the model first (run train_model.py).")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert('RGB') 
    st.image(pil_image, caption='Uploaded Image.', use_column_width=True)

    img = pil_image.resize((150, 150))
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    class_names = ['Dog', 'Cat', 'Human']
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.write(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")