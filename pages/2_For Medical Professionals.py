import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
import subprocess
import sys
from PIL import Image
import cv2

st.set_page_config(
    page_title="For Medical Professionals",
    page_icon=":health_worker:",
    layout="centered",
    initial_sidebar_state="expanded")

st.markdown(f'<h1 style="color:#000000;font-size:40px;">{"Powered by Patient Data"}</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <style>
        hr {
            border: 2px solid #7E587E;  /* Dark purple nude color */
            margin: 15px 0;  /* Adjust margin as needed */
        }
    </style>
    """,
    unsafe_allow_html=True)
#st.markdown("---") # Separator line

content_2 = """
    <div style="text-align: left; color:#7E587E;font-family: 'Arial', sans-serif; font-size: 30px;">
        <b>I'm Matthew!<b>

        
    <div style="text-align: left; color:#000000;font-family: 'Arial', sans-serif; font-size: 26px;">
        <b>And I'm here to flag out suspicious exams and create a priority track system <b>  
    </div>
"""
st.write(content_2, unsafe_allow_html=True)

#### For the Model in the CLOUD ####

model_path = "C:/Armazenamento/yolov5-master/runs/train/exp36/weights/last.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Running model on an image
def run_model(image_np):
    results = model(image_np)
    return results

# File buffer to cv2 image
def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

# File uploader for a single image - Just for the demo
img_file = st.file_uploader(label="Load Unseen_data.png",
                             type=['png', 'jpg', 'jpeg'])

if img_file:
    open_cv_image = create_opencv_image_from_stringio(img_file)

    # Display
    st.image(open_cv_image, caption='Uploaded Image', use_column_width=True)

    if st.button("Run Model"):
        try:
            results = run_model(open_cv_image)

            # Display the results
            st.success("Model has been run successfully!")
            st.subheader("Detection Results:")
            st.image(np.squeeze(results.render()), use_column_width=True)
            st.image("recall.png")  
        except Exception as e:
            st.write(f"An error occurred: {e}")
########
    
st.markdown("---") # Separator line
st.markdown(f'<h1 style="color:#7E587E;font-size:36px;">{"We have come this far, but we want more!"}</h1>', unsafe_allow_html=True)

st.markdown(f'<h1 style="color:#000000;font-size:32px;">{"More computacional power, but also:"}</h1>', unsafe_allow_html=True)
st.write("""
    <div style="text-align: left; font-family: 'Arial', sans-serif; font-size: 24px;">
    &#8594; fine-tuning your model
    <div style="text-align: left; font-family: 'Arial', sans-serif; font-size: 24px;">
    &#8594; ≥ 1500 images per class recommended
    <div style="text-align: left; font-family: 'Arial', sans-serif; font-size: 24px;">
    &#8594; background images to reduce false positives
    </div>
""", unsafe_allow_html=True)

st.markdown(f'<h1 style="color:#000000;font-size:32px;">{"Optimize diagnosis"}</h1>', unsafe_allow_html=True)
st.write("""
    <div style="text-align: left; font-family: 'Arial', sans-serif; font-size: 24px;">
    &#8594; tools like Roboflow can be used during image analysis
    <div style="text-align: left; font-family: 'Arial', sans-serif; font-size: 24px;">   
    &#8594; train the model to across various thoracic diseases
    <div style="text-align: left; font-family: 'Arial', sans-serif; font-size: 24px;">
    &#8594; improve image quality/contrast<p>  
    </div>
""", unsafe_allow_html=True)


st.markdown(f'<h1 style="color:#000000;font-size:32px;">{"Sustainable Deep Learning Model"}</h1>', unsafe_allow_html=True)
st.write("""
    <div style="text-align: left; font-family: 'Arial', sans-serif; font-size: 24px;">
    &#8594; cost-effectiveness: all clinical practices of the acquisition 
    and classification are inherent to healthcare professional's workflow - <b>we'll simply enhance it!<b>    
    </div>
""", unsafe_allow_html=True)

