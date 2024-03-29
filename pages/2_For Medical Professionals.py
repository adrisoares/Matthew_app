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

#Yolo
def run_yolov5(image_np, conf_thres=0.25, weights='yolov5s.pt'):
    model = torch.hub.load('ultralytics/yolov5:v5.0', 'yolov5s', path=weights, pretrained=False, force_reload=True)
    model.eval()
    
    # Run my model
    results = model(image_np, conf_thres=conf_thres)
    im0 = results.render()[0]
    return im0

# Sample image path (modify this to your sample image, just for the demo)
sample_image_path = "Unseen_data.png"
sample_image = Image.open(sample_image_path)
sample_image_np = np.array(sample_image)

# The image used for demo
st.image(sample_image, caption='Sample Image', use_column_width=True)

# Button - my favorite stuff in st
if st.button("Run Model on Sample Image"):
    im0 = run_yolov5(sample_image_np, conf_thres=0.25, weights='yolov5s.pt')
    
    if im0 is not None:
        st.image(im0, channels="BGR", caption="Detection Results")

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

