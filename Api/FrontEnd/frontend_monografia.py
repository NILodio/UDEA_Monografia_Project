import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# from classes import Context, ModelTfod
import requests
import json


st.title("Word location")


img_file_buffer2 = st.file_uploader("Choose another image...", type="jpg")

operation = st.selectbox('Which model do you want to use?',('detect all','Detect word', 'Detect sheet'))


if img_file_buffer2 is not None:

    files = {"file": img_file_buffer2.getvalue()}
    print(operation)
    response = requests.post(f"http://localhost:8000/predict_image/{operation}", files=files)
    image_list = json.loads(response.json()["image"])
    image_array = np.array(image_list)

    st.image(img_file_buffer2)
    st.image(image_array)



       