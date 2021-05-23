from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
from utils import make_predictions, draw_barplot, draw_boxes

st.title("UDEA Project")

st.markdown('## Select task prediction')
model = st.radio("Pick the model you want to generate the predictions",
                 ('WORD DECTION', 'SHEET DECTION'))

with st.beta_expander('Plot image with annotations.',):
    img_file_buffer2 = st.file_uploader("Choose another image...", type="jpg")

    if img_file_buffer2 is not None:
        # print(type(img_file_buffer2.getvalue()))
        image2 = Image.open(img_file_buffer2)
        img_array2 = np.array(image2)

        # predictions = make_predictions(img_file_buffer2, 'yolo')
        predictions = make_predictions(image2, model)
        print(predictions)

        df = pd.DataFrame(predictions)
        df['labels'] = df['labels'] + df.index.values.astype(str)
        
        df.sort_values(by=['scores'], ascending=False, inplace=True)
        df = df[df['scores'] > 0.1]
    
        if df.shape[0] > 0:
            add_selectbox = st.slider(
                'Select confidence', min_value=0, max_value=100, value=50, step=5) / 100.0
            
            annotated = draw_boxes(img_array2, df, add_selectbox)
            
            st.image(annotated)
            
            st.markdown(
                """
                
                --------------------------------------------------------------------------------------------------------------""")
            st.subheader('Labels score')
            draw_barplot(df, add_selectbox)
        else:
            st.write("The model wasn't able to detect anything in your image :(")
