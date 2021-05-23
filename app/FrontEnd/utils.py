import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import requests
from io import BytesIO


@st.cache
def make_predictions(image, model_choice):
    model = {'WORD_DECTION_TRANSFERLEARNING': '__WD_Transfer_Learning',
            'SHEET_DECTION_TRANSFERLEARNING' : '__SH_Transfer_Learning'}
    file = BytesIO()
    image.save(file, "jpeg")
    file.seek(0)
    files = {"file": ("img.jpg",
                      file,
                      'multipart/form-data',
                      {'Expires': '0'})}
    res = requests.post(f"http://localhost:8000/{model[model_choice]}",
                        files=files)
    return res.json()


def draw_barplot(df, add_selectbox):
    df = df[['labels', 'scores']]
    clrs = ['#42fff2' if (x > add_selectbox)
            else "#a6d6d6" for x in df['scores']]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.set_theme(style="darkgrid")
    sns.barplot(x='labels', y='scores',
                data=df, ax=ax, palette=clrs)
    ax.set_xlabel('labels', weight='bold')
    ax.set_ylabel('scores', weight='bold')
    ax.set_xticklabels(df.labels, rotation=80)
    st.pyplot(fig)


def draw_boxes(frame, df, add_selectbox):
    df = df[df['scores'] > add_selectbox]
    labels, boxes, confidences = df['labels'].tolist(
    ), df['boxes'].tolist(), df['scores'].tolist()
    # Box format: [x, y, w, h]
    boxColor = (66, 255, 242)  # very light green
    TextColor = (255, 255, 255)  # white
    boxThickness = 3
    textThickness = 2

    for lbl, box, conf in zip(labels, boxes, confidences):
        start_coord = tuple(int(val*frame.shape[(i+1)%2])
                            for i, val in enumerate(box[:2]))
        end_coord = tuple(int(val*frame.shape[(i+1)%2])
                          for i, val in enumerate(box[2:]))
        txt = '{} ({})'.format(
            lbl, round(conf, 3))
        frame = cv2.rectangle(frame, start_coord,
                              end_coord, boxColor, boxThickness)
        frame = cv2.putText(frame, txt, start_coord,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, TextColor, 2)
    return frame
