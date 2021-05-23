import os
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
from pylab import rcParams
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import albumentations as A
import random


def augmented_fuction(img_Aug: int, Data , doc_aug , DATASET_PATH: str, IMAGES_PATH_SAVE: str, IMAGES_PATH: str , NAME_FILE: str):


    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(IMAGES_PATH_SAVE, exist_ok=True)

    rows = []
    id_ = 0

    for r in Data.iterrows():
        id_ += 1
        form = cv2.imread(IMAGES_PATH + r[1][0])
        len_y, len_x, _ = form.shape
        STUDENT_ID_BBOX = [r[1][1]*len_x, r[1][3]* len_x, r[1][2]*len_x, r[1][4]*len_x]
        format_type = r[1][5]
        for i in tqdm(range(img_Aug)):
            augmented = doc_aug(image=form, bboxes=[
                                STUDENT_ID_BBOX], field_id=['1'])
            file_name = f'{id_}_form_aug_{i}.jpg'
            for bbox in augmented['bboxes']:
                x_min, y_min, x_max, y_max = map(lambda v: int(v), bbox)
                rows.append({
                    'filename': file_name,
                    'xmin': x_min,
                    'xmax': x_max,
                    'ymin': y_min,
                    'ymax': y_max,
                    'class': format_type
                })

            cv2.imwrite(f'{IMAGES_PATH_SAVE}/{file_name}', augmented['image'])
    
    Data_Aumented = pd.DataFrame(rows)
    Data_Aumented.to_csv(f'{DATASET_PATH}/{NAME_FILE}', header=True, index=None)

    return Data_Aumented                                               
