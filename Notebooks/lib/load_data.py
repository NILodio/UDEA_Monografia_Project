

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
import glob

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def create_and_read_metadata(images_path , medata_path,name_file, split_value , labels_dic):

    """
    Returns Dataframe with all data

    Args:
        initial_path (str): path of the xml and img you want to read.

        split_value (int): value between 0 and 1 .

        labels_dic (dic): Dictionary with the labels.
        
    Returns:
        metadata (df): Dataframe with all data .

    """

    if os.path.isfile(os.path.join(medata_path, name_file ))==False:

        w=[]
        for file in os.listdir(images_path):
            if file.endswith(".xml"):
                w.append(read_content(os.path.join(images_path,file)))


        names,splits,labl,xmin,xmax,ymin,ymax=[],[],[],[],[],[],[]

        for i in w:

            #split images in train and test
            split = np.random.rand(1)[0]
            if split< split_value:
                spl='train'
            else:
                spl='test'

            #create metadata columns
            splits.append(spl)
            labl.append(i[3][0])
            ymin.append(i[2][0][1]/i[4][0][1])
            ymax.append(i[2][0][3]/i[4][0][1])
            xmin.append(i[2][0][0]/i[4][0][0])
            xmax.append(i[2][0][2]/i[4][0][0])
            names.append(i[1])

        #generate dictionary to build csv
        df={}
        df["names"]=names
        df["xmin"]=xmin
        df["xmax"]=xmax
        df["ymin"]=ymin
        df["ymax"]=ymax
        df['split']=splits
        df['label']=labl
        df = pd.DataFrame(df)
        df=df.replace(labels_dic)

        #create csv with metadata information
        df.to_csv(os.path.join(medata_path,name_file), index=False)
        metadata=df
    else:
        metadata = pd.read_csv(os.path.join(medata_path,name_file))

    return metadata





def read_content(xml_file: str):
    """
    Returns important information from a xml with a Bbox

    Args:
        xml_file (str): path of the xml you want to read.
    Returns:
        file_path (str): path of the image that corresponds to this xml.

        file_name (str): name of the image that corresponds to this xml.

        list_with_all_boxes (list): list whith 4 points of the Bbox. shape[N,4]

        list_with_all_names (list): list with labels of each Bbox. shape[N,1]
        
        list_with_image_dimentions: list with the shape of the image shape[1,1]
    """
    
    
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_with_all_names = []
    list_with_image_dimentions=[]
    
    #get inmportant information from XML
    for boxes in root.iter("object"):

        file_name = root.find("filename").text
        file_path = root.find("path").text
        name = boxes.find("name").text
        width = int(root.find("size").find("width").text)
        height= int(root.find("size").find("height").text)

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)


        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
        list_with_all_names.append(name)
        list_with_image_dimentions.append([width,height])
        

    return file_path,file_name, list_with_all_boxes,list_with_all_names,list_with_image_dimentions

