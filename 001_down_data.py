#!/usr/bin/env python3

import requests
import os
import shutil
import argparse
import zipfile
import sys

"""
Script to download dataset

from Google Drive using Python3.8

http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/


"""

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Dowload DataSet")
parser.add_argument("-x",
                    "--name_data",
                    help="Name input img and xml will be stored",
                    type=str)
parser.add_argument("-l",
                    "--name_path",
                    help="Path to the images (.jpg)",
                    default= "data", 
                    type=str)

args = parser.parse_args()


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def makedirs_path(path):
    try:
        os.makedirs(path)
    except Exception as e:
        pass
        
def dowload_selected(path:str,type_data: str):
    print(path)
    if os.path.exists(os.path.join(path,"img.zip")) == False:
        print("downloading......")
        download_file_from_google_drive(type_data, os.path.join(path,"img.zip"))
    

def unzip_files(path:str):
    # print(path)
    if os.path.exists(path) == True:
        # print("Hola")
        with zipfile.ZipFile(os.path.join(path,"img.zip"),"r") as zip_ref:
            zip_ref.extractall(path)


if __name__ == '__main__':
    curr_path = os.getcwd()
    models_path = os.path.join(curr_path,args.name_path)
    makedirs_path(models_path)

    if args.name_data =="Word_Detection":
        dowload_selected(models_path ,"1uSfcUe3gA5Img_cxh1-Y-WXl9leZPNY6")
        print(f"files downloaded {args.name_data}")
        unzip_files(models_path)
        print(f"Unzip : {args.name_data}")
    elif args.name_data =="Sheet_detection":
        print(f"Data : {args.name_data}")
        dowload_selected(models_path ,"1zABTiY0YlXwtKyx2t2lqfgAKb3tARRWj")
        print(f"files downloaded {args.name_data}")
        unzip_files(models_path)
        print(f"Unzip : {args.name_data}")
    else:
        print("Error")

    
    



# if os.path.exists(os.path.join(models_path,"WIDER_val")) == False:
#     with zipfile.ZipFile(os.path.join(models_path,"val.zip"),"r") as zip_ref:
#         zip_ref.extractall(models_path)

# print("files unziped")

# # downloading from: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# url = 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz'

# if os.path.exists(os.path.join(models_path,"ssd_mobilenet_v1_coco_11_06_2017.tar.gz")) == False:
#     response = requests.get(url, stream=True)
#     with open(os.path.join(models_path,"ssd_mobilenet_v1_coco_11_06_2017.tar.gz"), 'wb') as out_file:
#         shutil.copyfileobj(response.raw, out_file)
#     del response


# import tarfile
# filePath = os.path.join(models_path,"ssd_mobilenet_v1_coco_11_06_2017.tar.gz")
# os.chdir(models_path)


# if (filePath.endswith("tar.gz")):
#     tar = tarfile.open(filePath, "r:gz")
#     tar.extractall()
#     tar.close()
# elif (filePath.endswith("tar")):
#     tar = tarfile.open(filePath, "r:")
#     tar.extractall()
#     tar.close()


# print("done")
