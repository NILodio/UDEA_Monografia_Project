
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
from . import IoU as iou
import numpy as np
import pandas as pd
import cv2

def display_random_image (images, label, show_box = True ,image_path = '../data/img'):
    """
        Display a random image
    """
    
    images = images[images['class'] == label]
    index = np.random.randint(images.shape[0])
    img = Image.open(os.path.join(image_path,images.iloc[index][0]))

    if show_box == True:
        img_n = np.array(img)
        len_y , len_x , _ = img_n.shape

        xmin = images.iloc[index][1] * len_x
        xmax = images.iloc[index][2] * len_x
        ymin = images.iloc[index][3] * len_y
        ymax = images.iloc[index][4] * len_y

        plt.figure(figsize=(10,5))
        plt.imshow(img)
        plt.title('Image #{} : Numero : {} '.format(index,images.iloc[index][5]))
        plt.plot((xmin,xmax),(ymin,ymin),"r")
        plt.plot((xmin,xmax),(ymax,ymax),"r")
        plt.plot((xmin,xmin),(ymax,ymin),"r")
        plt.plot((xmax,xmax),(ymax,ymin),"r")
        plt.show()

        
    else:
        plt.figure(figsize=(20,10))
        plt.imshow(img)
        plt.grid(False)
        plt.title('Image # {} : Nombre : {} '.format(index,images.iloc[index][0]))
        plt.show()


def display_random_image_Pre (Images,Images_Pre, show_box = True,image_path = '../data/img'):
    """
        Display a random image
    """
    
    index_img = np.random.randint(Images.shape[0])

    img = Image.open(os.path.join(image_path,Images.iloc[index_img][0]))


    if show_box == True:
        img_n = np.array(img)
        len_y , len_x , _ = img_n.shape

        xmin = Images.iloc[index_img][1] * len_x
        xmax = Images.iloc[index_img][2] * len_x
        ymin = Images.iloc[index_img][3] * len_y
        ymax = Images.iloc[index_img][4] * len_y

        xmin_p = Images_Pre[index_img][0] * len_x
        xmax_p = Images_Pre[index_img][1] * len_x
        ymin_p = Images_Pre[index_img][2] * len_y
        ymax_p = Images_Pre[index_img][3] * len_y


        plt.figure(figsize=(15,8))
        plt.imshow(img)
        plt.title('Image #{} : Label : {} '.format(index_img,Images.iloc[index_img][5]))
        plt.plot((xmin,xmax),(ymin,ymin),"r")
        plt.plot((xmin,xmax),(ymax,ymax),"r")
        plt.plot((xmin,xmin),(ymax,ymin),"r")
        plt.plot((xmax,xmax),(ymax,ymin),"r")


        Iou_values = iou.bb_intersection_over_union(Images.iloc[index_img].values[1:5],Images_Pre[index_img])


        plt.text(len_x, len_y, 'IoU: {:.4f}'.format(Iou_values[0]),
                verticalalignment='bottom', horizontalalignment='right',
                color='r', fontsize=15,bbox=dict(fill=True, edgecolor='red', linewidth=2))

                

        plt.plot((xmin_p,xmax_p),(ymin_p,ymin_p),"g")
        plt.plot((xmin_p,xmax_p),(ymax_p,ymax_p),"g")
        plt.plot((xmin_p,xmin_p),(ymax_p,ymin_p),"g")
        plt.plot((xmax_p,xmax_p),(ymax_p,ymin_p),"g")
        
        plt.show()

        
    else:
        print('Implementar')

#function to show a batch of three images

def imshow_batch_of_three(batch, show_box=True, num_images = 2):
    
    """
    Returns plot of 3 images with bbox.

    Args:
        batch (iter): iter with loaded images.

        show_box (bol)= True if you want to print the Bbox
                        False if you dont want to print the Bbox

    Returns:

        plot 3 images of the dataset
    """

    boxes_batch = batch[1].numpy()
    image_batch = batch[0].numpy()
    _ , axarr = plt.subplots(1, num_images, figsize=(15, 5), sharey=True)
    _,len_y,len_x,_=image_batch.shape

    for i in range(num_images):
        img = image_batch[i, ...]
        axarr[i].imshow(img)
        if show_box:
#             axarr[i].set(xlabel='cordenates = {}'.format(boxes_batch[i]))
            print(boxes_batch[i])
            print(boxes_batch[i][0]*len_x)
            print(boxes_batch[i][1]*len_x)
            axarr[i].plot((int(boxes_batch[i][0]*len_x),int(boxes_batch[i][1]*len_x)),(int(boxes_batch[i][2]*len_y),int(boxes_batch[i][2]*len_y)),"r")
            axarr[i].plot((int(boxes_batch[i][0]*len_x),int(boxes_batch[i][1]*len_x)),(int(boxes_batch[i][3]*len_y),int(boxes_batch[i][3]*len_y)),"r")
            axarr[i].plot((int(boxes_batch[i][0]*len_x),int(boxes_batch[i][0]*len_x)),(int(boxes_batch[i][2]*len_y),int(boxes_batch[i][3]*len_y)),"r")
            axarr[i].plot((int(boxes_batch[i][1]*len_x),int(boxes_batch[i][1]*len_x)),(int(boxes_batch[i][2]*len_y),int(boxes_batch[i][3]*len_y)),"r")


def Display_Examples(Examples):
    """
        Display a random image
    """

 