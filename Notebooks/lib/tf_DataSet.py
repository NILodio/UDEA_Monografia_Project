#import libraries

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
import tensorflow as tf


#function to separate train and test images

def build_sources_from_metadata(metadata, images_path="..\data\img", exclude_labels=None): 

    """
    Returns a list with images paths and points of Bbox

    Args:
        metadata (DataFrame): Datafreame with metadata. shape[N,2].

        images_path (str): path with location of images.

        mode (str): "train" if you want to get the train dataset.
                    "test" if you want to get the test dataset.
        
        exclude_labels (list): list with labels you want to exclude 

    Returns:

        sources (list): list with path and Bbox points [["xmin"],["xmax"],df["ymin"],["ymax"]]. shape[N,2]
    """
    
    if exclude_labels is None:
        exclude_labels = set()
    if isinstance(exclude_labels, (list, tuple)):
        exclude_labels = set(exclude_labels)

    df = metadata.copy()
    df['filepath'] = df['filename'].apply(lambda x: os.path.join(images_path, x))
    include_mask = df['class'].apply(lambda x: x not in exclude_labels)
    df = df[include_mask]

    sources = list(zip(df['filepath'], zip(df["_xmin"],df["_xmax"],df["_ymin"],df["_ymax"]),df['class']))
    return sources




#preprocessing images 

#reshape images
def preprocess_image(image,new_size=(227,227)):

    """
    Returns the image whith a different size.

    Args:
        batch (iter): iter with loaded images.

        show_box (bol)= True if you want to print the Bbox
                        False if you dont want to print the Bbox

    Returns:

        plot 3 images of the dataset
    """

    image = tf.image.resize(image, size=new_size)
    image = image / 255.0
    return image

def augment_image(image):

    return image



def make_tf_dataset(sources, training=False, batch_size=1,
    num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None):
    """
    Returns an operation to iterate over the dataset specified in sources

    Args:
        sources (list): A list of (filepath, label_id) pairs.
        training (bool): whether to apply certain processing steps
            defined only in training mode (e.g. shuffle).
        batch_size (int): number of elements the resulting tensor
            should have.
        num_epochs (int): Number of epochs to repeat the dataset.
        num_parallel_calls (int): Number of parallel calls to use in
            map operations.
        shuffle_buffer_size (int): Number of elements from this dataset
            from which the new dataset will sample.

    Returns:
        A tf.data.Dataset object. It will return a tuple images of shape
        [N, H, W, CH] and labels shape [N, 1].
    """
    def load(row):
        filepath = row['image']
        
        #read file
        img = tf.io.read_file(filepath)
        
        #tell TF this is an image jpg
        img = tf.io.decode_jpeg(img)
        
        return img, row['bbox']

    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size*4

    images, boxes , labels = zip(*sources)
    
    #create a TF dataset
    ds = tf.data.Dataset.from_tensor_slices({
        'image': list(images), 'bbox': list(boxes) , 'label': list(labels)}) 
    #line from the link. As with most code, if you remove an arbitrary line, expectin

    #shuffle dataset
    if training:
        ds = ds.shuffle(shuffle_buffer_size)
    #load images 
    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    
    #preprocces images
    ds = ds.map(lambda x,y: (preprocess_image(x), y), num_parallel_calls=num_parallel_calls)
    
    #data aumentation
    if training:
        ds = ds.map(lambda x,y: (augment_image(x), y))
        
    #repeat this order num_epochs times        
    ds = ds.repeat(count=num_epochs)
    #set size of the batch to return
    ds = ds.batch(batch_size=batch_size)
    #pre load x times of baches
    ds = ds.prefetch(1)

    return ds