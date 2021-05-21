import os
from random import choice
import shutil
import argparse


# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Split Data")

parser.add_argument("-o",
                    "--train_ration",
                    help="Falta",
                    default=0.7,
                    type=str)
parser.add_argument("-p",
                    "--test_ratio",
                    help="Falta",
                    default=0.25,
                    type=str)
parser.add_argument("-e",
                    "--val_ratio",
                    help="Falta",
                    default=0.05,
                    type=str)
parser.add_argument("-t",
                    "--train_path",
                    help="Falta",
                    default="data/train",
                    type=str)
parser.add_argument("-l",
                    "--test_path",
                    help="Falta",
                    default="data/test",
                    type=str)
parser.add_argument("-q",
                    "--val_path",
                    help="Falta",
                    default="data/val",
                    type=str)
parser.add_argument("-i",
                    "--img_path",
                    help="Falta",
                    default="data/img",
                    type=str)

args = parser.parse_args()


def sorting_files(path):
    imgs = []
    xmls = []
    for (dirname, dirs, files) in os.walk(path):
        for filename in files:
            if filename.endswith('.xml'):
                xmls.append(filename)
            else:
                imgs.append(filename)
    return imgs, xmls


def img_for_copy(countFor, img_path, d_path, imgs, xmls):
    # cycle for train dir
    for x in range(countFor):

        fileJpg = choice(imgs)  # get name of random image from origin dir
        # get name of corresponding annotation file
        fileXml = fileJpg[:-4] + '.xml'

        # move both files into train dir
        shutil.copy2(os.path.join(img_path, fileJpg),
                     os.path.join(d_path, fileJpg))
        shutil.copy2(os.path.join(img_path, fileXml),
                     os.path.join(d_path, fileXml))
        # remove files from arrays
        imgs.remove(fileJpg)
        xmls.remove(fileXml)

    return imgs, xmls

def makedirs_path(path):
    try:
        os.makedirs(path)
    except Exception as e:
        pass


if __name__ == '__main__':
    curr_path = os.getcwd()
    imgs, xmls = sorting_files(os.path.join(curr_path, args.img_path))

    countForTrain = int(len(imgs)*args.train_ration)
    countForTest = int(len(imgs)*args.test_ratio)
    countForval = int(len(imgs)*args.val_ratio)

    makedirs_path(os.path.join(curr_path, args.train_path))
    imgs, xmls = img_for_copy(countForTrain, os.path.join(
        curr_path, args.img_path), os.path.join(curr_path, args.train_path), imgs, xmls)

    makedirs_path(os.path.join(curr_path, args.test_path))
    imgs, xmls = img_for_copy(countForTest, os.path.join(
    curr_path, args.img_path), os.path.join(curr_path, args.test_path), imgs, xmls)

    makedirs_path(os.path.join(curr_path, args.val_path))
    imgs, xmls = img_for_copy(countForval, os.path.join(
    curr_path, args.img_path), os.path.join(curr_path, args.val_path), imgs, xmls)

    # print(imgs , xmls)

