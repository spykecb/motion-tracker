import os.path
import csv
from PIL import Image
import pandas as pd


def check_boxes(file_name,root):
    df = pd.open_csv(file_name)
    for index, row in df.iterrows():
        path = "{}/{}".format(root, row[0])
        bboxes = [row[-4],row[-3],row[-2],row[-1]]
        if os.path.isfile(path):
            image = Image.open(img_name)
            print(bboxes)


check_boxes('train/input.csv', 'train')
check_boxes('test/input.csv', 'test')


os.remove("test/input.csv")
os.remove("train/input.csv")
os.rename('train/input_new.csv', 'train/input.csv')
os.rename('test/input_new.csv', 'test/input.csv')