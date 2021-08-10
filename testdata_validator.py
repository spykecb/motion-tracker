import os.path
import csv
import numpy as np
from PIL import Image

input_to = 3
positions_to = input_to + 22 * 2
confidences_to = positions_to + 22
boundaries_to = confidences_to + 4

with open('test/input.csv', 'r') as inp, open('test/input_new.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        path = "test/{}".format(row[0])
        zeros = np.count_nonzero(np.array(row[positions_to:confidences_to]) == '0')
        ones = np.count_nonzero(np.array(row[positions_to:confidences_to]) == '1')
        if os.path.isfile(path)  and ones > zeros:
            writer.writerow(row)
widths, heights = [],[]
with open('train/input.csv', 'r') as inp, open('train/input_new.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        path = "train/{}".format(row[0])
        x = float(row[-4])
        y = float(row[-3])
        x2 = float(row[-2])
        y2 = float(row[-1])
        zeros = np.count_nonzero(np.array(row[positions_to:confidences_to]) == '0')
        ones = np.count_nonzero(np.array(row[positions_to:confidences_to]) == '1')
        w = x2 - x
        h = y2 - y
        widths.append(w)
        heights.append(h)
        bbox_output = [x,y,x2,y2]
        if os.path.isfile(path) and ones > zeros:
        # if os.path.isfile(path) and w >= 64 and h >= 64:
            # image = Image.open(path)
            # rect = tuple(int(b) for b in bbox_output)
            # image = image.crop(rect)
            # image.save(path.replace(".jpg", "_cropped.jpg"), "JPEG")
            writer.writerow(row)

widths, heights = np.array(widths), np.array(heights)
print(len(widths))
print(np.count_nonzero(widths < 64))
print(np.count_nonzero(heights < 64))
print(np.count_nonzero(np.logical_or(widths < 64, heights < 64)))
print(widths.mean(), heights.mean())



os.remove("test/input.csv")
os.remove("train/input.csv")
os.rename('train/input_new.csv', 'train/input.csv')
os.rename('test/input_new.csv', 'test/input.csv')