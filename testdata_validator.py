import os.path
import csv
import numpy as np



with open('test/input.csv', 'r') as inp, open('test/input_new.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        path = "test/{}".format(row[0])
        if os.path.isfile(path):
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
        w = x2 - x
        h = y2 - y
        widths.append(w)
        heights.append(h)
        if os.path.isfile(path) and w >= 32 and h >= 32:
            writer.writerow(row)

widths, heights = np.array(widths), np.array(heights)
print(len(widths))
print(np.count_nonzero(widths < 32))
print(np.count_nonzero(heights < 32))
print(np.count_nonzero(np.logical_or(widths < 32, heights < 32)))
print(widths.mean(), heights.mean())


os.remove("test/input.csv")
os.remove("train/input.csv")
os.rename('train/input_new.csv', 'train/input.csv')
os.rename('test/input_new.csv', 'test/input.csv')