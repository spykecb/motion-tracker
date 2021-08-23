import os.path
import csv
import numpy as np
from PIL import Image

input_to = 3
positions_to = input_to + 22 * 2
confidences_to = positions_to + 22
boundaries_to = confidences_to + 4
count_test = 0
count_test_kept = 0
count_train = 0
count_train_kept = 0

crop_path_train = 'train/crops'
if not os.path.exists(crop_path_train):
    os.makedirs(crop_path_train)
crop_path_test = 'test/crops'
if not os.path.exists(crop_path_test):
    os.makedirs(crop_path_test)


with open('test/input.csv', 'r') as inp, open('test/input_new.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        count_test += 1
        path = "test/{}".format(row[0])
        zeros = np.count_nonzero(np.array(row[positions_to:confidences_to]) == '0')
        ones = np.count_nonzero(np.array(row[positions_to:confidences_to]) == '1')
        # if os.path.isfile(path) and ones > zeros:
        if os.path.isfile(path) and zeros == 0:
            count_test_kept += 1
            writer.writerow(row)
widths, heights = [],[]
with open('train/input.csv', 'r') as inp, open('train/input_new.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        count_train += 1
        path = "train/{}".format(row[0])
        crop_path = "{}/{}".format(crop_path_train, row[0])
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
        # if os.path.isfile(path) and ones > zeros:
        if os.path.isfile(path) and zeros == 0:
        # if os.path.isfile(path) and w >= 64 and h >= 64:
            
            # image = Image.open(path)
            # rect = tuple(int(b) for b in bbox_output)
            # image = image.crop(rect)
            # image.save(crop_path, "JPEG")
            count_train_kept += 1
            writer.writerow(row)

widths = np.array(widths)
heights = np.array(heights)

print("Avg width/height:", widths.mean(), heights.mean())
print("Min width/height:", widths.min(), heights.min())
print("Train kept/total:", count_train_kept, count_train)
print("Test kept/total:", count_test_kept, count_test)

# os.remove("test/input.csv")
# os.remove("train/input.csv")
# os.rename('train/input_new.csv', 'train/input.csv')
# os.rename('test/input_new.csv', 'test/input.csv')