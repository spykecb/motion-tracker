import os.path
import csv

with open('test/input.csv', 'r') as inp, open('test/input_new.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        path = "test/{}".format(row[0])
        if os.path.isfile(path):
            writer.writerow(row)

with open('train/input.csv', 'r') as inp, open('train/input_new.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        path = "train/{}".format(row[0])
        if os.path.isfile(path):
            writer.writerow(row)



os.remove("test/input.csv")
os.remove("train/input.csv")
os.rename('train/input_new.csv', 'train/input.csv')
os.rename('test/input_new.csv', 'test/input.csv')