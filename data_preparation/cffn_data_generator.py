from glob import glob
import os
import json
import sys

labels_file = "/home/ubuntu/prep_data/labels.json"
train_file = "/home/ubuntu/prep_data/cffn_classifier_train.txt"
val_file = "/home/ubuntu/prep_data/cffn_classifier_val.txt"

metadata = {}
with open(labels_file) as f:
    metadata = json.load(f)

num_train = int(len(metadata) * 0.9)
#print(len(metadata))
#print(num_train)
#sys.exit(0)

data = list(metadata.items())

# first do the training data
f = open(train_file, "w")
for item, label in data[:num_train]:
    lbl = 1 if label == 'REAL' else 0
    f.write(item + " " + str(lbl) + "\n")
f.close()

# now validation data
f = open(val_file, "w")
for item, label in data[num_train:]:
    lbl = 1 if label == 'REAL' else 0
    f.write(item + " " + str(lbl) + "\n")
f.close()
