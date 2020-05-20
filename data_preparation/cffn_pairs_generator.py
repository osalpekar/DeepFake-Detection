from glob import glob
import os
import json

labels_file = "/home/ubuntu/prep_data/labels.json"
output_file = "/home/ubuntu/prep_data/cffn_pairs.txt"
metadata = {}
with open(labels_file) as f:
    metadata = json.load(f)

fakes = []
reals = []
for item, label in metadata.items():
    if label == 'FAKE':
        fakes.append(item)
    else:
        reals.append(item)

pairs = []
for i in range(len(fakes)):
    # first create all fake-fake combos for this fake
    first = fakes[i]
    for second in fakes[i+1:]:
        pairs.append((first, second, 1))

    # then create all real-fake combos for this fake
    for real in reals:
        pairs.append((first, real, 0))

# now create all the real-real combos
for i in range(len(reals)):
    first = reals[i]
    for second in reals[i+1:]:
        pairs.append((first, second, 1))

f = open(output_file, "w")
for tup in pairs:
    first, second, key = tup
    f.write(first + " " + second + " " + str(key) + "\n")
f.close()
