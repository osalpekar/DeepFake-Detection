import os
import json
import cv2
from glob import glob
import sys

video_filenames = glob("/home/ubuntu/cs230_data/*/*.mp4")
num_examples = len(video_filenames) # change to use a smaller subset of the data
subset_vid_names = video_filenames[:num_examples]
meta_files = glob("/home/ubuntu/cs230_data/*/metadata.json")

sample_frames = [0, 60, 120, 180, 240]

# TODO: make this a util and dedupe with equivalent function in the EDA folder
metadata = {}
for meta_file in meta_files:
    with open(meta_file) as f:
        data = json.load(f)
    metadata.update(data)

labels = {}

for vid in subset_vid_names:
    vid_name = os.path.basename(vid)
    label = metadata[vid_name]['label']
    for sample in sample_frames:
        name = vid_name.split(".")[0] + "_" + str(sample) + ".jpg"
        labels[name] = label

print(len(labels))
print(list(labels.keys())[654])
print(list(labels.values())[654])
print(list(labels.keys())[78451])
print(list(labels.values())[78451])
with open("/home/ubuntu/big_data/labels.json", "w+") as fp:
    json.dump(labels, fp)

count = 0
# experimenting with sampling only once from the model. The baseline model
# sampled 4 times and so the naming sequence is slightly different in the
# PyTorch DataLoader. This current experiment samples the last frame only
# (which we were not sampling for the baseline).
sampling_frequency = 300
last_name = ""
for vid in subset_vid_names:
    img_name = os.path.basename(vid).split(".")[0]
    cap = cv2.VideoCapture(vid)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        if i in sample_frames:
            name = "/home/ubuntu/big_data/" + img_name + "_" + str(i) + ".jpg"
            cv2.imwrite(name, frame)
            last_name = name
        i += 1
    count += 1
    cap.release()
    if count == 1:
        print(last_name)
    if count % 50 == 0:
        print("vids completed: " + str(count))
