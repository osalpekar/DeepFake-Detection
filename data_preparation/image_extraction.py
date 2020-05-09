import os
import json
import cv2
from glob import glob

video_filenames = glob("/home/ubuntu/cs230_data/*/*.mp4")
num_examples = len(video_filenames) # change to use a smaller subset of the data
subset_vid_names = video_filenames[:num_examples]
meta_files = "/home/ubuntu/cs230_data/*/metadata.json"

# TODO: make this a util and dedupe with equivalent function in the EDA folder
metadata = {}
for meta_file in meta_files:
    with open(meta_file) as f:
	data = json.load(f)
    metadata.update(data)

labels = {}

for vid in subset_vid_names:
    name = os.path.basename(vid)
    label = metadata[name]['label']
    name = name.split(".")[0] + ".jpg"
    labels[name] = label

with open("/home/ubuntu/prep_data/labels.json", "w") as fp:
    json.dump(labels, fp)

count = 0
# experimenting with sampling only once from the model. The baseline model
# sampled 4 times and so the naming sequence is slightly different in the
# PyTorch DataLoader. This current experiment samples the last frame only
# (which we were not sampling for the baseline).
sampling_frequency = 300
for vid in subset_vid_names:
    name = os.path.basename(vid).split(".")[0]
    name = "/home/ubuntu/prep_data/" + name + ".jpg"
    cap = cv2.VideoCapture(vid)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        if i % sampling_frequency == 0:
            # we're only sampling once so we can retain the original name
            cv2.imwrite(name, frame)
        i += 1
    count += 1
    cap.release()
    if count % 500 == 0:
        print("images completed: " + str(count))
