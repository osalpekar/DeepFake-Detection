from glob import glob

video_filenames = glob("/home/ubuntu/cs230_data/*/*.mp4")
print(len(video_filenames))
