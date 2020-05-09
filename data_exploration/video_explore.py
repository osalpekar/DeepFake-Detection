import os
import cv2
from glob import glob

def main():
    video_filenames = glob("/home/ubuntu/cs230_data/*/*.mp4")

    video_sizes(video_filenames)
    video_frames_analyze(video_filenames)
    image_frame_analyze(video_filenames[0])

def image_frame_analyze(filename):
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    if ret == False:
        print("Read unsuccessful")
        return
    print("Shape of Image (Height, Width, Channels): " + str(frame.shape))
    test_image = "./test.jpg"
    cv2.imwrite(test_image, frame)
    print("File Size of One Frame: " + humansize(os.path.getsize(test_image)))

def video_frames_analyze(video_filenames):
    durations = 0
    fps = 0
    frames = 0
    sub_sample_size = 100
    for fname in video_filenames[:sub_sample_size]:
        vid_fps, vid_frames, vid_duration = analyze_single_video(fname)
        fps += vid_fps
        frames += vid_frames
        durations += vid_duration
    print("Average FPS of videos: " + str(fps/sub_sample_size))
    print("Average Frame Count of videos: " + str(frames/sub_sample_size))
    print("Average Duration Count of videos: " + str(durations/sub_sample_size))

# This function computes the fps, number of frames, and duration in seconds
# for a single video
def analyze_single_video(filename):
    cap = cv2.VideoCapture(filename)
    # Compute the Frames-Per-Second for the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get the number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Compute the duration of the video (in seconds)
    duration = frame_count/fps
    cap.release()
    return fps, frame_count, duration

# This function computes the sizes (in human-readable form) for each video
def video_sizes(video_filenames):
    sizes = [os.path.getsize(vid) for vid in video_filenames if
             os.path.isfile(vid) and "metadata" not in vid]
    print("Min Video File Size: " + humansize(min(sizes)))
    print("Max Video File Size: " + humansize(max(sizes)))
    print("Avg Video File Size: " + humansize(sum(sizes)/len(sizes)))

suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

main()
