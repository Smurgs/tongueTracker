import os
import sys
import datetime
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image


def save(color_frame, depth_frame, frame_number, state):
    # Convert images to numpy arrays     rgb:8-bit,3ch depth:16-bit,1ch
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Save image from both streams       rgb:8-bit,3ch depth:32-bit,1ch
    color_im = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    color_path = '../tongue_dataset/frames/%s/%03d_%04d_rgb.png' % (state, case_number, frame_number)
    color_im.save(color_path)
    depth_im = Image.fromarray(depth_image).convert(mode='I')
    depth_path = '../tongue_dataset/frames/%s/%03d_%04d_depth.png' % (state, case_number, frame_number)
    depth_im.save(depth_path)

    # Append file name to annotations
    with open('../tongue_dataset/frames/annotations.txt', 'a') as f:
        f.write(color_path + ',' + depth_path + ',' + state + '\n')


if len(sys.argv) < 2:
    print('Need bag filename')


for filename in sys.argv[1:]:

    bag_filename = filename
    state = bag_filename.split('/')[2]
    case_number = int(bag_filename.split('/')[-1].split('.')[0])

    # Skip if already done bag file
    if os.path.isfile('../tongue_dataset/frames/%s/%03d_0001_rgb.png' % (state, case_number)):
        print('Skipping %s - %03d' % (state, case_number))
        continue

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_filename, repeat_playback=False)

    # Start streaming
    playback = pipeline.start(config).get_device().as_playback()
    frames = pipeline.wait_for_frames()
    playback.pause()
    duration = playback.get_duration()
    loc = datetime.timedelta()

    frame_number = 0

    try:
        while True:
            loc += datetime.timedelta(milliseconds=100)  # 10fps
            if loc > duration:
                print('Done %s - %03d' % (state, case_number))
                break
            playback.seek(loc)

            if pipeline.poll_for_frames(frames):
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                frame_number += 1
                save(color_frame, depth_frame, frame_number, state)
    finally:
        pipeline.stop()

