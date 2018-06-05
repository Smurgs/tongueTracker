import sys
import cv2
import pygame
import numpy as np
import pyrealsense2 as rs


def display(color_frame, depth_frame):
    # Convert images to numpy arrays     rgb:8-bit,3ch depth:16-bit,1ch
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Scale depth image
    depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

    # Stack both images horizontally
    images = np.hstack((color_image, depth_colormap))

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)


name = input('Enter name: ')

with open('../participants.txt', 'r') as f:
    last_line = f.readlines()[-1]
    case_number = 1 + int(last_line.split('-')[0])

with open('../participants.txt', 'a') as f:
    f.write('%03d-%s\n' % (case_number, name))


# Init pygame
pygame.init()
pygame.display.set_mode((200,200))
clock = pygame.time.Clock()

state = None
capture_frames = False
pipeline = None
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    while True:
        if capture_frames:
            frames = pipeline.wait_for_frames()

        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:

                # Capture and display single frame
                if event.key == pygame.K_SPACE:
                    print('SPACE pressed')
                    state = None
                    pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    pipeline.start(config)
                    frames = None
                    for _ in range(20):
                        frames = pipeline.wait_for_frames()
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    display(color_frame, depth_frame)
                    pipeline.stop()

                if event.key == pygame.K_UP:
                    capture_frames = not capture_frames
                    state = 'tongue_up'

                if event.key == pygame.K_DOWN:
                    capture_frames = not capture_frames
                    state = 'tongue_down'

                if event.key == pygame.K_RIGHT:
                    capture_frames = not capture_frames
                    state = 'tongue_right'

                if event.key == pygame.K_LEFT:
                    capture_frames = not capture_frames
                    state = 'tongue_left'

                if event.key == pygame.K_m:
                    capture_frames = not capture_frames
                    state = 'tongue_middle'

                if event.key == pygame.K_c:
                    capture_frames = not capture_frames
                    state = 'mouth_closed'

                if event.key == pygame.K_o:
                    capture_frames = not capture_frames
                    state = 'mouth_open'

                # Quit program
                if event.key == pygame.K_q:
                    print('Quiting')
                    pygame.quit()
                    sys.exit()

                if capture_frames:
                    # Start recording
                    config.enable_record_to_file('../tongue_dataset/%s/%03d.bag' % (state, case_number))
                    pipeline = rs.pipeline()
                    pipeline.start(config)
                    for _ in range(20):
                        frames = pipeline.wait_for_frames()
                    print('Capturing ' + state)
                elif state is not None:
                    # Stop recording
                    pipeline.stop()
                    print('Finished ' + state)


finally:
    pygame.quit()
