import os.path
import numpy as np
import cv2
from PIL import Image
from PIL import ImageFilter

cropped_annotation = '../tongue_dataset/cropped2/annotations.txt'
scaled_annotation = '../tongue_dataset/scaled2/annotations.txt'
target_width = 227
target_height = 227


def main():

    # Loop through sources annotation file
    with open(cropped_annotation) as f:
        annotations = f.readlines()
    annotations = [x.strip().split(',') for x in annotations]
    total = len(annotations)
    count = 0
    for annotation in annotations:
        src_rgb, src_depth, state, mode = annotation
        dest_rgb = src_rgb.replace('cropped2', 'scaled2')
        dest_depth = src_depth.replace('cropped2', 'scaled2')

        count += 1
        if count % int(total / 100) == 0:
            print('Progress: %d' % (count / int(total / 100)))

        # Check if source files exist
        if not os.path.isfile(src_rgb) or not os.path.isfile(src_depth):
            print('One of these files does not exist: "%s" "%s"' % (src_rgb, src_depth))
            continue

        # Check if destination files exist
        if os.path.isfile(dest_rgb) or os.path.isfile(dest_depth):
            continue

        # Read and get image info
        rgb_fg = Image.open(src_rgb)
        rgb_bg = rgb_fg.copy()
        depth_fg = Image.open(src_depth)
        depth_bg = depth_fg.copy()
        width, height = rgb_fg.size

        # Resize background and foreground images
        if width/height <= target_width/target_height:
            rgb_bg = rgb_bg.resize((target_width, int(height * (target_width/width))))
            depth_bg = depth_bg.resize((target_width, int(height * (target_width/width))))

            start_crop = (rgb_bg.size[1]-target_height) / 2
            rgb_bg = rgb_bg.crop((0, start_crop, target_width, start_crop + target_height))
            depth_bg = depth_bg.crop((0, start_crop, target_width, start_crop + target_height))

            rgb_fg = rgb_fg.resize((int(width * (target_height/height)), target_height))
            depth_fg = depth_fg.resize((int(width * (target_height/height)), target_height))

            fg_top_left_corner = (int((target_width-rgb_fg.size[0]) / 2), 0)

        else:
            rgb_bg = rgb_bg.resize((int(width * (target_height/height)), target_height))
            depth_bg = depth_bg.resize((int(width * (target_height/height)), target_height))

            start_crop = (rgb_bg.size[0]-target_width) / 2
            rgb_bg = rgb_bg.crop((start_crop, 0, start_crop + target_width, target_height))
            depth_bg = depth_bg.crop((start_crop, 0, start_crop + target_width, target_height))

            rgb_fg = rgb_fg.resize((target_width, int(height * (target_width/width))))
            depth_fg = depth_fg.resize((target_width, int(height * (target_width/width))))

            fg_top_left_corner = (0, int((target_height - rgb_fg.size[1]) / 2))

        # Blur background
        rgb_bg = Image.fromarray(cv2.GaussianBlur(np.asanyarray(rgb_bg), (5, 5), 0))
        depth_bg = Image.fromarray(cv2.GaussianBlur(np.asanyarray(depth_bg).astype(np.uint16), (5, 5), 0)).convert(mode='I')

        # Copy fg onto bg
        rgb_bg.paste(rgb_fg, fg_top_left_corner)
        depth_bg.paste(depth_fg, fg_top_left_corner)

        # Resize again, in case there was a rounding error
        rgb_bg = rgb_bg.resize((target_width, target_height))
        depth_bg = depth_bg.resize((target_width, target_height))

        # Append line to dest annotations.txt
        with open(scaled_annotation, 'a') as f:
            f.write(dest_rgb + ',' + dest_depth + ',' + state + ',' + mode + '\n')

        # Save images
        rgb_bg.save(dest_rgb)
        depth_bg.save(dest_depth)


if __name__ == '__main__':
    main()