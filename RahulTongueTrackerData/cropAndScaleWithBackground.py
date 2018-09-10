import os.path
import os
import numpy as np
import cv2
from PIL import Image

# 'cascade_detector/MouthDataSet/scaled/p1c-l1-0001.png,doesntmatter.png,mouth_closed,auto'

scaled_annotation = '../RahulTongueTrackerData/scaled/annotations.txt'
target_width = 227
target_height = 227


def main():

    count = 0
    for path, subdirs, files in os.walk('../RahulTongueTrackerData'):
        for name in files:
            if 'jpg' not in name or '.png' in name:
                continue

            src_rgb = os.path.join(path, name)
            state = None
            if 'C' in name: state = 'mouth_closed'
            if 'D' in name: state = 'tongue_down'
            if 'L' in name: state = 'tongue_right'    # The two datasets have opposite definition of 'tongue_left' and 'tongue_right'
            if 'O' in name: state = 'mouth_open'
            if 'R' in name: state = 'tongue_left'     # The two datasets have opposite definition of 'tongue_left' and 'tongue_right'
            if 'U' in name: state = 'tongue_up'

            dest_rgb = 'RahulTongueTrackerData/scaled/img' + str(count) + '.png'
            count += 1


            # Read and get image info
            rgb_fg = Image.open(src_rgb)
            width, height = rgb_fg.size

            # Crop image to lower half to get mouth only
            rgb_fg = rgb_fg.crop((0, height*0.5, width, height))

            # Make copy for background
            rgb_bg = rgb_fg.copy()

            # Resize background and foreground images
            if width/height <= target_width/target_height:
                rgb_bg = rgb_bg.resize((target_width, int(height * (target_width/width))))

                start_crop = (rgb_bg.size[1]-target_height) / 2
                rgb_bg = rgb_bg.crop((0, start_crop, target_width, start_crop + target_height))

                rgb_fg = rgb_fg.resize((int(width * (target_height/height)), target_height))

                fg_top_left_corner = (int((target_width-rgb_fg.size[0]) / 2), 0)

            else:
                rgb_bg = rgb_bg.resize((int(width * (target_height/height)), target_height))

                start_crop = (rgb_bg.size[0]-target_width) / 2
                rgb_bg = rgb_bg.crop((start_crop, 0, start_crop + target_width, target_height))

                rgb_fg = rgb_fg.resize((target_width, int(height * (target_width/width))))

                fg_top_left_corner = (0, int((target_height - rgb_fg.size[1]) / 2))

            # Blur background
            rgb_bg = Image.fromarray(cv2.GaussianBlur(np.asanyarray(rgb_bg), (5, 5), 0))

            # Copy fg onto bg
            rgb_bg.paste(rgb_fg, fg_top_left_corner)

            # Resize again, in case there was a rounding error
            rgb_bg = rgb_bg.resize((target_width, target_height))

            # Append line to dest annotations.txt
            with open(scaled_annotation, 'a') as f:
                f.write('../' + dest_rgb + ',' + '../' + dest_rgb + ',' + state + ',' + 'auto' + '\n')

            # Save images
            rgb_bg.save('../' + dest_rgb)


if __name__ == '__main__':
    main()