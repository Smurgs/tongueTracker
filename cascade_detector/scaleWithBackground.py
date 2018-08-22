import os.path
import os
import numpy as np
import cv2
from PIL import Image

# 'cascade_detector/MouthDataSet/scaled/p1c-l1-0001.png,doesntmatter.png,mouth_closed,auto'

scaled_annotation = '../cascade_detector/MouthDataSet/scaled/annotations.txt'
target_width = 227
target_height = 227


def main():

    for path, subdirs, files in os.walk('MouthDataSet'):
        for name in files:
            if '.png' not in name:
                continue

            src_rgb = os.path.join(path, name)
            state = None
            if 'c-' in name: state = 'mouth_closed'
            if 'd-' in name: state = 'tongue_down'
            if 'l-' in name: state = 'tongue_right'    # The two datasets have opposite definition of 'tongue_left' and 'tongue_right'
            if 'o-' in name: state = 'mouth_open'
            if 'r-' in name: state = 'tongue_left'     # The two datasets have opposite definition of 'tongue_left' and 'tongue_right'
            if 'u-' in name: state = 'tongue_up'

            dest_rgb = 'MouthDataSet/scaled/' + name


            # Read and get image info
            rgb_fg = Image.open(src_rgb)
            rgb_bg = rgb_fg.copy()
            width, height = rgb_fg.size

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
            rgb_bg.save(dest_rgb)


if __name__ == '__main__':
    main()