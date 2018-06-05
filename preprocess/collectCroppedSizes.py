import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt

cropped_annoation = '../tongue_dataset/cropped/annotations.txt'


def main():

    dne_lst = []
    height_lst = []
    width_lst = []
    w_to_h_ratio_lst = []
    auto_count = 0
    manual_count = 0

    # Loop through annotations file
    with open(cropped_annoation) as f:
        annotations = f.readlines()
    annotations = [x.strip().split(',') for x in annotations]
    for src_rgb, src_depth, state, mode in annotations:

        # Check if files exist
        if not os.path.isfile(src_rgb) or not os.path.isfile(src_depth):
            dne_lst.append((src_rgb, src_depth, state, mode))
            continue

        # Get size of image
        rgb_img = cv2.imread(src_rgb)
        height, width, _ = rgb_img.shape
        height_lst.append(height)
        width_lst.append(width)
        w_to_h_ratio_lst.append(round(width/height, 4))

        # Get mode
        if mode == 'auto':
            auto_count += 1
        else:
            manual_count += 1

    # Display stuff
    print('Dne length: %d' % len(dne_lst))
    print('Dne list:')
    print(dne_lst)
    print('Auto length: %d' % auto_count)
    print('Manual length %d' % manual_count)
    print('Total length: %d' % (auto_count + manual_count))
    print('Average height: %.2f' % (np.average(np.array(height_lst))))
    print('Average width: %.2f' % (np.average(np.array(width_lst))))
    print('Ratio of avg w and h: %.2f' % (np.average(np.array(width_lst)) / np.average(np.array(height_lst))))
    print('Average ratio: %.2f' % (np.average(np.array(w_to_h_ratio_lst))))
    plt.subplot(2,3,1)
    plt.title('Width to height ratio')
    plt.boxplot(np.array(w_to_h_ratio_lst))
    plt.subplot(2,3,4)
    plt.hist(np.array(w_to_h_ratio_lst))
    plt.subplot(2,3,2)
    plt.title('Width')
    plt.boxplot(np.array(width_lst))
    plt.subplot(2,3,5)
    plt.hist(np.array(width_lst))
    plt.subplot(2,3,3)
    plt.title('Height')
    plt.boxplot(np.array(height_lst))
    plt.subplot(2,3,6)
    plt.hist(np.array(height_lst))
    plt.show()


if __name__ == '__main__':
    main()
