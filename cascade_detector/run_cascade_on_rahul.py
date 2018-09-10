import os
import numpy as np
import cv2


def main():
    closed_cascade = cv2.CascadeClassifier('C12345.xml')
    down_cascade = cv2.CascadeClassifier('D12345.xml')
    left_cascade = cv2.CascadeClassifier('L12345.xml')
    open_cascade = cv2.CascadeClassifier('O12345.xml')
    right_cascade = cv2.CascadeClassifier('R12345.xml')
    up_cascade = cv2.CascadeClassifier('U12345.xml')

    total_predictions = 0
    correct_predictions = 0

    for path, subdirs, files in os.walk('../RahulTongueTrackerData'):
        for name in files:
            if 'jpg' not in name or '.png' in name:
                continue

            src_rgb = os.path.join(path, name)

            img = cv2.imread(src_rgb)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            closed_results = closed_cascade.detectMultiScale(gray)
            down_results = down_cascade.detectMultiScale(gray)
            left_results = left_cascade.detectMultiScale(gray)
            open_results = open_cascade.detectMultiScale(gray)
            right_results = right_cascade.detectMultiScale(gray)
            up_results = up_cascade.detectMultiScale(gray)

            total_predictions += 1
            if 'C' in name and len(closed_results) > 0:
                correct_predictions += 1
            if 'D' in name and len(down_results) > 0:
                correct_predictions += 1
            if 'L' in name and len(left_results) > 0:     # The two datasets have opposite definition of 'tongue_left' and 'tongue_right'
                correct_predictions += 1
            if 'O' in name and len(open_results) > 0:
                correct_predictions += 1
            if 'R' in name and len(right_results) > 0:     # The two datasets have opposite definition of 'tongue_left' and 'tongue_right'
                correct_predictions += 1
            if 'U' in name and len(up_results) > 0:
                correct_predictions += 1

    print('Images view: %d' % total_predictions)
    print('Correctly classified: %d' % correct_predictions)
    print('Accuracy: %.2f' % (correct_predictions/total_predictions))


if __name__ == '__main__':
    main()