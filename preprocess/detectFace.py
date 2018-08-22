#
# https://github.com/yeephycho/tensorflow-face-detection
#

import os.path
import argparse
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

frame_annotations = '../tongue_dataset/frames/annotations.txt'
not_cropped_annotations = '../tongue_dataset/frames/not_cropped2.txt'
cropped_annotations = '../tongue_dataset/cropped2/annotations.txt'
path_to_ckpt = 'frozen_inference_graph_face.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=detection_graph, config=config)


def detect_faces(image):
    with detection_graph.as_default():
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        boxes, scores = sess.run([boxes, scores], feed_dict={image_tensor: image_np_expanded})
    return boxes[0][0], scores[0][0]


def auto_detect_face(image, thresh=0.5):
    box, score = detect_faces(image)
    if score > thresh:
        return box
    else:
        return None


def get_absolute_dims(img, face):
    box = tuple(face.tolist())
    ymin, xmin, ymax, xmax = box
    im_height, im_width, *_ = img.shape
    (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                  int(ymin * im_height), int(ymax * im_height))
    bottom = bottom + int((bottom - top) * 0.15)  # Add 15% to bottom of bounding box
    top = top + int((bottom - top) * 0.4)         # Take bottom 60% of bounding box
    return top, bottom, left, right


def crop_lower_face(img, face):
    top, bottom, left, right = get_absolute_dims(img, face)
    crop_img = img[top:bottom, left:right]
    return crop_img


def place_bounding_box(img, face):
    top, bottom, left, right = get_absolute_dims(img, face)
    new_img = np.copy(img)
    cv2.rectangle(new_img, (left, top), (right, bottom), (0, 255, 0), 2)
    return new_img


def main(mode):

    rejected_count = 0
    skipped_count = 0
    cropped_count = 0
    not_cropped_count = 0

    # For each set of image paths in frames/annotations.txt
    with open(frame_annotations) as f:
        annotations = f.readlines()
    annotations = [x.strip().split(',') for x in annotations]
    total = len(annotations)
    count = 0
    for src_rgb, src_depth, state in annotations:
        count += 1
        if count % int(total/100) == 0:
            print('Progress: %d' % (count / int(total/100)))

        # Continue if image was already cropped or rejected
        dest_rgb = src_rgb.replace('frames', 'cropped2')
        dest_depth = src_depth.replace('frames', 'cropped2')
        reject_rgb = src_rgb.replace('frames', 'rejected')
        reject_depth = src_depth.replace('frames', 'rejected')
        if os.path.isfile(dest_rgb) or os.path.isfile(reject_rgb):
            skipped_count += 1
            continue

        # Load image (rbg and depth)
        rgb_img = cv2.imread(src_rgb)
        depth_img = cv2.imread(src_depth, -1)

        if mode == 'auto':
            # Make prediction (score >= 0.25)
            face = auto_detect_face(rgb_img, 0.25)

            # If no good prediction, add to not_cropped annotations and continue
            if face is None:
                not_cropped_count += 1
                with open(not_cropped_annotations, 'a') as f:
                    f.write(src_rgb + ',' + src_depth + ',' + state + '\n')
                continue
        else:
            # Find best prediction
            face, _ = detect_faces(rgb_img)

            # Display image with bounding box
            new_img = place_bounding_box(rgb_img, face)
            cv2.imshow('bounding_box', new_img)
            ret = cv2.waitKey(0)

            if ret == 102:                  # 'f' key
                # Reject bounding box
                rejected_count += 1
                cv2.imwrite(reject_rgb, rgb_img)
                cv2.imwrite(reject_depth, depth_img)
                print('Rejected!')
                continue
            elif ret != 106:                # 'j' key
                skipped_count += 1
                continue

        # Crop and resize depth image to align with rgb image
        depth_img = Image.fromarray(depth_img).convert(mode='I')
        depth_img = depth_img.crop((120, 90, 520, 390))
        depth_img = depth_img.resize((640, 480))
        depth_img = np.asanyarray(depth_img).astype(np.uint16)

        # Crop to lower half of face
        rgb_img = crop_lower_face(rgb_img, face)
        depth_img = crop_lower_face(depth_img, face)

        # Save images
        cv2.imwrite(dest_rgb, rgb_img)
        cv2.imwrite(dest_depth, depth_img)

        # Append line to dest annotations.txt
        cropped_count += 1
        with open(cropped_annotations, 'a') as f:
            f.write(dest_rgb + ',' + dest_depth + ',' + state + ',' + mode + '\n')

    # Report stats
    print('Rejected: %d' % rejected_count)
    print('Skipped: %d' % skipped_count)
    print('Cropped: %d' % cropped_count)
    print('Not cropped: %d' % not_cropped_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='detectFace', description='Detect faces and crop lower portion of face')
    parser.add_argument('mode', type=str, choices=['auto', 'manual'])
    args = parser.parse_args()
    main(args.mode)
