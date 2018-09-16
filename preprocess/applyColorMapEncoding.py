import cv2


dataset_dir = '../tongue_dataset/scaled2/'


def applyColorMapEncoding(file_name):
    # Read train annotations
    with open(dataset_dir + file_name + '.txt') as f:
        annotations = f.readlines()
    annotations = [x.strip().split(',') for x in annotations]

    with open(dataset_dir + file_name + '_colormap.txt', 'a') as f:
        with open(dataset_dir + file_name + '_6classes_colormap.txt', 'a') as d:
            for rgb, depth, state, mode in annotations:

                depth_img = cv2.imread(depth, -1)
                depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

                dest_depth = depth[:-4] + '_colormap.png'
                cv2.imwrite(dest_depth, depth_img)
                f.write(rgb + ',' + dest_depth + ',' + state + ',' + mode + '\n')

                if 'tongue_middle' != state:
                    d.write(rgb + ',' + dest_depth + ',' + state + ',' + mode + '\n')


if __name__ == '__main__':
    applyColorMapEncoding('train_annotations2')
    applyColorMapEncoding('val_annotations2')
    applyColorMapEncoding('test_annotations2')