import cv2


dataset_dir = '../tongue_dataset/scaled/'


def applyColorMapEncoding(file_name):
    # Read train annotations
    with open(dataset_dir + file_name + '.txt') as f:
        annotations = f.readlines()
    annotations = [x.strip().split(',') for x in annotations]

    with open(dataset_dir + file_name.replace('_', '_colormap_') + '_6classes.txt', 'a') as f:
        for rgb, depth, state, mode in annotations[:1]:
                f.write(rgb + ',' + depth + ',' + state + ',' + mode + '\n')


if __name__ == '__main__':
    #applyColorMapEncoding('train_annotations2')
    #applyColorMapEncoding('val_annotations2')
    applyColorMapEncoding('test_annotations2')