dataset_dir = '../tongue_dataset/scaled2/'


def removeTongueMiddle(file_name):
    # Read train annotations
    with open(dataset_dir + file_name + '.txt') as f:
        annotations = f.readlines()
    annotations = [x.strip().split(',') for x in annotations]

    with open(dataset_dir + file_name + '_6classes.txt', 'a') as f:
        for rgb, depth, state, mode in annotations:
            if 'tongue_middle' != state:
                f.write(rgb + ',' + depth + ',' + state + ',' + mode + '\n')


if __name__ == '__main__':
    removeTongueMiddle('train_annotations2')
    removeTongueMiddle('val_annotations2')
    removeTongueMiddle('test_annotations2')