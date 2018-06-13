import random

train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

annotations_path = '../tongue_dataset/scaled/annotations.txt'


def v1():
    # Read annotations file
    with open(annotations_path) as f:
        annotations = f.readlines()
    annotations = [x.strip().split(',') for x in annotations]

    # Shuffle annotations
    random.shuffle(annotations)

    # Split into train/validation/test sets
    total = len(annotations)
    train_annotations = annotations[:int(train_ratio * total)]
    validation_annotations = annotations[int(train_ratio * total):int((train_ratio + validation_ratio) * total)]
    test_annotations = annotations[int((train_ratio + validation_ratio) * total):]

    # Write new annotation files
    for rgb, depth, state, mode in train_annotations:
        train_annotations_path = annotations_path.replace('annotations', 'train_annotations')
        with open(train_annotations_path, 'a') as f:
            f.write(rgb + ',' + depth + ',' + state + ',' + mode + '\n')

    for rgb, depth, state, mode in validation_annotations:
        validation_annotations_path = annotations_path.replace('annotations', 'validation_annotations')
        with open(validation_annotations_path, 'a') as f:
            f.write(rgb + ',' + depth + ',' + state + ',' + mode + '\n')

    for rgb, depth, state, mode in test_annotations:
        test_annotations_path = annotations_path.replace('annotations', 'test_annotations')
        with open(test_annotations_path, 'a') as f:
            f.write(rgb + ',' + depth + ',' + state + ',' + mode + '\n')


def v2():
    # Read annotations file
    with open(annotations_path) as f:
        annotations = f.readlines()
    annotations = [x.strip().split(',') for x in annotations]

    # Shuffle annotations
    random.shuffle(annotations)

    # Write new annotation files
    train_annotations_path = annotations_path.replace('annotations', 'train_annotations2')
    val_annotations_path = annotations_path.replace('annotations', 'val_annotations2')
    test_annotations_path = annotations_path.replace('annotations', 'test_annotations2')
    for rgb, depth, state, mode in annotations:
        if '/007_' in rgb:
            with open(test_annotations_path, 'a') as f:
                f.write(rgb + ',' + depth + ',' + state + ',' + mode + '\n')
        elif '/014_' in rgb:
            with open(val_annotations_path, 'a') as f:
                f.write(rgb + ',' + depth + ',' + state + ',' + mode + '\n')
        else:
            with open(train_annotations_path, 'a') as f:
                f.write(rgb + ',' + depth + ',' + state + ',' + mode + '\n')


if __name__ == '__main__':
    v2()