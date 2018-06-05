import random

from models.ModelManager import ModelManager

import tensorflow as tf
import numpy as np

dataset_dir = 'tongue_dataset/scaled/'
states = ['mouth_closed', 'mouth_open', 'tongue_down', 'tongue_left',
          'tongue_middle', 'tongue_right', 'tongue_up']


def feed_from_annotation(annotation_path):
    with open(annotation_path) as f:
        annotations = f.readlines()
    annotations = [x.strip().split(',') for x in annotations[:2]]
    rgb_path, depth_path, state, _ = zip(*annotations)
    rgb_path = [x[3:] for x in rgb_path]
    depth_path = [x[3:] for x in depth_path]
    state = [states.index(x) for x in state]
    return rgb_path, depth_path, state


def train():

    # Get train and validation dataset feeds
    train_rgb, train_depth, train_state = feed_from_annotation(dataset_dir + 'train_annotations.txt')

    print('Preparing model')
    # Start up model
    sess = tf.Session()
    model_manager = ModelManager(sess)
    model_manager.prepare_graph()

    print('Setting up summary writer')
    # Setup Tensorboard stuff
    merged_summaries = tf.summary.merge_all()
    summary_writter = tf.summary.FileWriter('logs/' + model_manager.model_name(), sess.graph)

    # Train for a bunch of epochs
    print('Training model')
    epochs = 2
    for epoch in range(epochs):
        model_manager.dataset_init(train_rgb, train_depth, train_state, batch_size=1)
        while True:
            try:
                _, summary = sess.run([model_manager.train_op(), merged_summaries],
                                      feed_dict={model_manager.learning_rate(): 0.1})
            except tf.errors.OutOfRangeError:
                break
        print('Done epoch: %d' % tf.train.global_step(sess, tf.train.get_global_step()))

        summary_writter.add_summary(summary, tf.train.global_step(sess, tf.train.get_global_step()))
        model_manager.save()

    model_manager.save()


def check_accuarcy(dataset_annotation):
    rgb, depth, state = feed_from_annotation(dataset_dir + dataset_annotation)

    # Start up model
    print('Preparing model')
    sess = tf.Session()
    model_manager = ModelManager(sess)
    model_manager.prepare_graph()

    print('Initializing dataset')
    model_manager.dataset_init(rgb, depth, state, batch_size=1)

    print('Collecting model accuracy on dataset')
    accs = []
    while True:
        try:
            accs.append(sess.run([model_manager.avg_acc()]))
        except tf.errors.OutOfRangeError:
            break

    print('Dataset accuracy %.4f' % np.mean(accs))


def run_validation():
    print('Running validation')
    check_accuarcy('validation_annotations.txt')


def run_test():
    print('Running test')
    check_accuarcy('test_annotations.txt')


def inference():
    print('Running inference')

    with open(dataset_dir + 'annotations.txt') as f:
        annotations = f.readlines()
    rand = random.randint(0, len(annotations)-1)
    rgb_path, depth_path, state, _ = annotations[rand].strip().split(',')
    rgb_path = rgb_path[3:]
    depth_path = depth_path[3:]
    state = states.index(state)

    print('State is %d' % state)

    sess = tf.Session()
    model_manager = ModelManager(sess)
    model_manager.prepare_graph()
    model_manager.dataset_init([rgb_path], [depth_path], [state], batch_size=1)
    inference, acc = sess.run([model_manager.inference_op(), model_manager.inference_accuracy()])
    print('Inference: ' + str(inference))
    print('Accuracy: %.4f' % acc)


if __name__ == '__main__':
    run_test()
