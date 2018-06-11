import random
import time

from models.ModelManager import ModelManager

import tensorflow as tf
import numpy as np

epochs = 200
batch_size = 32
dataset_parent_dir = '/scratch/smurga/'
dataset_dir = '/scratch/smurga/tongue_dataset/scaled/'
states = ['mouth_closed', 'mouth_open', 'tongue_down', 'tongue_left',
          'tongue_middle', 'tongue_right', 'tongue_up']


def feed_from_annotation(annotation_path):
    with open(annotation_path) as f:
        annotations = f.readlines()
    annotations = [x.strip().split(',') for x in annotations]
    rgb_path, depth_path, state, _ = zip(*annotations)
    rgb_path = [dataset_parent_dir + x[3:] for x in rgb_path]
    depth_path = [dataset_parent_dir + x[3:] for x in depth_path]
    state = [states.index(x) for x in state]
    return rgb_path, depth_path, state


def add_static_summary(writer, identifier, value, sess):
    summary = tf.Summary()
    summary.value.add(tag=identifier, simple_value=value)
    writer.add_summary(summary, tf.train.global_step(sess, tf.train.get_global_step()))


def train():

    # Get train and validation dataset feeds
    train_rgb, train_depth, train_state = feed_from_annotation(dataset_dir + 'train_annotations.txt')
    val_rgb, val_depth, val_state = feed_from_annotation(dataset_dir + 'validation_annotations.txt')

    # Start up model
    print('Preparing model')
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config=config)
    model_manager = ModelManager(sess)
    model_manager.prepare_graph()

    # Setup Tensorboard stuff
    print('Setting up summary writers')
    merged_summaries = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter('logs/' + model_manager.model_name() + '/events/train/', sess.graph)
    val_summary_writer = tf.summary.FileWriter('logs/' + model_manager.model_name() + '/events/validation/', sess.graph)

    # Train for a bunch of epochs
    print('Training model for %d epochs' % epochs)
    epoch_times = []
    for epoch in range(epochs):

        # Learn on training data for an epoch
        epoch_start_time = time.time()
        model_manager.dataset_init(train_rgb, train_depth, train_state, batch_size=batch_size)
        while True:
            try:
                _, summary = sess.run([model_manager.train_op(), merged_summaries],
                                      feed_dict={model_manager.learning_rate(): 0.1})
                train_summary_writer.add_summary(summary, tf.train.global_step(sess, tf.train.get_global_step()))
            except tf.errors.OutOfRangeError:
                break

        # Write summary and save model
        epoch_end_time = time.time() - epoch_start_time
        epoch_times.append(epoch_end_time)
        add_static_summary(train_summary_writer, 'summaries/epoch_time', epoch_end_time, sess)
        print('Done epoch # %d in %d seconds' % (epoch, epoch_end_time))
        model_manager.save()

        # Collect loss and acc on validation dataset
        model_manager.dataset_init(val_rgb, val_depth, val_state, batch_size=batch_size)
        val_losses = []
        val_accs = []
        while True:
            try:
                batch_loss, batch_acc = sess.run([model_manager.avg_loss(), model_manager.avg_acc()],
                                                  feed_dict={model_manager.learning_rate(): 0})
                val_losses.append(batch_loss)
                val_accs.append(batch_acc)
            except tf.errors.OutOfRangeError:
                break
        add_static_summary(val_summary_writer, 'summaries/avg_loss', np.mean(val_losses), sess)
        add_static_summary(val_summary_writer, 'summaries/avg_acc', np.mean(val_accs), sess)

    # Print runtime
    print('Finished training %d epochs in %d seconds' % (epochs, np.sum(epoch_times)))


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
    train()
