import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import init_ops
from tensorflow.contrib.slim.nets import inception

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def get_learning_rate(): return 0.1


def get_batch_size(): return 8


def get_train_vars():
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        train_vars = [tf.get_variable('model/fc1/weights'),
                      tf.get_variable('model/fc1/biases'),
                      tf.get_variable('model/fc2/weights'),
                      tf.get_variable('model/fc2/biases')]
    return train_vars


def assign_variable_values(sess):
    pretrained_variables = ['InceptionV3/Conv2d_1a_3x3/weights',
                            'InceptionV3/Conv2d_1a_3x3/BatchNorm/beta',
                            'InceptionV3/Conv2d_1a_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Conv2d_1a_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Conv2d_2a_3x3/weights',
                            'InceptionV3/Conv2d_2a_3x3/BatchNorm/beta',
                            'InceptionV3/Conv2d_2a_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Conv2d_2a_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Conv2d_2b_3x3/weights',
                            'InceptionV3/Conv2d_2b_3x3/BatchNorm/beta',
                            'InceptionV3/Conv2d_2b_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Conv2d_2b_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Conv2d_3b_1x1/weights',
                            'InceptionV3/Conv2d_3b_1x1/BatchNorm/beta',
                            'InceptionV3/Conv2d_3b_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Conv2d_3b_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Conv2d_4a_3x3/weights',
                            'InceptionV3/Conv2d_4a_3x3/BatchNorm/beta',
                            'InceptionV3/Conv2d_4a_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Conv2d_4a_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5b/Branch_0/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5b/Branch_1/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5b/Branch_1/Conv2d_0b_5x5/weights',
                            'InceptionV3/Mixed_5b/Branch_1/Conv2d_0b_5x5/BatchNorm/beta',
                            'InceptionV3/Mixed_5b/Branch_1/Conv2d_0b_5x5/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5b/Branch_1/Conv2d_0b_5x5/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5b/Branch_2/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5b/Branch_2/Conv2d_0b_3x3/weights',
                            'InceptionV3/Mixed_5b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta',
                            'InceptionV3/Mixed_5b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5b/Branch_2/Conv2d_0c_3x3/weights',
                            'InceptionV3/Mixed_5b/Branch_2/Conv2d_0c_3x3/BatchNorm/beta',
                            'InceptionV3/Mixed_5b/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5b/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5b/Branch_3/Conv2d_0b_1x1/weights',
                            'InceptionV3/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5c/Branch_0/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5c/Branch_1/Conv2d_0b_1x1/weights',
                            'InceptionV3/Mixed_5c/Branch_1/Conv2d_0b_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_5c/Branch_1/Conv2d_0b_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5c/Branch_1/Conv2d_0b_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5c/Branch_1/Conv_1_0c_5x5/weights',
                            'InceptionV3/Mixed_5c/Branch_1/Conv_1_0c_5x5/BatchNorm/beta',
                            'InceptionV3/Mixed_5c/Branch_1/Conv_1_0c_5x5/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5c/Branch_1/Conv_1_0c_5x5/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5c/Branch_2/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5c/Branch_2/Conv2d_0b_3x3/weights',
                            'InceptionV3/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta',
                            'InceptionV3/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5c/Branch_2/Conv2d_0c_3x3/weights',
                            'InceptionV3/Mixed_5c/Branch_2/Conv2d_0c_3x3/BatchNorm/beta',
                            'InceptionV3/Mixed_5c/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5c/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5c/Branch_3/Conv2d_0b_1x1/weights',
                            'InceptionV3/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5d/Branch_0/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_5d/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_5d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5d/Branch_1/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_5d/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_5d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5d/Branch_1/Conv2d_0b_5x5/weights',
                            'InceptionV3/Mixed_5d/Branch_1/Conv2d_0b_5x5/BatchNorm/beta',
                            'InceptionV3/Mixed_5d/Branch_1/Conv2d_0b_5x5/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5d/Branch_1/Conv2d_0b_5x5/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5d/Branch_2/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_5d/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_5d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5d/Branch_2/Conv2d_0b_3x3/weights',
                            'InceptionV3/Mixed_5d/Branch_2/Conv2d_0b_3x3/BatchNorm/beta',
                            'InceptionV3/Mixed_5d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5d/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5d/Branch_2/Conv2d_0c_3x3/weights',
                            'InceptionV3/Mixed_5d/Branch_2/Conv2d_0c_3x3/BatchNorm/beta',
                            'InceptionV3/Mixed_5d/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5d/Branch_2/Conv2d_0c_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_5d/Branch_3/Conv2d_0b_1x1/weights',
                            'InceptionV3/Mixed_5d/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_5d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_5d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6a/Branch_0/Conv2d_1a_1x1/weights',
                            'InceptionV3/Mixed_6a/Branch_0/Conv2d_1a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6a/Branch_0/Conv2d_1a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6a/Branch_0/Conv2d_1a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6a/Branch_1/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_6a/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6a/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6a/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6a/Branch_1/Conv2d_0b_3x3/weights',
                            'InceptionV3/Mixed_6a/Branch_1/Conv2d_0b_3x3/BatchNorm/beta',
                            'InceptionV3/Mixed_6a/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6a/Branch_1/Conv2d_0b_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6a/Branch_1/Conv2d_1a_1x1/weights',
                            'InceptionV3/Mixed_6a/Branch_1/Conv2d_1a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6a/Branch_1/Conv2d_1a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6a/Branch_1/Conv2d_1a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6b/Branch_0/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_6b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6b/Branch_1/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_6b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6b/Branch_1/Conv2d_0b_1x7/weights',
                            'InceptionV3/Mixed_6b/Branch_1/Conv2d_0b_1x7/BatchNorm/beta',
                            'InceptionV3/Mixed_6b/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6b/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6b/Branch_1/Conv2d_0c_7x1/weights',
                            'InceptionV3/Mixed_6b/Branch_1/Conv2d_0c_7x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6b/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6b/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0b_7x1/weights',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0b_7x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0b_7x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0b_7x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0c_1x7/weights',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0c_1x7/BatchNorm/beta',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0c_1x7/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0c_1x7/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0d_7x1/weights',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0d_7x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0d_7x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0d_7x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0e_1x7/weights',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0e_1x7/BatchNorm/beta',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0e_1x7/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6b/Branch_2/Conv2d_0e_1x7/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6b/Branch_3/Conv2d_0b_1x1/weights',
                            'InceptionV3/Mixed_6b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6c/Branch_0/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_6c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6c/Branch_1/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_6c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6c/Branch_1/Conv2d_0b_1x7/weights',
                            'InceptionV3/Mixed_6c/Branch_1/Conv2d_0b_1x7/BatchNorm/beta',
                            'InceptionV3/Mixed_6c/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6c/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6c/Branch_1/Conv2d_0c_7x1/weights',
                            'InceptionV3/Mixed_6c/Branch_1/Conv2d_0c_7x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6c/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6c/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0b_7x1/weights',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0b_7x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0b_7x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0b_7x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0c_1x7/weights',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0c_1x7/BatchNorm/beta',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0c_1x7/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0c_1x7/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0d_7x1/weights',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0d_7x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0d_7x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0d_7x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0e_1x7/weights',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0e_1x7/BatchNorm/beta',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0e_1x7/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6c/Branch_2/Conv2d_0e_1x7/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6c/Branch_3/Conv2d_0b_1x1/weights',
                            'InceptionV3/Mixed_6c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6d/Branch_0/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_6d/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6d/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6d/Branch_1/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_6d/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6d/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6d/Branch_1/Conv2d_0b_1x7/weights',
                            'InceptionV3/Mixed_6d/Branch_1/Conv2d_0b_1x7/BatchNorm/beta',
                            'InceptionV3/Mixed_6d/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6d/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6d/Branch_1/Conv2d_0c_7x1/weights',
                            'InceptionV3/Mixed_6d/Branch_1/Conv2d_0c_7x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6d/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6d/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0b_7x1/weights',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0b_7x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0b_7x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0b_7x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0c_1x7/weights',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0c_1x7/BatchNorm/beta',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0c_1x7/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0c_1x7/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0d_7x1/weights',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0d_7x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0d_7x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0d_7x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0e_1x7/weights',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0e_1x7/BatchNorm/beta',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0e_1x7/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6d/Branch_2/Conv2d_0e_1x7/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6d/Branch_3/Conv2d_0b_1x1/weights',
                            'InceptionV3/Mixed_6d/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6e/Branch_0/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_6e/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6e/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6e/Branch_1/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_6e/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6e/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6e/Branch_1/Conv2d_0b_1x7/weights',
                            'InceptionV3/Mixed_6e/Branch_1/Conv2d_0b_1x7/BatchNorm/beta',
                            'InceptionV3/Mixed_6e/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6e/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6e/Branch_1/Conv2d_0c_7x1/weights',
                            'InceptionV3/Mixed_6e/Branch_1/Conv2d_0c_7x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6e/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6e/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0b_7x1/weights',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0b_7x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0b_7x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0b_7x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0c_1x7/weights',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0c_1x7/BatchNorm/beta',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0c_1x7/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0c_1x7/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0d_7x1/weights',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0d_7x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0d_7x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0d_7x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0e_1x7/weights',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0e_1x7/BatchNorm/beta',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0e_1x7/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6e/Branch_2/Conv2d_0e_1x7/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_6e/Branch_3/Conv2d_0b_1x1/weights',
                            'InceptionV3/Mixed_6e/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_6e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_6e/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7a/Branch_0/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_7a/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7a/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7a/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7a/Branch_0/Conv2d_1a_3x3/weights',
                            'InceptionV3/Mixed_7a/Branch_0/Conv2d_1a_3x3/BatchNorm/beta',
                            'InceptionV3/Mixed_7a/Branch_0/Conv2d_1a_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7a/Branch_0/Conv2d_1a_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_0b_1x7/weights',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_0b_1x7/BatchNorm/beta',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_0b_1x7/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_0c_7x1/weights',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_0c_7x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_0c_7x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_1a_3x3/weights',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_1a_3x3/BatchNorm/beta',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_1a_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7a/Branch_1/Conv2d_1a_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7b/Branch_0/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_7b/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7b/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7b/Branch_1/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_7b/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7b/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_1x3/weights',
                            'InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_1x3/BatchNorm/beta',
                            'InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_3x1/weights',
                            'InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_3x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_3x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_3x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0b_3x3/weights',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0b_3x3/BatchNorm/beta',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0c_1x3/weights',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0c_1x3/BatchNorm/beta',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0c_1x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0c_1x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0d_3x1/weights',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0d_3x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0d_3x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7b/Branch_2/Conv2d_0d_3x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7b/Branch_3/Conv2d_0b_1x1/weights',
                            'InceptionV3/Mixed_7b/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7c/Branch_0/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_7c/Branch_0/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7c/Branch_0/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7c/Branch_1/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_7c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7c/Branch_1/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7c/Branch_1/Conv2d_0b_1x3/weights',
                            'InceptionV3/Mixed_7c/Branch_1/Conv2d_0b_1x3/BatchNorm/beta',
                            'InceptionV3/Mixed_7c/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7c/Branch_1/Conv2d_0b_1x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7c/Branch_1/Conv2d_0c_3x1/weights',
                            'InceptionV3/Mixed_7c/Branch_1/Conv2d_0c_3x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7c/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7c/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0a_1x1/weights',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0a_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0a_1x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0b_3x3/weights',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0b_3x3/BatchNorm/beta',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0b_3x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0c_1x3/weights',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0c_1x3/BatchNorm/beta',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0c_1x3/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0c_1x3/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0d_3x1/weights',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0d_3x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0d_3x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7c/Branch_2/Conv2d_0d_3x1/BatchNorm/moving_variance',
                            'InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/weights',
                            'InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/BatchNorm/beta',
                            'InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean',
                            'InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance',
                            'InceptionV3/AuxLogits/Conv2d_1b_1x1/weights',
                            'InceptionV3/AuxLogits/Conv2d_1b_1x1/BatchNorm/beta',
                            'InceptionV3/AuxLogits/Conv2d_1b_1x1/BatchNorm/moving_mean',
                            'InceptionV3/AuxLogits/Conv2d_1b_1x1/BatchNorm/moving_variance',
                            'InceptionV3/AuxLogits/Conv2d_2a_5x5/weights',
                            'InceptionV3/AuxLogits/Conv2d_2a_5x5/BatchNorm/beta',
                            'InceptionV3/AuxLogits/Conv2d_2a_5x5/BatchNorm/moving_mean',
                            'InceptionV3/AuxLogits/Conv2d_2a_5x5/BatchNorm/moving_variance',
                            'InceptionV3/AuxLogits/Conv2d_2b_1x1/weights',
                            'InceptionV3/AuxLogits/Conv2d_2b_1x1/biases',
                            'InceptionV3/Logits/Conv2d_1c_1x1/weights',
                            'InceptionV3/Logits/Conv2d_1c_1x1/biases']

    with tf.device('/cpu:0'):
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            rgb_inception_vars = {}
            for var in pretrained_variables:
                rgb_inception_vars[var] = tf.get_variable('model/rgb_inception/' + var)

        rgb_saver = tf.train.Saver(rgb_inception_vars)
        rgb_saver.restore(sess, 'models/inception_v3.ckpt')


def get_model_name(): return 'RGB_Inception2'


def build_model(rgb_x, depth_x, y, batch_size, reuse, training_ph, outputs):

    name_scope = tf.contrib.framework.get_name_scope()

    # Build graph
    with tf.variable_scope('model'):
        # Create an inception model on cpu that never gets used.
        # Other instances of the model will use the variables that were created on the cpu
        if not reuse:
            with tf.device('/cpu:0'):
                inception_input_size = tf.zeros([batch_size, 299, 299, 3])
                with tf.name_scope('cpu_inception'):
                    with tf.variable_scope('rgb_inception'):
                        with tf.contrib.slim.arg_scope(inception.inception_v3_arg_scope()):
                            inception.inception_v3(inception_input_size, 1001)

        with tf.variable_scope('rgb_inception', reuse=True):
            rgb_out = tf.image.resize_images(rgb_x, (299, 299))
            with tf.contrib.slim.arg_scope(inception.inception_v3_arg_scope()):
                inception.inception_v3(rgb_out, 1001)

        model_logits = tf.get_default_graph().get_tensor_by_name(name_scope + '/model/rgb_inception/InceptionV3/Logits/Dropout_1b/dropout/mul:0')
        model_logits = layers.conv2d(model_logits, 2048, [1, 1], weights_initializer=trunc_normal(0.001), scope='fc1')
        model_logits = layers.conv2d(model_logits, outputs, [1, 1], activation_fn=None, normalizer_fn=None,
                                     weights_initializer=trunc_normal(0.001), scope='fc2')
        model_logits = tf.squeeze(model_logits)
        model_logits = tf.reshape(model_logits, [-1, outputs])

    # Inference
    with tf.variable_scope('inference'):
        inference = tf.identity(tf.nn.softmax(model_logits), name='inference')

    # Loss
    with tf.variable_scope('loss'):
        tf.losses.softmax_cross_entropy(y, model_logits)

    # Accuracy
    with tf.variable_scope('accuracy'):
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(inference, 1)), tf.float32), name='acc')
        tf.add_to_collection('accuracy_collection', acc)

