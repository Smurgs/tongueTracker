import h5py
import numpy as np
import tensorflow as tf


def assign_variable_values(sess):

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        pass


def get_model_name():
    return 'RGBD_AlexNet_Pretrained2'


def build_model(rgb_x, depth_x, y, reuse=False):

    # Combine rgb and depth data
    x = tf.concat([rgb_x, depth_x], axis=-1)

    # Load pretrained variables
    f = h5py.File('models/alexnet_weights.h5', 'r')

    # Create variables
    with tf.variable_scope('model'):
        with tf.variable_scope('conv_variables'):
            with tf.device('/cpu:0'):
                # Conv1
                conv1W_values = np.transpose(np.array(f['conv_1']['conv_1_W']))
                conv1Wa = tf.get_variable('conv1Wa', initializer=tf.constant(conv1W_values))
                conv1Wb = tf.get_variable('conv1Wb', shape=[11, 11, 1, 96], initializer=tf.contrib.layers.xavier_initializer())
                conv1W = tf.concat([conv1Wa, conv1Wb], axis=2)
                conv1B_values = np.array(f['conv_1']['conv_1_b'])
                conv1B = tf.get_variable('conv1B', initializer=tf.constant(conv1B_values))

                # Conv2
                conv2W_values = np.concatenate(
                    [
                        np.concatenate(
                            [np.transpose(np.array(f['conv_2_1']['conv_2_1_W'])),
                             np.transpose(np.array(f['conv_2_1']['conv_2_1_W']))], axis=2),
                        np.concatenate(
                            [np.transpose(np.array(f['conv_2_2']['conv_2_2_W'])),
                             np.transpose(np.array(f['conv_2_2']['conv_2_2_W']))], axis=2)
                    ], axis=3)
                conv2W = tf.get_variable('conv2W', initializer=tf.constant(conv2W_values))
                conv2B_values = np.concatenate([np.array(f['conv_2_1']['conv_2_1_b']), np.array(f['conv_2_2']['conv_2_2_b'])])
                conv2B = tf.get_variable('conv2B', initializer=tf.constant(conv2B_values))

                # Conv3
                conv3W_values = np.transpose(np.array(f['conv_3']['conv_3_W']))
                conv3W = tf.get_variable('conv3W', initializer=tf.constant(conv3W_values))
                conv3B_values = np.array(f['conv_3']['conv_3_b'])
                conv3B = tf.get_variable('conv3B', initializer=tf.constant(conv3B_values))

                # Conv4
                conv4W_values = np.concatenate(
                    [
                        np.concatenate(
                            [np.transpose(np.array(f['conv_4_1']['conv_4_1_W'])),
                             np.transpose(np.array(f['conv_4_1']['conv_4_1_W']))], axis=2),
                        np.concatenate(
                            [np.transpose(np.array(f['conv_4_2']['conv_4_2_W'])),
                             np.transpose(np.array(f['conv_4_2']['conv_4_2_W']))], axis=2)
                    ], axis=3)
                conv4W = tf.get_variable('conv4W', initializer=tf.constant(conv4W_values))
                conv4B_values = np.concatenate([np.array(f['conv_4_1']['conv_4_1_b']), np.array(f['conv_4_2']['conv_4_2_b'])])
                conv4B = tf.get_variable('conv4B', initializer=tf.constant(conv4B_values))

                # Conv5
                conv5W_values = np.concatenate(
                    [
                        np.concatenate(
                            [np.transpose(np.array(f['conv_5_1']['conv_5_1_W'])),
                             np.transpose(np.array(f['conv_5_1']['conv_5_1_W']))], axis=2),
                        np.concatenate(
                            [np.transpose(np.array(f['conv_5_2']['conv_5_2_W'])),
                             np.transpose(np.array(f['conv_5_2']['conv_5_2_W']))], axis=2)
                    ], axis=3)
                conv5W = tf.get_variable('conv5W', initializer=tf.constant(conv5W_values))
                conv5B_values = np.concatenate([np.array(f['conv_5_1']['conv_5_1_b']), np.array(f['conv_5_2']['conv_5_2_b'])])
                conv5B = tf.get_variable('conv5B', initializer=tf.constant(conv5B_values))

        # Build graph
        with tf.variable_scope('conv1'):
            model_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, conv1W, [1, 4, 4, 1], 'VALID'), conv1B))
        with tf.variable_scope('max_pool1'):
            model_out = tf.nn.max_pool(model_out, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
        with tf.variable_scope('conv2'):
            model_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_out, conv2W, [1, 1, 1, 1], 'SAME'), conv2B))
        with tf.variable_scope('max_pool2'):
            model_out = tf.nn.max_pool(model_out, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
        with tf.variable_scope('conv3'):
            model_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_out, conv3W, [1, 1, 1, 1], 'SAME'), conv3B))
        with tf.variable_scope('conv4'):
            model_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_out, conv4W, [1, 1, 1, 1], 'SAME'), conv4B))
        with tf.variable_scope('conv5'):
            model_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_out, conv5W, [1, 1, 1, 1], 'SAME'), conv5B))
        with tf.variable_scope('max_pool3'):
            model_out = tf.nn.max_pool(model_out, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')

        # Flatten
        model_out = tf.contrib.layers.flatten(model_out)

        # Dropout
        model_out = tf.nn.dropout(model_out, 0.5)

        # Fc1
        tf.get_variable('fc1/weights', initializer=tf.constant(np.array(f['dense_1']['dense_1_W'])))
        tf.get_variable('fc1/biases', initializer=tf.constant(np.array(f['dense_1']['dense_1_b'])))
        model_out = tf.contrib.layers.fully_connected(model_out, 4096, reuse=True, scope='fc1')

        # Dropout
        model_out = tf.nn.dropout(model_out, 0.5)

        # Fc2
        tf.get_variable('fc2/weights', initializer=tf.constant(np.array(f['dense_2']['dense_2_W'])))
        tf.get_variable('fc2/biases', initializer=tf.constant(np.array(f['dense_2']['dense_2_b'])))
        model_out = tf.contrib.layers.fully_connected(model_out, 4096, reuse=True, scope='fc2')

        # Fc3
        model_out = tf.contrib.layers.fully_connected(model_out, 7, reuse=reuse, scope='fc3', activation_fn=None)

    # Inference
    with tf.variable_scope('inference'):
        inference = tf.identity(tf.nn.softmax(model_out), name='inference')

    # Loss
    with tf.variable_scope('loss'):
        tf.losses.softmax_cross_entropy(y, model_out)

    # Accuracy
    with tf.variable_scope('accuracy'):
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(inference, 1)), tf.float32), name='acc')
        tf.add_to_collection('accuracy_collection', acc)

    # Tensorboard
    if reuse is False:
        with tf.name_scope('summaries'):
            tf.summary.histogram('conv1W', conv1W)
            tf.summary.histogram('conv1B', conv1B)
            tf.summary.histogram('conv2W', conv2W)
            tf.summary.histogram('conv2B', conv2B)
            tf.summary.histogram('conv3W', conv3W)
            tf.summary.histogram('conv3B', conv3B)
            tf.summary.histogram('conv4W', conv4W)
            tf.summary.histogram('conv4B', conv4B)
            tf.summary.histogram('conv5W', conv5W)
            tf.summary.histogram('conv5B', conv5B)
