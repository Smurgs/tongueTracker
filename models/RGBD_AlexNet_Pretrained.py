import h5py
import numpy as np
import tensorflow as tf


def assign_variable_values(sess):
    # Load pretrained variables
    f = h5py.File('models/alexnet_weights.h5', 'r')

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        # Conv1
        conv1W_tensor = tf.get_variable('model/conv_variables/conv1W')
        conv1W_values = sess.run(conv1W_tensor)
        conv1W_values[:, :, :3, :] = np.transpose(np.array(f['conv_1']['conv_1_W']))
        conv1W_tensor.load(conv1W_values, session=sess)
        conv1B_tensor = tf.get_variable('model/conv_variables/conv1B')
        conv1B_tensor.load(np.array(f['conv_1']['conv_1_b']), session=sess)

        # Conv2
        conv2W_tensor = tf.get_variable('model/conv_variables/conv2W')
        conv2W_values = np.concatenate(
            [
                np.concatenate(
                    [np.transpose(np.array(f['conv_2_1']['conv_2_1_W'])),
                     np.transpose(np.array(f['conv_2_1']['conv_2_1_W']))], axis=2),
                np.concatenate(
                    [np.transpose(np.array(f['conv_2_2']['conv_2_2_W'])),
                     np.transpose(np.array(f['conv_2_2']['conv_2_2_W']))], axis=2)
            ], axis=3)
        conv2W_tensor.load(conv2W_values, session=sess)
        conv2B_tensor = tf.get_variable('model/conv_variables/conv2B')
        conv2B_values = np.concatenate([np.array(f['conv_2_1']['conv_2_1_b']), np.array(f['conv_2_2']['conv_2_2_b'])])
        conv2B_tensor.load(conv2B_values, session=sess)

        # Conv3
        conv3W_tensor = tf.get_variable('model/conv_variables/conv3W')
        conv3W_values = np.transpose(np.array(f['conv_3']['conv_3_W']))
        conv3W_tensor.load(conv3W_values, session=sess)
        conv3B_tensor = tf.get_variable('model/conv_variables/conv3B')
        conv3B_tensor.load(np.array(f['conv_3']['conv_3_b']), session=sess)

        # Conv4
        conv4W_tensor = tf.get_variable('model/conv_variables/conv4W')
        conv4W_values = np.concatenate(
            [
                np.concatenate(
                    [np.transpose(np.array(f['conv_4_1']['conv_4_1_W'])),
                     np.transpose(np.array(f['conv_4_1']['conv_4_1_W']))], axis=2),
                np.concatenate(
                    [np.transpose(np.array(f['conv_4_2']['conv_4_2_W'])),
                     np.transpose(np.array(f['conv_4_2']['conv_4_2_W']))], axis=2)
            ], axis=3)
        conv4W_tensor.load(conv4W_values, session=sess)
        conv4B_tensor = tf.get_variable('model/conv_variables/conv4B')
        conv4B_values = np.concatenate([np.array(f['conv_4_1']['conv_4_1_b']), np.array(f['conv_4_2']['conv_4_2_b'])])
        conv4B_tensor.load(conv4B_values, session=sess)

        # Conv5
        conv5W_tensor = tf.get_variable('model/conv_variables/conv5W')
        conv5W_values = np.concatenate(
            [
                np.concatenate(
                    [np.transpose(np.array(f['conv_5_1']['conv_5_1_W'])),
                     np.transpose(np.array(f['conv_5_1']['conv_5_1_W']))], axis=2),
                np.concatenate(
                    [np.transpose(np.array(f['conv_5_2']['conv_5_2_W'])),
                     np.transpose(np.array(f['conv_5_2']['conv_5_2_W']))], axis=2)
            ], axis=3)
        conv5W_tensor.load(conv5W_values, session=sess)
        conv5B_tensor = tf.get_variable('model/conv_variables/conv5B')
        conv5B_values = np.concatenate([np.array(f['conv_5_1']['conv_5_1_b']), np.array(f['conv_5_2']['conv_5_2_b'])])
        conv5B_tensor.load(conv5B_values, session=sess)

        # Fc1
        fc1W_tensor = tf.get_variable('model/fc1/weights')
        fc1W_tensor.load(np.array(f['dense_1']['dense_1_W']), session=sess)
        fc1B_tensor = tf.get_variable('model/fc1/biases')
        fc1B_tensor.load(np.array(f['dense_1']['dense_1_b']), session=sess)

        # Fc2
        fc2W_tensor = tf.get_variable('model/fc2/weights')
        fc2W_tensor.load(np.array(f['dense_2']['dense_2_W']), session=sess)
        fc2B_tensor = tf.get_variable('model/fc2/biases')
        fc2B_tensor.load(np.array(f['dense_2']['dense_2_b']), session=sess)


def get_model_name():
    return 'RGBD_AlexNet_Pretrained'


def build_model(rgb_x, depth_x, y, reuse=False):

    # Combine rgb and depth data
    x = tf.concat([rgb_x, depth_x], axis=-1)

    # Create variables
    with tf.variable_scope('model'):
        with tf.variable_scope('conv_variables'):
            with tf.device('/cpu:0'):
                conv1W = tf.get_variable('conv1W', shape=[11, 11, 4, 96], initializer=tf.contrib.layers.xavier_initializer())
                conv1B = tf.get_variable('conv1B', shape=[96], initializer=tf.zeros_initializer())
                conv2W = tf.get_variable('conv2W', shape=[5, 5, 96, 256], initializer=tf.contrib.layers.xavier_initializer())
                conv2B = tf.get_variable('conv2B', shape=[256], initializer=tf.zeros_initializer())
                conv3W = tf.get_variable('conv3W', shape=[3, 3, 256, 384], initializer=tf.contrib.layers.xavier_initializer())
                conv3B = tf.get_variable('conv3B', shape=[384], initializer=tf.zeros_initializer())
                conv4W = tf.get_variable('conv4W', shape=[3, 3, 384, 384], initializer=tf.contrib.layers.xavier_initializer())
                conv4B = tf.get_variable('conv4B', shape=[384], initializer=tf.zeros_initializer())
                conv5W = tf.get_variable('conv5W', shape=[3, 3, 384, 256], initializer=tf.contrib.layers.xavier_initializer())
                conv5B = tf.get_variable('conv5B', shape=[256], initializer=tf.zeros_initializer())

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
        model_out = tf.contrib.layers.flatten(model_out)
        model_out = tf.nn.dropout(model_out, 0.5)
        model_out = tf.contrib.layers.fully_connected(model_out, 4096, reuse=reuse, scope='fc1')
        model_out = tf.nn.dropout(model_out, 0.5)
        model_out = tf.contrib.layers.fully_connected(model_out, 4096, reuse=reuse, scope='fc2')
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

