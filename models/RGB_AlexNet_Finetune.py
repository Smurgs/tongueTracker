import h5py
import numpy as np
import tensorflow as tf


def get_learning_rate(): return 0.1


def get_batch_size(): return 32


def get_train_vars(): return None


def get_depth_channels(): return 1


def assign_variable_values(sess): pass


def get_model_name(): return 'RGB_AlexNet_Finetune'


def build_model(rgb_x, depth_x, y, batch_size, reuse, training_ph, outputs):

    # Load pretrained variables
    f = h5py.File('models/alexnet_weights.h5', 'r')

    # Create variables
    with tf.variable_scope('model'):
        with tf.variable_scope('conv_variables'):
            with tf.device('/cpu:0'):
                # Conv1
                conv1W_values = np.array(f['conv_1']['conv_1_W']).transpose()
                conv1W = tf.get_variable('conv1Wa', initializer=tf.constant(conv1W_values))
                conv1B_values = np.array(f['conv_1']['conv_1_b'])
                conv1B = tf.get_variable('conv1B', initializer=tf.constant(conv1B_values))

                # Conv2
                conv2W_1_values = np.array(f['conv_2_1']['conv_2_1_W']).transpose()
                conv2W_2_values = np.array(f['conv_2_2']['conv_2_2_W']).transpose()
                conv2W_1 = tf.get_variable('conv2W_1', initializer=tf.constant(conv2W_1_values))
                conv2W_2 = tf.get_variable('conv2W_2', initializer=tf.constant(conv2W_2_values))
                conv2B_1_values = np.array(f['conv_2_1']['conv_2_1_b'])
                conv2B_2_values = np.array(f['conv_2_2']['conv_2_2_b'])
                conv2B_1 = tf.get_variable('conv2B_1', initializer=tf.constant(conv2B_1_values))
                conv2B_2 = tf.get_variable('conv2B_2', initializer=tf.constant(conv2B_2_values))

                # Conv3
                conv3W_values = np.array(f['conv_3']['conv_3_W']).transpose()
                conv3W = tf.get_variable('conv3W', initializer=tf.constant(conv3W_values))
                conv3B_values = np.array(f['conv_3']['conv_3_b'])
                conv3B = tf.get_variable('conv3B', initializer=tf.constant(conv3B_values))

                # Conv4
                conv4W_1_values = np.array(f['conv_4_1']['conv_4_1_W']).transpose()
                conv4W_2_values = np.array(f['conv_4_2']['conv_4_2_W']).transpose()
                conv4W_1 = tf.get_variable('conv4W_1', initializer=tf.constant(conv4W_1_values))
                conv4W_2 = tf.get_variable('conv4W_2', initializer=tf.constant(conv4W_2_values))
                conv4B_1_values = np.array(f['conv_4_1']['conv_4_1_b'])
                conv4B_2_values = np.array(f['conv_4_2']['conv_4_2_b'])
                conv4B_1 = tf.get_variable('conv4B_1', initializer=tf.constant(conv4B_1_values))
                conv4B_2 = tf.get_variable('conv4B_2', initializer=tf.constant(conv4B_2_values))

                # Conv5
                conv5W_1_values = np.array(f['conv_5_1']['conv_5_1_W']).transpose()
                conv5W_2_values = np.array(f['conv_5_2']['conv_5_2_W']).transpose()
                conv5W_1 = tf.get_variable('conv5W_1', initializer=tf.constant(conv5W_1_values))
                conv5W_2 = tf.get_variable('conv5W_2', initializer=tf.constant(conv5W_2_values))
                conv5B_1_values = np.array(f['conv_5_1']['conv_5_1_b'])
                conv5B_2_values = np.array(f['conv_5_2']['conv_5_2_b'])
                conv5B_1 = tf.get_variable('conv5B_1', initializer=tf.constant(conv5B_1_values))
                conv5B_2 = tf.get_variable('conv5B_2', initializer=tf.constant(conv5B_2_values))

        # Build graph
        with tf.variable_scope('conv1'):
            model_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(rgb_x, conv1W, [1, 4, 4, 1], 'VALID'), conv1B))
        with tf.variable_scope('max_pool1'):
            model_out = tf.nn.max_pool(model_out, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
        with tf.variable_scope('conv2'):
            model_1, model_2 = tf.split(model_out, 2, axis=-1)
            model_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_1, conv2W_1, [1, 1, 1, 1], 'SAME'), conv2B_1))
            model_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_2, conv2W_2, [1, 1, 1, 1], 'SAME'), conv2B_2))
        with tf.variable_scope('max_pool2'):
            model_out = tf.concat([model_1, model_2], axis=-1)
            model_out = tf.nn.max_pool(model_out, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
        with tf.variable_scope('conv3'):
            model_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_out, conv3W, [1, 1, 1, 1], 'SAME'), conv3B))
        with tf.variable_scope('conv4'):
            model_1, model_2 = tf.split(model_out, 2, axis=-1)
            model_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_1, conv4W_1, [1, 1, 1, 1], 'SAME'), conv4B_1))
            model_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_2, conv4W_2, [1, 1, 1, 1], 'SAME'), conv4B_2))
        with tf.variable_scope('conv5'):
            model_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_1, conv5W_1, [1, 1, 1, 1], 'SAME'), conv5B_1))
            model_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_2, conv5W_2, [1, 1, 1, 1], 'SAME'), conv5B_2))
        with tf.variable_scope('max_pool3'):
            model_out = tf.concat([model_1, model_2], axis=-1)
            model_out = tf.nn.max_pool(model_out, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')

        # Flatten
        model_out = tf.contrib.layers.flatten(model_out)

        # Dropout
        model_out = tf.layers.dropout(model_out, 0.5, training=training_ph)

        # Fc1
        tf.get_variable('fc1/weights', initializer=tf.constant(np.array(f['dense_1']['dense_1_W'])))
        tf.get_variable('fc1/biases', initializer=tf.constant(np.array(f['dense_1']['dense_1_b'])))
        model_out = tf.contrib.layers.fully_connected(model_out, 4096, reuse=True, scope='fc1')

        # Dropout
        model_out = tf.layers.dropout(model_out, 0.5, training=training_ph)

        # Fc2
        tf.get_variable('fc2/weights', initializer=tf.constant(np.array(f['dense_2']['dense_2_W'])))
        tf.get_variable('fc2/biases', initializer=tf.constant(np.array(f['dense_2']['dense_2_b'])))
        model_out = tf.contrib.layers.fully_connected(model_out, 4096, reuse=True, scope='fc2')

        # Fc3
        model_out = tf.contrib.layers.fully_connected(model_out, outputs, reuse=reuse, scope='fc3', activation_fn=None)

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
    #if reuse is False:
    #    with tf.name_scope('summaries'):
    #        tf.summary.histogram('conv1W', conv1W)
    #        tf.summary.histogram('conv1B', conv1B)
    #        tf.summary.histogram('conv2W', conv2W_1)
    #        tf.summary.histogram('conv2W', conv2W_2)
    #        tf.summary.histogram('conv2B', conv2B_1)
    #        tf.summary.histogram('conv2B', conv2B_2)
    #        tf.summary.histogram('conv3W', conv3W)
    #        tf.summary.histogram('conv3B', conv3B)
    #        tf.summary.histogram('conv4W', conv4W_1)
    #        tf.summary.histogram('conv4W', conv4W_2)
    #        tf.summary.histogram('conv4B', conv4B_1)
    #        tf.summary.histogram('conv4B', conv4B_2)
    #        tf.summary.histogram('conv5W', conv5W_1)
    #        tf.summary.histogram('conv5W', conv5W_2)
    #        tf.summary.histogram('conv5B', conv5B_1)
    #        tf.summary.histogram('conv5B', conv5B_2)
