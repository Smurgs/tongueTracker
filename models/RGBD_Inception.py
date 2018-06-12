import tensorflow as tf


def assign_variable_values(sess):
    pass


def get_model_name():
    return 'RGBD_Inception'


def build_model(rgb_x, depth_x, y, reuse=False):

    # Create variables
    with tf.variable_scope('model'):
        with tf.variable_scope('conv_variables'):
            with tf.device('/cpu:0'):
                conv1W = tf.get_variable('conv1W', shape=[11, 11, 1, 96], initializer=tf.contrib.layers.xavier_initializer())
                conv1B = tf.get_variable('conv1B', shape=[96], initializer=tf.zeros_initializer())
                conv2W = tf.get_variable('conv2W', shape=[5, 5, 96, 192], initializer=tf.contrib.layers.xavier_initializer())
                conv2B = tf.get_variable('conv2B', shape=[192], initializer=tf.zeros_initializer())
                conv3W = tf.get_variable('conv3W', shape=[3, 3, 192, 3], initializer=tf.contrib.layers.xavier_initializer())
                conv3B = tf.get_variable('conv3B', shape=[3], initializer=tf.zeros_initializer())

        # Build graph
        with tf.variable_scope('rgb_model'):
            # RGB model
            with tf.variable_scope('rgb_inception'):
                rgb_out = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_tensor=rgb_x,
                                                            input_shape=(227, 227, 3), pooling='max')
                rgb_out.outputs

        with tf.variable_scope('depth_model'):
            # Depth model
            with tf.variable_scope('conv1'):
                depth_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(depth_x, conv1W, [1, 1, 1, 1], 'SAME'), conv1B))
            with tf.variable_scope('conv2'):
                depth_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(depth_out, conv2W, [1, 1, 1, 1], 'SAME'), conv2B))
            with tf.variable_scope('conv3'):
                depth_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(depth_out, conv3W, [1, 1, 1, 1], 'SAME'), conv3B))
            with tf.variable_scope('depth_inception'):
                depth_out = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_tensor=depth_out,
                                                              input_shape=(227, 227, 3), pooling='max')

        with tf.variable_scope('combined_models'):
            model_out = tf.concat([tf.contrib.layers.flatten(rgb_out), tf.contrib.layers.flatten(depth_out)], axis=-1)
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

