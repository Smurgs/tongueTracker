import tensorflow as tf


def get_learning_rate(): return 0.1


def get_batch_size(): return 32


def get_train_vars(): return None


def get_depth_channels(): return 1


def assign_variable_values(sess): pass


def get_model_name(): return 'RGBD_AlexNet2h'


def build_model(rgb_x, depth_x, y, batch_size, reuse, training_ph, outputs):

    # Combine rgb and depth data
    x = tf.concat([rgb_x, depth_x], axis=-1)

    # Create variables
    with tf.variable_scope('model'):
        with tf.variable_scope('conv_variables'):
            with tf.device('/cpu:0'):
                reg = tf.contrib.layers.l2_regularizer(scale=0.15)
                conv1W = tf.get_variable('conv1W', shape=[11, 11, 4, 96], initializer=tf.contrib.layers.xavier_initializer(), regularizer=reg)
                conv1B = tf.get_variable('conv1B', shape=[96], initializer=tf.zeros_initializer())
                conv2W = tf.get_variable('conv2W', shape=[5, 5, 96, 256], initializer=tf.contrib.layers.xavier_initializer(), regularizer=reg)
                conv2B = tf.get_variable('conv2B', shape=[256], initializer=tf.zeros_initializer())
                conv3W = tf.get_variable('conv3W', shape=[3, 3, 256, 384], initializer=tf.contrib.layers.xavier_initializer(), regularizer=reg)
                conv3B = tf.get_variable('conv3B', shape=[384], initializer=tf.zeros_initializer())
                conv4W = tf.get_variable('conv4W', shape=[3, 3, 384, 384], initializer=tf.contrib.layers.xavier_initializer(), regularizer=reg)
                conv4B = tf.get_variable('conv4B', shape=[384], initializer=tf.zeros_initializer())
                conv5W = tf.get_variable('conv5W', shape=[3, 3, 384, 256], initializer=tf.contrib.layers.xavier_initializer(), regularizer=reg)
                conv5B = tf.get_variable('conv5B', shape=[256], initializer=tf.zeros_initializer())

        # Build graph
        with tf.variable_scope('conv1'):
            model_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, conv1W, [1, 4, 4, 1], 'VALID'), conv1B))
        with tf.variable_scope('max_pool1'):
            model_out = tf.nn.max_pool(model_out, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
            model_out = tf.layers.dropout(model_out, 0.1, training=training_ph)
        with tf.variable_scope('conv2'):
            model_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_out, conv2W, [1, 1, 1, 1], 'SAME'), conv2B))
        with tf.variable_scope('max_pool2'):
            model_out = tf.nn.max_pool(model_out, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
            model_out = tf.layers.dropout(model_out, 0.1, training=training_ph)
        with tf.variable_scope('conv3'):
            model_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_out, conv3W, [1, 1, 1, 1], 'SAME'), conv3B))
            model_out = tf.layers.dropout(model_out, 0.1, training=training_ph)
        with tf.variable_scope('conv4'):
            model_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_out, conv4W, [1, 1, 1, 1], 'SAME'), conv4B))
            model_out = tf.layers.dropout(model_out, 0.1, training=training_ph)
        with tf.variable_scope('conv5'):
            model_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model_out, conv5W, [1, 1, 1, 1], 'SAME'), conv5B))
        with tf.variable_scope('max_pool3'):
            model_out = tf.nn.max_pool(model_out, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
        model_out = tf.contrib.layers.flatten(model_out)
        model_out = tf.layers.dropout(model_out, 0.5, training=training_ph)
        model_out = tf.contrib.layers.fully_connected(model_out, 4096, reuse=reuse, scope='fc1', weights_regularizer=reg)
        model_out = tf.layers.dropout(model_out, 0.5, training=training_ph)
        model_out = tf.contrib.layers.fully_connected(model_out, 4096, reuse=reuse, scope='fc2', weights_regularizer=reg)
        model_out = tf.layers.dropout(model_out, 0.5, training=training_ph)
        model_out = tf.contrib.layers.fully_connected(model_out, 7, reuse=reuse, scope='fc3', activation_fn=None, weights_regularizer=reg)

    # Inference
    with tf.variable_scope('inference'):
        inference = tf.identity(tf.nn.softmax(model_out), name='inference')

    # Loss
    with tf.variable_scope('loss'):
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_out, labels=y))
        loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.losses.add_loss(loss)
        #tf.losses.softmax_cross_entropy(y, model_out)

    # Accuracy
    with tf.variable_scope('accuracy'):
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(inference, 1)), tf.float32), name='acc')
        tf.add_to_collection('accuracy_collection', acc)

    # Tensorboard
    #if reuse is False:
    #    with tf.name_scope('summaries'):
    #        tf.summary.histogram('conv1W', conv1W)
    #        tf.summary.histogram('conv1B', conv1B)
    #        tf.summary.histogram('conv2W', conv2W)
    #        tf.summary.histogram('conv2B', conv2B)
    #        tf.summary.histogram('conv3W', conv3W)
    #        tf.summary.histogram('conv3B', conv3B)
    #        tf.summary.histogram('conv4W', conv4W)
    #        tf.summary.histogram('conv4B', conv4B)
    #        tf.summary.histogram('conv5W', conv5W)
    #        tf.summary.histogram('conv5B', conv5B)

