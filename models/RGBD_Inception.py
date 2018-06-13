import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import init_ops
from tensorflow.contrib.slim.nets import inception

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def assign_variable_values(sess):
    saver = tf.train.Saver({tf.get_variable('model/rgb_model/InceptionV3/Conv2d_1a_3x3/weights'): 'InceptionV3/Conv2d_1a_3x3/weights'})
    saver.restore(sess, 'models/inception_v3.ckpt')


def get_model_name():
    return 'RGBD_Inception2'


def build_model(rgb_x, depth_x, y, reuse=False):

    name_scope = tf.contrib.framework.get_name_scope()

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
                rgb_out = tf.image.resize_images(rgb_x, (299, 299))
                with tf.contrib.slim.arg_scope(inception.inception_v3_arg_scope()):
                    inception.inception_v3(rgb_out)
                    rgb_aux_logits = tf.get_default_graph().get_tensor_by_name(name_scope + '/model/rgb_model/rgb_inception/InceptionV3/AuxLogits/Conv2d_2a_5x5/Relu:0')
                    rgb_aux_logits = layers.conv2d(rgb_aux_logits, 7, [1, 1], activation_fn=None, normalizer_fn=None,
                                                   weights_initializer=trunc_normal(0.001), scope='Conv2d_2b_1x1')
                    rgb_aux_logits = tf.squeeze(rgb_aux_logits)

        with tf.variable_scope('depth_model'):
            # Depth model
            with tf.variable_scope('conv1'):
                depth_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(depth_x, conv1W, [1, 1, 1, 1], 'SAME'), conv1B))
            with tf.variable_scope('conv2'):
                depth_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(depth_out, conv2W, [1, 1, 1, 1], 'SAME'), conv2B))
            with tf.variable_scope('conv3'):
                depth_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(depth_out, conv3W, [1, 1, 1, 1], 'SAME'), conv3B))
            with tf.variable_scope('depth_inception'):
                depth_out = tf.image.resize_images(depth_out, (299, 299))
                with tf.contrib.slim.arg_scope(inception.inception_v3_arg_scope()):
                    inception.inception_v3(depth_out)
                    depth_aux_logits = tf.get_default_graph().get_tensor_by_name(name_scope + '/model/depth_model/depth_inception/InceptionV3/AuxLogits/Conv2d_2a_5x5/Relu:0')
                    depth_aux_logits = layers.conv2d(depth_aux_logits, 7, [1, 1], activation_fn=None, normalizer_fn=None,
                                                     weights_initializer=trunc_normal(0.001), scope='Conv2d_2b_1x1')
                    depth_aux_logits = tf.squeeze(depth_aux_logits)

        with tf.variable_scope('fused_models'):
            rgb_logits = tf.get_default_graph().get_tensor_by_name(name_scope + '/model/rgb_model/rgb_inception/InceptionV3/Logits/Dropout_1b/dropout/mul:0')
            depth_logits = tf.get_default_graph().get_tensor_by_name(name_scope + '/model/depth_model/depth_inception/InceptionV3/Logits/Dropout_1b/dropout/mul:0')
            model_logits = tf.concat([rgb_logits, depth_logits], axis=-1)
            model_logits = tf.nn.dropout(model_logits, 0.5)
            model_logits = layers.conv2d(model_logits, 4096, [1, 1], weights_initializer=trunc_normal(0.001), scope='fc1')
            model_logits = tf.nn.dropout(model_logits, 0.5)
            model_logits = layers.conv2d(model_logits, 4096, [1, 1], weights_initializer=trunc_normal(0.001), scope='fc2')
            model_logits = layers.conv2d(model_logits, 7, [1, 1], activation_fn=None, normalizer_fn=None,
                                         weights_initializer=trunc_normal(0.001), scope='fc3')
            model_logits = tf.squeeze(model_logits)
            model_logits = tf.reshape(model_logits, [-1, 7])

    # Inference
    with tf.variable_scope('inference'):
        inference = tf.identity(tf.nn.softmax(model_logits), name='inference')

    # Loss
    with tf.variable_scope('loss'):
        tf.losses.softmax_cross_entropy(y, rgb_aux_logits)
        tf.losses.softmax_cross_entropy(y, depth_aux_logits)
        tf.losses.softmax_cross_entropy(y, model_logits)

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

