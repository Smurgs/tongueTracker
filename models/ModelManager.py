import os
import shutil
import tensorflow as tf
from tensorflow.python.client import device_lib

from models.RGBD_AlexNet import *


class ModelManager(object):

    def __init__(self, sess):
        self.sess = sess
        self.save_dir = 'logs/'
        self.max_saves = 5

    def inference_op(self):
        return tf.get_default_graph().get_tensor_by_name('tower_0/inference/inference:0')

    def train_op(self):
        return tf.get_default_graph().get_tensor_by_name('train:0')

    def avg_acc(self):
        return tf.reduce_mean(tf.stack(tf.get_collection('accuracy_collection')))

    def inference_accuracy(self):
        return tf.get_default_graph().get_tensor_by_name('tower_0/accuracy/acc:0')

    def avg_loss(self):
        return tf.reduce_mean(tf.stack(tf.losses.get_losses()))

    def learning_rate(self):
        return tf.get_default_graph().get_tensor_by_name('learning_rate:0')

    def model_name(self):
        return get_model_name()

    def dataset_init(self, rgb_paths, depth_paths, states, batch_size=32):
        init_op = tf.get_default_graph().get_operation_by_name('dataset_init')
        rgb_placeholder = tf.get_default_graph().get_tensor_by_name('rgb_placeholder:0')
        depth_placeholder = tf.get_default_graph().get_tensor_by_name('depth_placeholder:0')
        state_placeholder = tf.get_default_graph().get_tensor_by_name('state_placeholder:0')
        batch_placeholder = tf.get_default_graph().get_tensor_by_name('batch_size_placeholder:0')

        self.sess.run(init_op, feed_dict={rgb_placeholder: rgb_paths,
                                          depth_placeholder: depth_paths,
                                          state_placeholder: states,
                                          batch_placeholder: batch_size})

    def create_dataset(self):
        rgb_placeholder = tf.placeholder(tf.string, shape=[None], name='rgb_placeholder')
        depth_placeholder = tf.placeholder(tf.string, shape=[None], name='depth_placeholder')
        state_placeholder = tf.placeholder(tf.int32, [None], name='state_placeholder')
        batch_size_placeholder = tf.placeholder(tf.int64, name='batch_size_placeholder')
        dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(rgb_placeholder),
                                                      tf.convert_to_tensor(depth_placeholder),
                                                      tf.convert_to_tensor(state_placeholder)))

        def parse_function(rgb_path, depth_path, state):
            rgb_string = tf.read_file(rgb_path)
            rgb_img = tf.image.decode_png(rgb_string, channels=3, dtype=tf.uint16)
            rgb_img = tf.cast(rgb_img, tf.float32)
            rgb_img = tf.reshape(rgb_img, [227, 227, 3])

            depth_string = tf.read_file(depth_path)
            depth_img = tf.image.decode_png(depth_string, channels=1, dtype=tf.uint16)
            depth_img = tf.cast(depth_img, tf.float32)
            depth_img = tf.reshape(depth_img, [227, 227, 1])

            label = tf.one_hot(state, 7)
            return rgb_img, depth_img, label

        dataset = dataset.map(parse_function).batch(batch_size_placeholder)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        iterator.make_initializer(dataset, name='dataset_init')
        return iterator

    def prepare_graph(self):
        # Check for savedModel of the current model
        if os.path.isdir(self.save_dir + get_model_name()):
            # Load latest save
            saves = os.listdir("%s/%s" % (self.save_dir, get_model_name()))
            saves = [x for x in saves if 'events' not in x]
            if len(saves) > 0:
                saves = [(x, int(x.split('-')[-1])) for x in saves]
                saves.sort(key=lambda tup: tup[1])
                latest_save = ('%s%s/%s' % (self.save_dir, get_model_name(), saves[-1][0]))
                print('Loading model: %s' % latest_save)
                tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], latest_save)
                return

        print('Building model')

        # Build dataset and get iterator
        print('Creating dataset iterator')
        dataset_iterator = self.create_dataset()

        # Get device list
        print('Getting device list')
        local_device_protos = device_lib.list_local_devices()
        devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
        devices = ['/gpu:%d' % x for x in range(len(devices))]
        if len(devices) < 1:
            devices.append('/cpu:0')
        print('Devices found: ' + str(devices))

        # Build model on each available device
        for x in range(len(devices)):
            print('Building tower %d' % x)
            with tf.device(devices[x]):
                with tf.name_scope('tower_%d' % x):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=x!=0):
                        rgb_x, depth_x, y = dataset_iterator.get_next()
                        tf.identity(rgb_x, name='rgb_x')
                        tf.identity(depth_x, name='depth_x')

                        # Build instance of model
                        build_model(rgb_x, depth_x, y, x!=0)

        # Average gradients from all devices
        print('Building train op')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        with tf.name_scope('average_gradients'):
            average_gradients = []
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            losses = tf.losses.get_losses()
            grads = []
            for loss in losses:
                grad = optimizer.compute_gradients(loss)
                grads.append(grad)

            for grad_and_vars in zip(*grads):
                grads = []
                for g, _ in grad_and_vars:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_gradients.append(grad_and_var)

        # Overall train op
        global_step = tf.train.create_global_step()
        optimizer.apply_gradients(average_gradients, global_step=global_step, name='train')

        print('Initializing variables')
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # Setup Tensorboard summary writers
        with tf.name_scope('summaries'):
            tf.summary.scalar('avg_loss', self.avg_loss())
            tf.summary.scalar('avg_acc', self.avg_acc())
            tf.summary.scalar('learning_rate', learning_rate)

    def save(self):
        # Check if this iteration has already been saved
        saves = os.listdir("%s/%s" % (self.save_dir, get_model_name()))
        saves = [x for x in saves if 'events' not in x]
        if len(saves) > 0:
            save_numbers = [int(x.split('-')[-1]) for x in saves]
            if tf.train.global_step(self.sess, tf.train.get_global_step()) in save_numbers:
                return

        # Save model
        rgb_x = tf.get_default_graph().get_tensor_by_name('tower_0/rgb_x:0')
        depth_x = tf.get_default_graph().get_tensor_by_name('tower_0/depth_x:0')
        inference = self.inference_op()
        save_path = "%s%s/%s-%d" % (self.save_dir,
                                    get_model_name(),
                                    get_model_name(),
                                    tf.train.global_step(self.sess, tf.train.get_global_step()))
        tf.saved_model.simple_save(self.sess, save_path, {'rgb_x': rgb_x, 'depth_x': depth_x}, {'inference': inference})
        print('Saved model: %s' % save_path)

        # Delete oldest save if max number of saves reached
        if len(saves) > self.max_saves:
            saves = [(x, int(x.split('-')[-1])) for x in saves]
            saves.sort(key=lambda tup: tup[1])
            shutil.rmtree("%s%s/%s" % (self.save_dir, get_model_name(), saves[0][0]))
