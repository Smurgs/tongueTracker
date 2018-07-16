import os
import json
import time
import shutil
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

import models.RGB_AlexNet
import models.RGB_AlexNet_Pretrained_Bvlc
import models.RGB_AlexNet_Finetune_Bvlc
import models.RGBD_AlexNet
import models.RGBD_AlexNet_Finetune_Bvlc
import models.RGB_Inception
import models.RGBD_Inception

states = ['mouth_closed', 'mouth_open', 'tongue_down', 'tongue_left', 'tongue_middle', 'tongue_right', 'tongue_up']

models = {'RGB_AlexNet': models.RGB_AlexNet,
          'RGB_AlexNet_Pretrained': models.RGB_AlexNet_Pretrained_Bvlc,
          'RGB_AlexNet_Finetune': models.RGB_AlexNet_Finetune_Bvlc,
          'RGBD_AlexNet': models.RGBD_AlexNet,
          'RGBD_AlexNet_Finetune': models.RGBD_AlexNet_Finetune_Bvlc,
          'RGB_Inception': models.RGB_Inception,
          'RGBD_Inception': models.RGBD_Inception}


class ModelManager(object):

    def __init__(self, model):
        # Get all configs
        self.model = models[model]
        with open('config') as f:
            data = json.load(f)
        self.train_epochs = data['train_epochs']
        self.dataset_parent_dir = data['dataset_parent_dir']
        self.dataset_dir = self.dataset_parent_dir + 'tongue_dataset/scaled/'
        self.dataset_size_limit = None if 'dataset_size_limit' not in data else data['dataset_size_limit']
        self.batch_size = self.model.get_batch_size() if 'batch_size' not in data else data['batch_size']
        self.train_annotations = data['train_annotations']
        self.val_annotations = data['val_annotations']
        self.test_annotations = data['test_annotations']
        self.save_dir = data['save_dir']
        self.max_saves = data['max_saves']

        # Prepare model
        session_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=session_config)
        self.prepare_graph()

        # Get handles to important tensors
        self.inference_op = tf.get_default_graph().get_tensor_by_name('tower_0/inference/inference:0')
        self.train_op = tf.get_default_graph().get_tensor_by_name('train:0')
        self.avg_acc_op = tf.reduce_mean(tf.stack(tf.get_collection('accuracy_collection')))
        self.inference_acc_op = tf.get_default_graph().get_tensor_by_name('tower_0/accuracy/acc:0')
        self.avg_loss_op = tf.reduce_mean(tf.stack(tf.losses.get_losses()))
        self.lr_ph = tf.get_default_graph().get_tensor_by_name('learning_rate:0')
        self.training_ph = tf.get_default_graph().get_tensor_by_name('training_ph:0')
        self.add_scalar_summaries()

    def dataset_init(self, rgb_paths, depth_paths, state_values):
        init_op = tf.get_default_graph().get_operation_by_name('dataset_init')
        rgb_ph = tf.get_default_graph().get_tensor_by_name('rgb_placeholder:0')
        depth_ph = tf.get_default_graph().get_tensor_by_name('depth_placeholder:0')
        state_ph = tf.get_default_graph().get_tensor_by_name('state_placeholder:0')
        batch_ph = tf.get_default_graph().get_tensor_by_name('batch_size_placeholder:0')
        self.sess.run(init_op, feed_dict={rgb_ph: rgb_paths, depth_ph: depth_paths, state_ph: state_values, batch_ph: self.batch_size})

    def create_dataset(self):
        rgb_ph = tf.placeholder(tf.string, shape=[None], name='rgb_placeholder')
        depth_ph = tf.placeholder(tf.string, shape=[None], name='depth_placeholder')
        state_ph = tf.placeholder(tf.int32, [None], name='state_placeholder')
        batch_size_ph = tf.placeholder(tf.int64, name='batch_size_placeholder')
        dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(rgb_ph), tf.convert_to_tensor(depth_ph), tf.convert_to_tensor(state_ph)))

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

        dataset = dataset.map(parse_function).batch(batch_size_ph)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        iterator.make_initializer(dataset, name='dataset_init')
        return iterator

    def prepare_graph(self):
        # Check for savedModel of the current model
        if os.path.isdir(self.save_dir + self.model.get_model_name()):
            # Load latest save
            saves = os.listdir("%s/%s" % (self.save_dir, self.model.get_model_name()))
            saves = [x for x in saves if 'events' not in x]
            if len(saves) > 0:
                saves = [(x, int(x.split('-')[-1])) for x in saves]
                saves.sort(key=lambda tup: tup[1])
                latest_save = ('%s%s/%s' % (self.save_dir, self.model.get_model_name(), saves[-1][0]))
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
        training_ph = tf.placeholder(tf.bool, name='training_ph')
        for x in range(len(devices)):
            print('Building tower %d' % x)
            with tf.device(devices[x]):
                with tf.name_scope('tower_%d' % x):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=x != 0):
                        rgb_x, depth_x, y = dataset_iterator.get_next()
                        tf.identity(rgb_x, name='rgb_x')
                        tf.identity(depth_x, name='depth_x')

                        # Build instance of model
                        self.model.build_model(rgb_x, depth_x, y, self.batch_size, x != 0, training_ph)

        # Average gradients from all devices
        print('Building train op')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        with tf.name_scope('average_gradients'):
            average_gradients = []
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            losses = tf.losses.get_losses()
            grads = [optimizer.compute_gradients(x, var_list=self.model.get_train_vars()) for x in losses]
            for grad_and_vars in zip(*grads):
                grads = []
                for g, a in grad_and_vars:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_gradients.append(grad_and_var)

        # Overall train op
        global_step = tf.train.create_global_step()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer.apply_gradients(average_gradients, global_step=global_step, name='train')
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        self.model.assign_variable_values(self.sess)

    def add_scalar_summaries(self):
        # Setup Tensorboard summary writers
        with tf.name_scope('summaries'):
            tf.summary.scalar('avg_loss', self.avg_loss_op)
            tf.summary.scalar('avg_acc', self.avg_acc_op)
            tf.summary.scalar('learning_rate', self.lr_ph)

    def save(self):
        # Check if this iteration has already been saved
        saves = os.listdir("%s/%s" % (self.save_dir, self.model.get_model_name()))
        saves = [x for x in saves if 'events' not in x]
        if len(saves) > 0:
            save_numbers = [int(x.split('-')[-1]) for x in saves]
            if tf.train.global_step(self.sess, tf.train.get_global_step()) in save_numbers:
                return

        # Save model
        rgb_x = tf.get_default_graph().get_tensor_by_name('tower_0/rgb_x:0')
        depth_x = tf.get_default_graph().get_tensor_by_name('tower_0/depth_x:0')
        inference = self.inference_op
        save_path = "%s%s/%s-%d" % (self.save_dir,
                                    self.model.get_model_name(),
                                    self.model.get_model_name(),
                                    tf.train.global_step(self.sess, tf.train.get_global_step()))
        tf.saved_model.simple_save(self.sess, save_path, {'rgb_x': rgb_x, 'depth_x': depth_x}, {'inference': inference})
        print('Saved model: %s' % save_path)

        # Delete oldest save if max number of saves reached
        if len(saves) >= self.max_saves:
            saves = [(x, int(x.split('-')[-1])) for x in saves]
            saves.sort(key=lambda tup: tup[1])
            shutil.rmtree("%s%s/%s" % (self.save_dir, self.model.get_model_name(), saves[0][0]))

    def feed_from_annotation(self, annotation_path):
        with open(annotation_path) as f:
            annotations = f.readlines()
        dataset_size = len(annotations) if self.dataset_size_limit is None else self.dataset_size_limit
        annotations = [x.strip().split(',') for x in annotations[:dataset_size]]
        rgb_path, depth_path, state, _ = zip(*annotations)
        rgb_path = [self.dataset_parent_dir + x[3:] for x in rgb_path]
        depth_path = [self.dataset_parent_dir + x[3:] for x in depth_path]
        state = [states.index(x) for x in state]
        return rgb_path, depth_path, state

    def add_static_summary(self, writer, identifier, value):
        summary = tf.Summary()
        summary.value.add(tag=identifier, simple_value=value)
        writer.add_summary(summary, tf.train.global_step(self.sess, tf.train.get_global_step()))

    def train(self):
        # Get train and validation dataset feeds
        train_rgb, train_depth, train_state = self.feed_from_annotation(self.dataset_dir + self.train_annotations)
        val_rgb, val_depth, val_state = self.feed_from_annotation(self.dataset_dir + self.val_annotations)

        # Setup Tensorboard stuff
        print('Setting up summary writers')
        merged_summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.save_dir + self.model.get_model_name() + '/events/train/', self.sess.graph)
        val_writer = tf.summary.FileWriter(self.save_dir + self.model.get_model_name() + '/events/validation/', self.sess.graph)

        # Train for a bunch of epochs
        print('Training model for %d epochs' % self.train_epochs)
        epoch_times = []
        for epoch in range(self.train_epochs):

            # Learn on training data for an epoch
            epoch_start_time = time.time()
            self.dataset_init(train_rgb, train_depth, train_state)
            while True:
                try:
                    _, summary = self.sess.run([self.train_op, merged_summaries],
                                               feed_dict={self.lr_ph: self.model.get_learning_rate(), self.training_ph: True})
                except tf.errors.OutOfRangeError:
                    train_writer.add_summary(summary, tf.train.global_step(self.sess, tf.train.get_global_step()))
                    break

            # Collect loss and acc on validation dataset
            self.dataset_init(val_rgb, val_depth, val_state)
            val_losses = []
            val_accs = []
            while True:
                try:
                    loss, acc = self.sess.run([self.avg_loss_op, self.avg_acc_op],
                                              feed_dict={self.lr_ph: 0, self.training_ph: False})
                    val_losses.append(loss)
                    val_accs.append(acc)
                except tf.errors.OutOfRangeError:
                    break

            # Write summaries and save model
            epoch_end_time = time.time() - epoch_start_time
            epoch_times.append(epoch_end_time)
            print('Done epoch # %d in %d seconds' % (epoch, epoch_end_time))
            self.add_static_summary(train_writer, 'summaries/epoch_time', epoch_end_time)
            self.add_static_summary(val_writer, 'summaries/avg_loss', np.mean(val_losses))
            self.add_static_summary(val_writer, 'summaries/avg_acc', np.mean(val_accs))
            self.save()
        print('Finished training %d epochs in %d seconds' % (self.train_epochs, int(np.sum(epoch_times))))

    def test(self):
        print('Running test')
        rgb, depth, state = self.feed_from_annotation(self.dataset_dir + self.test_annotations)
        self.dataset_init(rgb, depth, state)
        accs = []
        while True:
            try:
                accs.append(self.sess.run([self.avg_acc_op], feed_dict={self.lr_ph: 0, self.training_ph: False}))
            except tf.errors.OutOfRangeError:
                break
        print('Dataset accuracy %.4f' % np.mean(accs))

    def inference(self):
        # TODO: Make it use dynamic annotation file, handle one without state and crop mode
        print('Running inference')

        with open(self.dataset_dir + 'annotations.txt') as f:
            annotations = f.readlines()
        rand = random.randint(0, len(annotations) - 1)
        rgb_path, depth_path, state, _ = annotations[rand].strip().split(',')
        rgb_path = rgb_path[3:]
        depth_path = depth_path[3:]
        state = states.index(state)

        print('State is %d' % state)

        self.dataset_init([rgb_path], [depth_path], [state])
        inference, acc = self.sess.run([self.inference_op(), self.inference_acc_op],
                                       feed_dict={self.lr_ph: 0, self.training_ph: False})
        print('Inference: ' + str(inference))
        print('Accuracy: %.4f' % acc)
