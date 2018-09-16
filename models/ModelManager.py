import os
import json
import time
import shutil
import random
import itertools

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt

import models.RGB_AlexNet
import models.RGB_AlexNet_Pretrained_Bvlc
import models.RGB_AlexNet_Finetune_Bvlc
import models.RGBD_AlexNet
import models.RGBD_AlexNet_Finetune_Bvlc
import models.RGBD_AlexNet_Late_Single_Channel
import models.RGB_Inception
import models.RGBD_Inception
import models.RGBD_AlexNet_Colormap


models = {'RGB_AlexNet': models.RGB_AlexNet,
          'RGB_AlexNet_Pretrained': models.RGB_AlexNet_Pretrained_Bvlc,
          'RGB_AlexNet_Finetune': models.RGB_AlexNet_Finetune_Bvlc,
          'RGBD_AlexNet': models.RGBD_AlexNet,
          'RGBD_AlexNet_Finetune': models.RGBD_AlexNet_Finetune_Bvlc,
          'RGBD_AlexNet_Single_Channel': models.RGBD_AlexNet_Late_Single_Channel,
          'RGBD_AlexNet_Colormap': models.RGBD_AlexNet_Colormap,
          'RGB_Inception': models.RGB_Inception,
          'RGBD_Inception': models.RGBD_Inception}


class ModelManager(object):

    def __init__(self, model, config):
        # Get all configs
        self.model = models[model]
        self.config = config
        with open(self.config) as f:
            data = json.load(f)
        self.train_epochs = data['train_epochs']
        self.dataset_parent_dir = data['dataset_parent_dir']
        self.dataset_dir = self.dataset_parent_dir + 'tongue_dataset/scaled2/'
        self.dataset_size_limit = None if 'dataset_size_limit' not in data else data['dataset_size_limit']
        self.batch_size = self.model.get_batch_size() if 'batch_size' not in data else data['batch_size']
        self.train_annotations = data['train_annotations']
        self.val_annotations = data['val_annotations']
        self.test_annotations = data['test_annotations']
        self.save_dir = data['save_dir']
        self.max_saves = data['max_saves']
        self.number_outputs = data['number_outputs']
        self.cross = None if 'cross' not in data else data['cross']
        if self.number_outputs == 7:
            self.states = ['mouth_closed', 'mouth_open', 'tongue_down', 'tongue_left', 'tongue_middle', 'tongue_right', 'tongue_up']
        elif self.number_outputs == 6:
            self.states = ['mouth_closed', 'mouth_open', 'tongue_down', 'tongue_left', 'tongue_right', 'tongue_up']
            self.train_annotations = self.train_annotations[:-4] + '_6classes.txt'
            self.val_annotations = self.val_annotations[:-4] + '_6classes.txt'
            self.test_annotations = self.test_annotations[:-4] + '_6classes.txt'
        else:
            self.states = None
            self.log('Invalid value for number of outputs, can only be 6 or 7')
            exit(1)
        if self.model.get_model_name() == 'RGBD_AlexNet_Colormap2':
            self.train_annotations = self.train_annotations[:-4] + '_colormap.txt'
            self.val_annotations = self.val_annotations[:-4] + '_colormap.txt'
            self.test_annotations = self.test_annotations[:-4] + '_colormap.txt'

        # Prepare model
        self.log('Starting tensorflow session')
        session_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=session_config)
        self.log('Preparing graph')
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

    def log(self, msg):
        logfile = self.save_dir + self.model.get_model_name() + '_' + str(self.number_outputs) + 'way'
        if self.cross is not None: 
            logfile += '_cross' + str(self.cross)
        logfile += '.txt'

        print(msg)
        with open(logfile, 'a') as f:
            f.write(msg + '\n')
        
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

        def _map(rgb_path, depth_path, state):
            rgb_string = tf.read_file(rgb_path)
            rgb_img = tf.image.decode_png(rgb_string, channels=3, dtype=tf.uint16)
            rgb_img = tf.cast(rgb_img, tf.float32)
            rgb_img = tf.reshape(rgb_img, [227, 227, 3])

            depth_string = tf.read_file(depth_path)
            depth_img = tf.image.decode_png(depth_string, channels=self.model.get_depth_channels(), dtype=tf.uint16)
            depth_img = tf.cast(depth_img, tf.float32)
            depth_img = tf.reshape(depth_img, [227, 227, self.model.get_depth_channels()])

            label = tf.one_hot(state, self.number_outputs)
            return rgb_img, depth_img, label

        dataset = dataset.map(_map)
        dataset = dataset.batch(batch_size_ph)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        iterator.make_initializer(dataset, name='dataset_init')
        return iterator

    def prepare_graph(self):
        # Check for savedModel of the current model
        if os.path.isdir(self.save_dir + self.model.get_model_name()):
            # Load latest save
            saves = os.listdir("%s/%s" % (self.save_dir, self.model.get_model_name()))
            saves = [x for x in saves if self.model.get_model_name() in x]
            if len(saves) > 0:
                saves = [(x, int(x.split('-')[-1])) for x in saves]
                saves.sort(key=lambda tup: tup[1])
                latest_save = ('%s%s/%s' % (self.save_dir, self.model.get_model_name(), saves[-1][0]))
                self.log('Loading model: %s' % latest_save)
                tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], latest_save)
                return

        self.log('Building model')

        # Build dataset and get iterator
        self.log('Creating dataset iterator')
        dataset_iterator = self.create_dataset()

        # Get device list
        self.log('Getting device list')
        local_device_protos = device_lib.list_local_devices()
        devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
        devices = ['/gpu:%d' % x for x in range(len(devices))]
        if len(devices) < 1:
            devices.append('/cpu:0')
        self.log('Devices found: ' + str(devices))

        # Build model on each available device
        training_ph = tf.placeholder(tf.bool, name='training_ph')
        for x in range(len(devices)):
            self.log('Building tower %d' % x)
            with tf.device(devices[x]):
                with tf.name_scope('tower_%d' % x):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=x != 0):
                        rgb_x, depth_x, y = dataset_iterator.get_next()
                        tf.identity(rgb_x, name='rgb_x')
                        tf.identity(depth_x, name='depth_x')
                        tf.identity(y, name='y')

                        # Build instance of model
                        self.model.build_model(rgb_x, depth_x, y, self.batch_size, x != 0, training_ph, self.number_outputs)

        # Average gradients from all devices
        self.log('Building train op')
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
        saves = [x for x in saves if self.model.get_model_name() in x]
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
        self.log('Saved model: %s' % save_path)

        # Delete oldest save if max number of saves reached
        if len(saves) >= self.max_saves:
            saves = [(x, int(x.split('-')[-1])) for x in saves]
            saves.sort(key=lambda tup: tup[1])
            shutil.rmtree("%s%s/%s" % (self.save_dir, self.model.get_model_name(), saves[0][0]))

    def feed_from_annotation(self, annotations):
        dataset_size = len(annotations) if self.dataset_size_limit is None else self.dataset_size_limit
        annotations = [x.strip().split(',') for x in annotations[:dataset_size]]
        rgb_path, depth_path, state, _ = zip(*annotations)
        rgb_path = [self.dataset_parent_dir + x[3:] for x in rgb_path]
        depth_path = [self.dataset_parent_dir + x[3:] for x in depth_path]
        state = [self.states.index(x) for x in state]
        return rgb_path, depth_path, state

    def add_static_summary(self, writer, identifier, value):
        summary = tf.Summary()
        summary.value.add(tag=identifier, simple_value=value)
        writer.add_summary(summary, tf.train.global_step(self.sess, tf.train.get_global_step()))

    def get_train_and_val_annotations(self):
        with open(self.dataset_dir + self.train_annotations) as f:
            orig_train_annotations = f.readlines()
        with open(self.dataset_dir + self.val_annotations) as f:
            orig_val_annotations = f.readlines()

        if self.cross is None:
            return orig_train_annotations, orig_val_annotations

        train_participants = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17]
        cross_train_annotations = []
        cross_val_annotations = []
        for line in orig_train_annotations:
            if '/%03d_0' % train_participants[self.cross-1] in line:
                cross_val_annotations.append(line)
            else:
                cross_train_annotations.append(line)
        return cross_train_annotations, cross_val_annotations


    def train(self):
        # Get train and validation dataset feeds
        train_annotations, val_annotations = self.get_train_and_val_annotations()
        train_rgb, train_depth, train_state = self.feed_from_annotation(train_annotations)
        val_rgb, val_depth, val_state = self.feed_from_annotation(val_annotations)

        # Setup Tensorboard stuff
        merged_summaries = tf.summary.merge_all()
        if self.cross is None:
            self.log('Setting up summary writers')
            train_writer = tf.summary.FileWriter(self.save_dir + self.model.get_model_name() + '/events/train/', self.sess.graph)
            val_writer = tf.summary.FileWriter(self.save_dir + self.model.get_model_name() + '/events/validation/', self.sess.graph)
            shutil.copy(self.config, self.save_dir + self.model.get_model_name() + '/' + self.config)

        # Train for a bunch of epochs
        self.log('Training model for %d epochs' % self.train_epochs)
        epoch_times = []
        metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        for epoch in range(self.train_epochs):

            # Learn on training data for an epoch
            epoch_start_time = time.time()
            self.dataset_init(train_rgb, train_depth, train_state)
            train_losses = []
            train_accs = []
            while True:
                try:
                    _, summary, loss, acc = self.sess.run([self.train_op, merged_summaries, self.avg_loss_op, self.avg_acc_op],
                                                          feed_dict={self.lr_ph: self.model.get_learning_rate(),
                                                                     self.training_ph: True})
                    train_losses.append(loss)
                    train_accs.append(acc)
                except tf.errors.OutOfRangeError:
                    if self.cross is None:
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
            self.log('Done epoch # %d in %d seconds' % (epoch, epoch_end_time))
            metrics['train_loss'].append(np.mean(train_losses))
            metrics['train_acc'].append(np.mean(train_accs))
            metrics['val_loss'].append(np.mean(val_losses))
            metrics['val_acc'].append(np.mean(val_accs))
            if self.cross is None:
                self.add_static_summary(train_writer, 'summaries/epoch_time', epoch_end_time)
                self.add_static_summary(val_writer, 'summaries/avg_loss', np.mean(val_losses))
                self.add_static_summary(val_writer, 'summaries/avg_acc', np.mean(val_accs))
                self.save()
        self.log('Finished training %d epochs in %d seconds' % (self.train_epochs, int(np.sum(epoch_times))))
        if self.cross is not None:
            np.savez(self.save_dir + '/' + self.model.get_model_name() + '_' + str(self.cross),
                     train_loss=np.asanyarray(metrics['train_loss']),
                     train_acc=np.asanyarray(metrics['train_acc']),
                     val_loss=np.asanyarray(metrics['val_loss']),
                     val_acc=np.asanyarray(metrics['val_acc']))

    def test(self, test_annotations_path=None):
        self.log('Running test')
        if test_annotations_path is None:
            test_annotations_path = self.dataset_dir + self.test_annotations
        with open(test_annotations_path) as f:
            test_annotations = f.readlines()
        rgb, depth, state = self.feed_from_annotation(test_annotations)
        self.dataset_init(rgb, depth, state)
        inferences = []
        accs = []
        while True:
            try:
                logits, acc = self.sess.run([self.inference_op, self.inference_acc_op],
                                          feed_dict={self.lr_ph: 0, self.training_ph: False})
                inference = np.argmax(logits, axis=-1)
                inferences.append(inference)
                accs.append(acc)
            except tf.errors.OutOfRangeError:
                break
        self.log('Test accuracy %.4f' % np.mean(accs))

        self.log('Confusion matrix')
        predictions = np.concatenate(inferences).tolist()
        predictions = [self.states[x] for x in predictions]
        state = [self.states[x] for x in state]
        cm = confusion_matrix(state, predictions, self.states)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(self.states))
        plt.xticks(tick_marks, self.states, rotation=45)
        plt.yticks(tick_marks, self.states)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.save_dir + self.model.get_model_name() +'/confusion_matrix.png')

    def inference(self):
        # TODO: Make it use dynamic annotation file, handle one without state and crop mode
        self.log('Running inference')

        with open(self.dataset_dir + 'annotations.txt') as f:
            annotations = f.readlines()
        rand = random.randint(0, len(annotations) - 1)
        rgb_path, depth_path, state, _ = annotations[rand].strip().split(',')
        rgb_path = rgb_path[3:]
        depth_path = depth_path[3:]
        state = self.states.index(state)

        self.log('State is %d' % state)

        self.dataset_init([rgb_path], [depth_path], [state])
        inference, acc = self.sess.run([self.inference_op, self.inference_acc_op],
                                       feed_dict={self.lr_ph: 0, self.training_ph: False})
        self.log('Inference: ' + str(inference))
        self.log('Accuracy: %.4f' % acc)
