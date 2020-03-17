#!/usr/bin/env python
# coding=utf-8

import numpy as np
import sys
import time
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)
import pickle

class DeepModel(BaseEstimator, TransformerMixin):
    def __init__(self, parser):
        self.deep_feature_size = parser.deep_feature_size
        self.hidden_units = parser.hidden_units
        self.epochs = parser.epochs
        self.batch_size = parser.batch_size
        self.learning_rate = parser.learning_rate
        self.model_pb = parser.model_pb
        self.decay_steps = parser.learning_rate_decay_steps
        self.decay_rate = parser.learning_rate_decay_rate
        self.l2_reg = parser.l2_reg
        self.metric_type = "auc"
        self.random_seed = 2019
        self.dropout_keep_deep = [1, 1, 1, 1, 1]
        self.cross_layer_num =2
        self.__init_graph()

    def __init_graph(self):
        tf.set_random_seed(self.random_seed)
        self.feature_values = tf.placeholder(tf.float32, [None, self.deep_feature_size], name='feature_values')
        self.input_data_size = tf.placeholder(tf.int32, [1], name="input_data_size")
        self.feature_indexs = tf.tile(
            tf.Variable([[i for i in range(self.hidden_units[-1])]], trainable=False, name="feature_indexs"),
            multiples=[self.input_data_size[0], 1])
        self.label = tf.placeholder(tf.float32, [None, 1], name='label')
        self.global_step = tf.Variable(0, trainable=False)
        self.weights = {}
        self.biases = {}

    def deep_func(self):
        with tf.name_scope('Embedding_Layer'):
                self.x0 = tf.concat([self.feature_values], axis=1)  # [None, category_feature_size*embedding_size + continuous_feature_size]

        with tf.name_scope('deep_network'):
            num_layer = len(self.hidden_units)
            input_size = self.x0.shape.as_list()[1]
            glorot = np.sqrt(2.0 / (input_size + self.hidden_units[0]))
            self.weights['deep_layer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(input_size, self.hidden_units[0])),
                dtype=np.float32)
            self.biases['deep_layer_bias_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_units[0])),
                dtype=np.float32)

            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.hidden_units[i - 1] + self.hidden_units[i]))
                self.weights['deep_layer_%s' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.hidden_units[i - 1], self.hidden_units[i])),
                    dtype=np.float32)
                self.biases['deep_layer_bias_%s' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_units[i])),
                    dtype=np.float32)

            self.deep_out = tf.nn.dropout(self.x0, self.dropout_keep_deep[0])
            for i in range(0, num_layer):
                self.deep_out = tf.add(tf.matmul(self.deep_out, self.weights['deep_layer_%s' % i]),
                                       self.biases['deep_layer_bias_%s' % i])
                self.deep_out = tf.nn.dropout(self.deep_out, self.dropout_keep_deep[i + 1])

        with tf.name_scope('cross_network'):
            input_size = self.x0.shape.as_list()[1]
            glorot = np.sqrt(2.0 / (input_size + input_size))
            # xl+1 = f(xl, wl, bl) + xl
            for i in range(self.cross_layer_num):
                self.weights['cross_layer_%s' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)), dtype=np.float32)
                self.biases['cross_bias_%s' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)), dtype=np.float32)

            self.xl = tf.reshape(self.x0, shape=[-1, input_size, 1])
            self._x0 = self.xl
            for i in range(self.cross_layer_num):
                self.xl = tf.tensordot(tf.matmul(self._x0, self.xl, transpose_b=True),
                                       self.weights['cross_layer_%s' % i], axes=1) + self.biases['cross_bias_%s' % i] + self.xl # [None, category_feature_size*embedding_size + continuous_feature_size, 1]

            self.cross_out = tf.reshape(self.xl, shape=[-1, input_size])

        with tf.name_scope('DCN_out'):
            input_size = self.cross_out.shape.as_list()[1] + self.hidden_units[-1]
            glorot = np.sqrt(2.0 / (input_size + 1))
            self.weights['concat_projection'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(input_size, 1)), dtype=np.float32)
            self.biases['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)
            self.out = tf.concat([self.cross_out, self.deep_out], axis=1)
            self.out = tf.add(tf.matmul(self.out, self.weights['concat_projection']), self.biases['concat_bias'])

    def modelOptimizer(self):
        self.deep_func()
        self.out = tf.nn.sigmoid(self.out)
        self.loss = tf.losses.log_loss(self.label, self.out)
        self.loss = tf.reduce_mean(self.loss)
        if self.l2_reg > 0:
            self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['concat_projection'])
        self.learning_rate_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                              self.decay_rate,
                                                              staircase=True)
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_decay).minimize(self.loss,
                                                                                                 global_step=self.global_step)
        tf.identity(self.out, name="output")
        sys.stdout.flush()

    def fit(self, trian_data, val_data):

        self.modelOptimizer()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            losses = []
            num_samples = 0
            for epoch in range(self.epochs):
                st = time.time()
                for i in range(len(trian_data)):
                    data_batch = pickle.loads(trian_data[i])
                    feed_dict = {
                        self.feature_values: data_batch["features"],
                        self.input_data_size: [len(data_batch["features"])],
                        self.label: data_batch["labels"],
                    }
                    self.loss_train, op = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
                    losses.append(self.loss_train * self.batch_size)
                    num_samples += self.batch_size

                end_time = time.time()
                total_loss = float(np.sum(losses) / num_samples)
                valid_metric = self.evaluate(sess, val_data)
                print('[%s] valid-%s=%.5f\tloss=%.5f [%.1f s]'
                      % (epoch + 1, self.metric_type, valid_metric, total_loss, end_time - st))
                sys.stdout.flush()

            # **************************保存为pb模型******************************
            model_signature = signature_def_utils.build_signature_def(
                inputs={
                    "feature_values": utils.build_tensor_info(self.feature_values),
                    "input_data_size": utils.build_tensor_info(self.input_data_size),
                },
                outputs={"output": utils.build_tensor_info(self.out)},
                method_name=signature_constants.PREDICT_METHOD_NAME)
            try:
                legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
                builder = saved_model_builder.SavedModelBuilder(self.model_pb)
                builder.add_meta_graph_and_variables(sess,
                                                     [tag_constants.SERVING],
                                                     clear_devices=True,
                                                     signature_def_map={
                                                         signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: model_signature, },
                                                     legacy_init_op=legacy_init_op)
                builder.save()
            except Exception as e:
                print("Fail to export saved model, exception: {}".format(e))
                sys.stdout.flush()

    def evaluate(self, sess, data_val):
        pred_list = []
        label_list = []
        for i in range(len(data_val)):
            data_batch = pickle.loads(data_val[i])
            feed_dict = {
                self.feature_values: data_batch["features"],
                self.input_data_size: [len(data_batch["features"])],
                self.label: data_batch["labels"],
            }
            label_list.extend(data_batch["labels"])
            y_pred = sess.run(self.out, feed_dict=feed_dict)
            pred_list.extend(y_pred)
        return roc_auc_score(label_list, pred_list)

    def predict(self, data_val):
        """
            加载pb模型
        """
        session = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], self.model_pb)
        pred_list = []
        label_list = []
        for i in range(len(data_val)):
            data_batch = pickle.loads(data_val[i])
            feed_dict_map = {
                "feature_values:0": data_batch["features"],
                "input_data_size:0": [len(data_batch["features"])],
            }
            var = session.run("Sigmoid:0", feed_dict=feed_dict_map)
            val_pred = var[:, 0]
            label_list.extend(data_batch["labels"])
            pred_list.extend(val_pred)
        print("val of auc:%.5f" % roc_auc_score(label_list, pred_list))
        sys.stdout.flush()
