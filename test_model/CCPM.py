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
        self.embedding_size=8
        self.convolution_layers_filters = [64, 64, 64]
        self.convolution_layers_num = 3
        self.__init_graph()

    def __init_graph(self):
        tf.set_random_seed(self.random_seed)
        self.feature_values = tf.placeholder(tf.float32, [None, self.deep_feature_size], name='feature_values')
        self.input_data_size = tf.placeholder(tf.int32, [1], name="input_data_size")
        self.feature_indexs = tf.tile(
            tf.Variable([[i for i in range(self.deep_feature_size)]], trainable=False, name="deep_feature_indexs"),
            multiples=[self.input_data_size[0], 1])
        self.label = tf.placeholder(tf.float32, [None, 1], name='label')
        self.global_step = tf.Variable(0, trainable=False)
        self.weights = {}
        self.biases = {}

    def deep_func(self):
        with tf.name_scope('Embedding_Layer'):
            self.embeddings = tf.keras.layers.Embedding(input_dim=self.deep_feature_size, output_dim=self.embedding_size)(
                self.feature_indexs)
        feat_value = tf.reshape(self.feature_values,
                                shape=[-1, self.deep_feature_size, 1])
        self.embeddings = tf.multiply(self.embeddings, feat_value)
        self.embeddings = tf.expand_dims(self.embeddings, axis=-1)
        self.l = tf.transpose(self.embeddings, perm=[0, 2, 1, 3])

        with tf.name_scope('Convlution_Layer'):
            self.l = tf.keras.layers.Conv2D(filters=self.convolution_layers_filters[0], kernel_size=(3, 3),
                                            padding='same',
                                            activation=tf.nn.relu,
                                            input_shape=(self.embedding_size, self.deep_feature_size, 1))(self.l)
            self.l = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(self.l)

            for i in range(1, self.convolution_layers_num):
                self.l = tf.keras.layers.Conv2D(filters=self.convolution_layers_filters[i], kernel_size=(3, 3),
                                                padding='same', activation=tf.nn.relu)(self.l)
                self.l = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(self.l)

            self.out = tf.keras.layers.Flatten()(self.l)
            self.out = tf.keras.layers.Dense(1, activation=None)(self.out)

    def modelOptimizer(self):
        self.deep_func()
        self.out = tf.nn.sigmoid(self.out)
        self.loss = tf.losses.log_loss(self.label, self.out)
        self.loss = tf.reduce_mean(self.loss)
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
