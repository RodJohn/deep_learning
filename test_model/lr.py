#!/usr/bin/env python
# coding=utf-8

import numpy as np
import sys
import time
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

class DeepModel:
    def __init__(self, parser):
        self.deep_feature_size = parser.deep_feature_size
        self.wide_feature_size = parser.wide_feature_size
        self.wide_feature_size_max = parser.wide_feature_size_max
        self.hidden_units = parser.hidden_units
        self.epochs = parser.epochs
        self.batch_size = parser.batch_size
        self.learning_rate = parser.learning_rate
        self.model_pb = parser.model_pb
        self.decay_steps = parser.learning_rate_decay_steps
        self.decay_rate = parser.learning_rate_decay_rate
        self.l2_reg = parser.l2_reg
        self.use_wide = parser.use_wide
        self.use_deep = parser.use_deep
        self.metric_type = "auc"
        self.random_seed = 2019
        self.__init_graph()

    def __init_graph(self):
        tf.set_random_seed(self.random_seed)
        self.wide_feature_values = tf.placeholder(tf.float32, [None, self.wide_feature_size],
                                                  name='wide_feature_values')
        self.wide_feature_indexs = tf.placeholder(tf.int32, [None, self.wide_feature_size], name='wide_feature_indexs')
        self.deep_feature_values = tf.placeholder(tf.float32, [None, self.deep_feature_size], name='deep_feature')
        self.input_data_size = tf.placeholder(tf.int32, [1], name="input_data_size")
        self.deep_feature_indexs = tf.tile(
            tf.Variable([[i for i in range(self.hidden_units[-1])]], trainable=False, name="deep_feature_indexs"),
            multiples=[self.input_data_size[0], 1])
        self.label = tf.placeholder(tf.float32, [None, 1], name='label')
        self.global_step = tf.Variable(0, trainable=False)
        self.weights = {}
        self.biases = {}

        print("--deep_feature_values:", self.deep_feature_values)
        print("--wide_feature_indexs:", self.wide_feature_indexs)
        print("--wide_feature_values:", self.wide_feature_values)
        print("--input_data_size:", self.input_data_size)

    def wide_func(self):
        """
            LR模型部分
        """
        with tf.name_scope('wide_part'):
            glorot = np.sqrt(2.0 / (self.wide_feature_size_max + 1))
            self.wide_weights_all = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.wide_feature_size_max, 1)),
                dtype=np.float32)
            if self.use_wide and self.use_deep == False:
                self.weights["wide_w"] = tf.nn.embedding_lookup(self.wide_weights_all, ids=self.wide_feature_indexs)
                self.weights["wide_w"] = tf.reshape(self.weights["wide_w"], shape=[-1, self.wide_feature_size])
                self.weights["wide_b"] = tf.Variable(tf.random_normal([1]))
                self.wide_res = tf.add(
                    tf.reduce_sum(tf.multiply(self.weights["wide_w"], self.wide_feature_values), axis=1,
                                  keep_dims=True), self.weights["wide_b"])
            elif self.use_wide and self.use_deep:
                self.wide_feature_indexs_add = tf.add(self.wide_feature_indexs, self.hidden_units[-1])
            else:
                pass

    def deep_func(self):
        """
            DNN模型部分
        """
        with tf.name_scope('deep_part'):
            len_layers = len(self.hidden_units)
            self.dense_vector = self.deep_feature_values
            input_size = self.dense_vector.shape.as_list()[1]
            glorot = np.sqrt(2.0 / (input_size + self.hidden_units[0]))
            self.weights['deep_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(input_size, self.hidden_units[0])),
                dtype=np.float32)
            self.biases['deep_bias_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_units[0])),
                dtype=np.float32)

            for i in range(1, len_layers):
                glorot = np.sqrt(2.0 / (self.hidden_units[i - 1] + self.hidden_units[i]))
                self.weights['deep_%s' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.hidden_units[i - 1], self.hidden_units[i])),
                    dtype=np.float32)
                self.biases['deep_bias_%s' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_units[i])),
                    dtype=np.float32)

            self.deep_res = tf.nn.dropout(self.dense_vector, 1)
            for i in range(0, len_layers):
                self.deep_res = tf.add(tf.matmul(self.deep_res, self.weights['deep_%s' % i]),
                                       self.biases['deep_bias_%s' % i])
                self.deep_res = tf.nn.relu(self.deep_res)

                self.deep_res = tf.nn.dropout(self.deep_res, 1)

            if self.use_deep and self.use_wide == False:
                glorot = np.sqrt(2.0 / (self.hidden_units[-1] + 1))
                self.weights['deep_res'] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.hidden_units[-1], 1)), dtype=np.float32)
                self.biases['deep_res_bias'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, 1)),
                                                           dtype=np.float32)
                self.deep_res = tf.add(tf.matmul(self.deep_res, self.weights['deep_res']),
                                       self.biases['deep_res_bias'])

    def wide_and_deep(self):
        """"
            wide and deep 合并
        """
        print("--wide deep:", self.use_wide, self.use_deep)
        self.wide_func()
        self.deep_func()

        with tf.name_scope('wide_deep'):
            if self.use_wide and self.use_deep:
                glorot = np.sqrt(2.0 / (self.wide_feature_size_max + 1))
                self.weights["wdl_weights"] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.wide_feature_size_max, 1)),
                    dtype=np.float32)
                self.biases['wdl_bias'] = tf.Variable(tf.random_normal([1]))
                self.wdl_feature_indexs = tf.concat([self.deep_feature_indexs, self.wide_feature_indexs_add], axis=1)
                self.wdl_feature_values = tf.concat([self.deep_res, self.wide_feature_values], axis=1)
                self.weights["wdl_weights"] = tf.nn.embedding_lookup(self.weights["wdl_weights"],
                                                                     ids=self.wdl_feature_indexs)
                self.weights["wdl_weights"] = tf.reshape(self.weights["wdl_weights"],
                                                         shape=[-1, self.hidden_units[-1] + self.wide_feature_size])

                self.out = tf.add(
                    tf.reduce_sum(tf.multiply(self.weights["wdl_weights"], self.wdl_feature_values), axis=1,
                                  keep_dims=True), self.biases['wdl_bias'])
            elif self.use_wide and self.use_deep == False:
                self.out = self.wide_res
            elif self.use_wide == False and self.use_deep:
                self.out = self.deep_res

        self.out = tf.nn.sigmoid(self.out)
        self.loss = tf.losses.log_loss(self.label, self.out)
        self.loss = tf.reduce_mean(self.loss)

        # l2 regularization on weights
        if self.l2_reg > 0:
            if self.use_wide and self.use_deep:
                self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['wdl_weights'])
            if self.use_deep:
                for i in range(len(self.hidden_units)):
                    self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.l2_reg)(
                        self.weights['deep_%s' % i])

        self.learning_rate_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                              self.decay_rate,
                                                              staircase=True)
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_decay).minimize(self.loss,
                                                                                                 global_step=self.global_step)

    def fit(self, train_data, val_data):
        self.wide_and_deep()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            losses = []
            num_samples = 0
            for epoch in range(self.epochs):
                st = time.time()
                for i in range(len(train_data)):
                    data_batch = pickle.loads(train_data[i])
                    feed_dict = {
                        self.wide_feature_values: data_batch["wide_feature_values"],
                        self.wide_feature_indexs: data_batch["wide_feature_indexs"],
                        self.deep_feature_values: data_batch["deep_feature"],
                        self.input_data_size: [len(data_batch["deep_feature"])],
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

            input_dict = {
                "deep_feature": utils.build_tensor_info(self.deep_feature_values),
                "wide_feature_values": utils.build_tensor_info(self.wide_feature_values),
                "wide_feature_indexs": utils.build_tensor_info(self.wide_feature_indexs),
                "input_data_size": utils.build_tensor_info(self.input_data_size),
            }
            output_dict = {"output": utils.build_tensor_info(self.out)}
            model_signature = signature_def_utils.build_signature_def(
                inputs=input_dict,
                outputs=output_dict,
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
        """
                验证集数据计算AUC
        """
        pred_list = []
        label_list = []
        for i in range(len(data_val)):
            data_batch = pickle.loads(data_val[i])
            feed_dict = {
                self.wide_feature_values: data_batch["wide_feature_values"],
                self.wide_feature_indexs: data_batch["wide_feature_indexs"],
                self.deep_feature_values: data_batch["deep_feature"],
                self.input_data_size: [len(data_batch["deep_feature"])],
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
                "deep_feature:0": data_batch["deep_feature"],
                "wide_feature_values:0": data_batch["wide_feature_values"],
                "wide_feature_indexs:0": data_batch["wide_feature_indexs"],
                "input_data_size:0": [len(data_batch["deep_feature"])],
            }
            var = session.run("Sigmoid:0", feed_dict=feed_dict_map)
            val_pred = var[:, 0]
            label_list.extend(data_batch["labels"])
            pred_list.extend(val_pred)
        print("val of auc:%.5f" % roc_auc_score(label_list, pred_list))
        sys.stdout.flush()
