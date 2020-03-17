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
        self.embedding_size = parser.embedding_size
        self.metric_type = "auc"
        self.random_seed = 2019
        self.deep_init_size = 50
        self.dropout_keep_deep = [1, 1, 1, 1, 1]
        self.use_inner =True

        self.__init_graph()

    def __init_graph(self):

        tf.set_random_seed(self.random_seed)
        self.feature_values = tf.placeholder(tf.float32, [None, self.deep_feature_size], name='feature_values')
        self.input_data_size = tf.placeholder(tf.int32, [1], name="input_data_size")
        self.feature_indexs = tf.tile(
            tf.Variable([[i for i in range(self.deep_feature_size)]], trainable=False, name="feature_indexs"),
            multiples=[self.input_data_size[0], 1])
        self.label = tf.placeholder(tf.float32, [None, 1], name='label')
        self.global_step = tf.Variable(0, trainable=False)
        self.weights = {}
        self.biases = {}


    def deep_func(self):
        with tf.name_scope('Embedding_Layer'):
            self.weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.deep_feature_size, self.embedding_size], 0.0, 0.01), name='feature_embeddings')
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feature_indexs)
            feat_value = tf.reshape(self.feature_values, shape=[-1,  self.deep_feature_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)
        with tf.name_scope('Product_Layer'):
            if self.use_inner:
                self.weights['product-quadratic-inner'] = tf.Variable(
                    tf.random_normal([self.deep_init_size,  self.deep_feature_size], 0.0, 0.01))
            else:
                self.weights['product-quadratic-outer'] = tf.Variable(
                    tf.random_normal([self.deep_init_size, self.embedding_size, self.embedding_size], 0.0, 0.01))
            self.weights['product-linear'] = tf.Variable(
                tf.random_normal([self.deep_init_size,  self.deep_feature_size, self.embedding_size], 0.0, 0.01))
            self.biases['product-bias'] = tf.Variable(tf.random_normal([self.deep_init_size], 0, 0, 1.0))

            # linear signals z
            linear_output = []
            for i in range(self.deep_init_size):
                self.linear_product = tf.multiply(self.embeddings, self.weights['product-linear'][i])
                self.linear_product = tf.reduce_sum(self.linear_product, axis=[1, 2])
                self.linear_product = tf.reshape(self.linear_product, shape=[-1, 1])
                linear_output.append(self.linear_product)

            self.lz = tf.concat(linear_output, axis=1)  # [N, deep_init_size]

            # quadratic signals p
            quadratic_output = []
            if self.use_inner:
                for i in range(self.deep_init_size):
                    theta = tf.multiply(self.embeddings, tf.reshape(self.weights['product-quadratic-inner'][i],
                                                                    [1, -1, 1]))  # [None, field_size, embedding_size]
                    quadratic_output.append(
                        tf.reshape(tf.norm(tf.reduce_sum(theta, axis=1), axis=1), shape=[-1, 1]))  # [None, 1]
            else:
                embedding_sum = tf.reduce_sum(self.embeddings, axis=1)
                p = tf.matmul(tf.expand_dims(embedding_sum, 2),
                              tf.expand_dims(embedding_sum, 1))  # [None, embedding_size, embedding_size]
                for i in range(self.deep_init_size):
                    theta = tf.multiply(p, tf.expand_dims(self.weights['product-quadratic-outer'][i],
                                                          0))  # [None, embedding_size, embedding_size]
                    quadratic_output.append(tf.reshape(tf.reduce_sum(theta, axis=[1, 2]), shape=(-1, 1)))  # [None, 1]

            self.lp = tf.concat(quadratic_output, axis=1)  # [N, deep_init_size]
            z_add_p = tf.add(self.lp, self.lz)
            self.deep_out = tf.nn.relu(tf.add(z_add_p, self.biases['product-bias']))
            self.deep_out = tf.nn.dropout(self.deep_out, self.dropout_keep_deep[0])  # [N, deep_init_size]

        with tf.name_scope('Hidden_Layer'):
            num_layer = len(self.hidden_units)
            input_size = self.deep_out.shape.as_list()[1]
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

            for i in range(0, num_layer):
                self.deep_out = tf.add(tf.matmul(self.deep_out, self.weights['deep_layer_%s' % i]),
                                       self.biases['deep_layer_bias_%s' % i])
                self.deep_out = tf.nn.relu(self.deep_out)
                self.deep_out = tf.nn.dropout(self.deep_out, self.dropout_keep_deep[i + 1])

            glorot = np.sqrt(2.0 / (self.hidden_units[-1] + 1))
            self.weights['deep_out'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_units[-1], 1)),
                dtype=np.float32)
            self.biases['deep_out_bias'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, 1)),
                                                  dtype=np.float32)
            self.out = tf.add(tf.matmul(self.deep_out, self.weights['deep_out']), self.biases['deep_out_bias'])

    def modelOptimizer(self):
        self.deep_func()
        self.out = tf.nn.sigmoid(self.out)
        self.loss = tf.losses.log_loss(self.label, self.out)
        self.loss = tf.reduce_mean(self.loss)
        if self.l2_reg > 0:
            for i in range(len(self.hidden_units)):
                self.loss = self.loss + tf.contrib.layers.l1_regularizer(self.l2_reg)(
                    self.weights['deep_layer_%s' % i])
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
