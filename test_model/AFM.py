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
        self.cross_layer_num = 3
        self.attention_size = 10
        self.embedding_size = 8
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
            self.x0 = tf.concat([self.feature_values], axis=1)
        with tf.name_scope('Embedding_Layer'):
            self.weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.deep_feature_size, self.embedding_size], 0.0, 0.01), name='feature_embeddings')
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],
                                                     self.feature_indexs)
            feat_value = tf.reshape(self.feature_values, shape=[-1, self.deep_feature_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)
        with tf.name_scope('linear_part'):
            self.biases['feature_bias'] = tf.Variable(tf.random_uniform([self.deep_feature_size, 1], 0.0, 1.0),
                                                      name='feature_bias')
            self.linear_part = tf.nn.embedding_lookup(self.biases['feature_bias'],
                                                      self.feature_indexs)
            self.linear_part = tf.reduce_sum(tf.multiply(self.linear_part, feat_value), axis=2)

        with tf.name_scope('Pair-wise_Interaction_Layer'):
            pair_wise_product_list = []
            for i in range(self.deep_feature_size):
                for j in range(i + 1, self.deep_feature_size):
                    pair_wise_product_list.append(
                        tf.multiply(self.embeddings[:, i, :], self.embeddings[:, j, :]))
            self.pair_wise_product = tf.stack(
                pair_wise_product_list)
            self.pair_wise_product = tf.transpose(self.pair_wise_product, perm=[1, 0, 2],
                                                  name='pair_wise_product')

        with tf.name_scope('attention_net'):
            glorot = np.sqrt(2.0 / (self.attention_size + self.embedding_size))
            self.weights['attention_w'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.embedding_size, self.attention_size)),
                dtype=tf.float32, name='attention_w')
            self.biases['attention_b'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.attention_size)),
                dtype=tf.float32, name='attention_b')
            self.weights['attention_h'] = tf.Variable(np.random.normal(loc=0, scale=1, size=(1, self.attention_size)),
                                                      dtype=tf.float32, name='attention_h')
            self.weights['attention_p'] = tf.Variable(np.random.normal(loc=0, scale=1, size=(self.embedding_size, 1)),
                                                      dtype=tf.float32, name='attention_p')

            num_interactions = self.pair_wise_product.shape.as_list()[1]
            # w*x + b
            self.attention_wx_plus_b = tf.reshape(
                tf.add(tf.matmul(tf.reshape(self.pair_wise_product, shape=[-1, self.embedding_size]),
                                 self.weights['attention_w']),
                       self.biases['attention_b']),
                shape=[-1, num_interactions,
                       self.attention_size])

            # relu(w*x + b)
            self.attention_relu_wx_plus_b = tf.nn.relu(
                self.attention_wx_plus_b)

            # h*relu(w*x + b)
            self.attention_h_mul_relu_wx_plus_b = tf.multiply(self.attention_relu_wx_plus_b, self.weights[
                'attention_h'])

            # exp(h*relu(w*x + b))
            self.attention_exp = tf.exp(tf.reduce_sum(self.attention_h_mul_relu_wx_plus_b, axis=2,
                                                      keep_dims=True))

            # sum(exp(h*relu(w*x + b)))
            self.attention_exp_sum = tf.reduce_sum(self.attention_exp, axis=1, keep_dims=True)

            # exp(h*relu(w*x + b)) / sum(exp(h*relu(w*x + b)))
            self.attention_out = tf.div(self.attention_exp, self.attention_exp_sum,
                                        name='attention_out')

            # attention*Pair-wise
            self.attention_product = tf.multiply(self.attention_out,
                                                 self.pair_wise_product)

            self.attention_product = tf.reduce_sum(self.attention_product, axis=1)

            # p*attention*Pair-wise
            self.attention_net_out = tf.matmul(self.attention_product, self.weights['attention_p'])

        with tf.name_scope('out'):
            self.weights['w0'] = tf.Variable(tf.constant(0.1), name='w0')
            self.linear_out = tf.reduce_sum(self.linear_part, axis=1, keep_dims=True)
            self.w0 = tf.multiply(self.weights['w0'], tf.ones_like(self.linear_out))
            self.out = tf.add_n(
                [self.w0, self.linear_out, self.attention_net_out])

    def modelOptimizer(self):
        self.deep_func()
        self.out = tf.nn.sigmoid(self.out)
        self.loss = tf.losses.log_loss(self.label, self.out)
        self.loss = tf.reduce_mean(self.loss)
        if self.l2_reg > 0:
            self.loss = self.loss +self.l2_reg * tf.nn.l2_loss(self.biases['feature_bias']) + self.l2_reg * tf.nn.l2_loss(self.weights['feature_embeddings'])

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
