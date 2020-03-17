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
    def __init__(self, args):
        self.deep_feature_size = args.deep_feature_size
        self.hidden_units = args.hidden_units
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.model_pb = args.model_pb
        self.decay_steps = args.learning_rate_decay_steps
        self.decay_rate = args.learning_rate_decay_rate
        self.l2_reg = args.l2_reg
        self.use_attention = args.use_attenton
        self.kernel_sizes = args.kernel_sizes
        self.out_filters = args.out_filters

        self.base_user_index = args.base_user_index
        self.base_item_index = args.base_item_index
        self.cross_index = args.cross_index
        self.w2v_user_index = args.w2v_user_index
        self.w2v_item_index = args.w2v_item_index

        self.dropout_keep_deep = [1, 1, 1, 1, 1]
        self.metric_type = "auc"
        self.random_seed = 2019

        self.__init_graph()

    def __init_graph(self):
        tf.set_random_seed(self.random_seed)
        self.feature_values = tf.placeholder(tf.float32, [None, self.deep_feature_size], name='deep_feature')
        self.wide_feature_values = tf.placeholder(tf.float32, name='wide_feature_values')
        self.wide_feature_indexs = tf.placeholder(tf.int32, name='wide_feature_indexs')

        self.input_data_size = tf.placeholder(tf.int32, [1], name="input_data_size")
        self.feature_indexs = tf.tile(
            tf.Variable([[i for i in range(self.hidden_units[-1])]], trainable=False, name="feature_indexs"),
            multiples=[self.input_data_size[0], 1])

        self.base_user_feature = self.feature_values[:, self.base_user_index[0]:self.base_user_index[1]]
        self.base_item_feature = self.feature_values[:, self.base_item_index[0]:self.base_item_index[1]][:, 0:18]
        self.cross_feature = self.feature_values[:, self.cross_index[0]:self.cross_index[1]]
        self.cross_feature1 = self.feature_values[:, self.cross_index[0]:self.cross_index[1]][:, 0:18]
        self.cross_feature2 = self.feature_values[:, self.cross_index[0]:self.cross_index[1]][:, 18:36]
        self.cross_feature3 = self.feature_values[:, self.cross_index[0]:self.cross_index[1]][:, 36:54]

        self.label = tf.placeholder(tf.float32, [None, 1], name='label')
        self.global_step = tf.Variable(0, trainable=False)

        self.weights = {}
        self.biases = {}

    def _kcnn(self, emb):
        concat_input1 = tf.expand_dims(emb, 1)
        outputs = []
        for kernel in self.kernel_sizes:
            conv1 = tf.layers.conv1d(inputs=concat_input1, filters=self.out_filters, kernel_size=kernel, strides=2,
                                     padding="same", activation=tf.nn.relu)
            max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=4, strides=4, padding="same")
            outputs.append(max_pool_1)
        output = tf.concat(outputs, axis=-1)
        max_pool = tf.reshape(output, [-1, self.out_filters * len(self.kernel_sizes)])

        return max_pool

    def _attention(self, user_emb, item_emb):
        # user_emb_cnn = self._kcnn(user_emb)
        # item_emb_cnn = self._kcnn(item_emb)
        clicked_embeddings = tf.reshape(user_emb, shape=[-1, 1, 18])
        news_embeddings_expanded = tf.expand_dims(item_emb, 1)
        attention_weights1 = tf.reduce_sum(clicked_embeddings * news_embeddings_expanded, axis=-1)
        attention_weights2 = tf.nn.sigmoid(attention_weights1)
        attention_weights_expanded = tf.expand_dims(attention_weights2, axis=-1)
        user_emb_att = tf.reduce_sum(clicked_embeddings * attention_weights_expanded, axis=1)
        item_emb_att = item_emb

        return user_emb_att, item_emb_att

    def deep_func(self):
        with tf.name_scope('deep_part'):
            bc_feat = self.feature_values
            if self.use_attention:
                cross_user_att1, _ = self._attention(self.cross_feature1, self.base_user_feature)
                cross_user_att2, _ = self._attention(self.cross_feature2, self.base_user_feature)
                cross_user_att3, _ = self._attention(self.cross_feature3, self.base_user_feature)
                cross_user_item, _ = self._attention(self.base_item_feature, self.base_user_feature)
                dnn_input = tf.concat([bc_feat, cross_user_att1, cross_user_att2, cross_user_att3, cross_user_item], axis=-1)
            else:
                dnn_input = self.feature_values

            len_layers = len(self.hidden_units)
            self.dense_vector = dnn_input
            input_size = self.dense_vector.shape.as_list()[1]
            print("--model input size:", input_size)
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

            self.deep_res = tf.nn.dropout(self.dense_vector, self.dropout_keep_deep[0])
            for i in range(0, len_layers):
                self.deep_res = tf.add(tf.matmul(self.deep_res, self.weights['deep_%s' % i]),
                                       self.biases['deep_bias_%s' % i])
                self.deep_res = tf.nn.relu(self.deep_res)

                self.deep_res = tf.nn.dropout(self.deep_res, self.dropout_keep_deep[i + 1])

            glorot = np.sqrt(2.0 / (self.hidden_units[-1] + 1))
            self.weights['deep_res'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_units[-1], 1)), dtype=np.float32)
            self.biases['deep_res_bias'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, 1)),
                                                       dtype=np.float32)
            self.out = tf.add(tf.matmul(self.deep_res, self.weights['deep_res']),
                              self.biases['deep_res_bias'])

    def modelOptimizer(self):
        self.deep_func()
        self.out = tf.nn.sigmoid(self.out)
        self.loss = tf.losses.log_loss(self.label, self.out)
        self.loss = tf.reduce_mean(self.loss)
        if self.l2_reg > 0:
            for i in range(len(self.hidden_units)):
                self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.l2_reg)(
                    self.weights['deep_%s' % i])
        self.learning_rate_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                              self.decay_rate,
                                                              staircase=True)
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_decay).minimize(self.loss,
                                                                                                 global_step=self.global_step)
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
                        self.wide_feature_values: [],
                        self.wide_feature_indexs: [],
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

            input_dict = {
                "deep_feature": utils.build_tensor_info(self.feature_values),
                "wide_feature_values": utils.build_tensor_info(self.wide_feature_values),
                "wide_feature_indexs": utils.build_tensor_info(self.wide_feature_indexs),
                "input_data_size": utils.build_tensor_info(self.input_data_size),
            }

            model_signature = signature_def_utils.build_signature_def(
                inputs=input_dict,
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
                self.wide_feature_values: [],
                self.wide_feature_indexs: [],
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
                "deep_feature:0": data_batch["features"],
                "wide_feature_values:0": [],
                "wide_feature_indexs:0": [],
                "input_data_size:0": [len(data_batch["features"])],

            }
            var = session.run("Sigmoid:0", feed_dict=feed_dict_map)
            val_pred = var[:, 0]
            label_list.extend(data_batch["labels"])
            pred_list.extend(val_pred)
        print("val of auc:%.5f" % roc_auc_score(label_list, pred_list))
        sys.stdout.flush()
