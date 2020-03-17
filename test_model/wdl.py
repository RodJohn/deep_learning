#!/usr/bin/env python
# coding=utf-8

import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)


class DeepModel:
    def __init__(self, args):
        self.hidden_units = args.hidden_units
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.model_pb = args.model_pb
        self.decay_steps = args.learning_rate_decay_steps
        self.decay_rate = args.learning_rate_decay_rate
        self.l2_reg = args.l2_reg
        self.metric_type = "auc"
        self.random_seed = 2019
        self.dropout_keep_deep = [1, 1, 1, 1, 1]
        self.cont_field_size = args.cont_field_size
        self.cate_field_size = args.cate_field_size
        self.wide_field_size = args.wide_field_size
        self.cate_feats_size = args.cate_feats_size
        self.embedding_size = args.embedding_size

        self.use_wide = True
        self.use_deep = True

        self.__init_graph()

    def __init_graph(self):
        tf.set_random_seed(self.random_seed)
        self.cont_feats = tf.placeholder(tf.float32, [None, self.cont_field_size], name='cont_feats')
        self.cate_feats = tf.placeholder(tf.int32, [None, self.cate_field_size], name='cate_feats')
        self.wide_feats = tf.placeholder(tf.int32, [None, self.wide_field_size], name='wide_feats')
        print(self.cont_feats)
        print(self.cate_feats)
        print(self.wide_feats)

        self.input_data_len = tf.placeholder(tf.int32, [1], name="input_data_len")
        self.label = tf.placeholder(tf.float32, [None, 1], name='label')
        self.global_step = tf.Variable(0, trainable=False)

        weights = {}
        biases = {}

        with tf.name_scope('wide_part'):
            self.wide_out = self.wide_feats
            self.wide_feats_value = tf.ones(shape=[self.input_data_len[0], self.wide_field_size], dtype=tf.float32, name="wide_feats_value")
            print(self.wide_feats_value)

        with tf.name_scope('deep_part'):
            len_layers = len(self.hidden_units)

            embeddings = tf.get_variable('weight_mat',
                                         dtype=tf.float32,
                                         shape=(self.cate_feats_size, self.embedding_size),
                                         initializer=tf.contrib.layers.xavier_initializer())

            # category -> Embedding
            cate_emb = tf.nn.embedding_lookup(embeddings, ids=self.cate_feats)
            cate_emb = tf.reshape(cate_emb, shape=[-1, self.cate_field_size * self.embedding_size])
            deep_feats = tf.concat([self.cont_feats, cate_emb], axis=1, name='deep_feats')
            cate_size = cate_emb.shape.as_list()[1]
            deep_size = self.cont_field_size + cate_size
            print("cate_size = ", cate_size)
            print("deep_size = ", deep_size)

            glorot = np.sqrt(2.0 / (deep_size + self.hidden_units[0]))
            weights['deep_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(deep_size, self.hidden_units[0])),
                dtype=np.float32)
            biases['deep_bias_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_units[0])),
                dtype=np.float32)

            for i in range(1, len_layers):
                glorot = np.sqrt(2.0 / (self.hidden_units[i - 1] + self.hidden_units[i]))
                weights['deep_%s' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.hidden_units[i - 1], self.hidden_units[i])),
                    dtype=np.float32)
                biases['deep_bias_%s' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_units[i])),
                    dtype=np.float32)

            self.deep_res = tf.nn.dropout(deep_feats, self.dropout_keep_deep[0])
            for i in range(0, len_layers):
                self.deep_res = tf.add(tf.matmul(self.deep_res, weights['deep_%s' % i]), biases['deep_bias_%s' % i])
                self.deep_res = tf.nn.relu(self.deep_res)
                self.deep_res = tf.nn.dropout(self.deep_res, self.dropout_keep_deep[i + 1])

        with tf.name_scope('wide_deep'):
            if self.use_wide and self.use_deep:
                wdl_feats_size = self.cate_feats_size + self.hidden_units[-1]
                glorot = np.sqrt(2.0 / wdl_feats_size)
                weights["wdl_weights"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(wdl_feats_size, 1)), dtype=np.float32)
                biases['wdl_bias'] = tf.Variable(tf.random_normal([1]))

                deep_feats_index = tf.tile(
                    tf.Variable([[i for i in range(self.hidden_units[-1])]], trainable=False, name="deep_feats_index"),
                    multiples=[self.input_data_len[0], 1])
                deep_feats_index = tf.add(deep_feats_index, self.cate_feats_size)
                wdl_feature_index = tf.concat([self.wide_feats, deep_feats_index], axis=1)
                wdl_feature_value = tf.concat([self.wide_feats_value, self.deep_res], axis=1)
                wdl_weight = tf.nn.embedding_lookup(weights["wdl_weights"], ids=wdl_feature_index)
                wdl_weight = tf.reshape(wdl_weight, shape=[-1, self.wide_field_size + self.hidden_units[-1]])
                self.out = tf.add(tf.reduce_sum(tf.multiply(wdl_weight, wdl_feature_value), axis=1, keep_dims=True), biases['wdl_bias'])
            else:
                pass

        self.out = tf.nn.sigmoid(self.out)
        # self.loss = tf.losses.sigmoid_cross_entropy(self.label, self.out)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.out)
        self.loss = tf.reduce_mean(self.loss)

        # l2 regularization on weights
        if self.l2_reg > 0:
            if self.use_wide and self.use_deep:
                self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.l2_reg)(weights['wdl_weights'])
            if self.use_deep:
                for i in range(len(self.hidden_units)):
                    self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.l2_reg)(weights['deep_%s' % i])

        self.learning_rate_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                              self.decay_rate,
                                                              staircase=True)
        # optimizer
        self.optimizer = tf.train.FtrlOptimizer(learning_rate=self.learning_rate_decay).minimize(self.loss, global_step=self.global_step)
        self.out = tf.identity(self.out, name="score")

    def fit(self, train_data, predict_data):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            losses = []
            num_samples = 0
            for epoch in range(self.epochs):
                st = time.time()
                for i in range(len(train_data)):
                    data_batch = pickle.loads(train_data[i])
                    feed_dict = {
                        self.cont_feats: data_batch["cont_feats"],
                        self.cate_feats: data_batch["cate_feats"],
                        self.wide_feats: data_batch["wide_feats"],
                        self.input_data_len: [len(data_batch["cont_feats"])],
                        self.label: data_batch["labels"]
                    }

                    loss_train, op = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
                    sys.stdout.flush()
                    losses.append(loss_train * self.batch_size)
                    num_samples += self.batch_size

                end_time = time.time()
                total_loss = float(np.sum(losses) / num_samples)
                valid_metric = self.evaluate(sess, predict_data)
                print('[%s] valid-%s=%.5f\tloss=%.5f [%.1f s]' % (
                    epoch + 1, self.metric_type, valid_metric, total_loss, end_time - st))
                sys.stdout.flush()

            # **************************保存为pb模型******************************
            model_signature = signature_def_utils.build_signature_def(
                inputs={
                    "cont_feats": utils.build_tensor_info(self.cont_feats),
                    "cate_feats": utils.build_tensor_info(self.cate_feats),
                    "wide_feats": utils.build_tensor_info(self.wide_feats),
                    "input_data_size": utils.build_tensor_info(self.input_data_len)
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
        predict_list = []
        label_list = []
        for i in range(len(data_val)):
            data_batch = pickle.loads(data_val[i])
            feed_dict = {
                self.cont_feats: data_batch["cont_feats"],
                self.cate_feats: data_batch["cate_feats"],
                self.wide_feats: data_batch["wide_feats"],
                self.input_data_len: [len(data_batch["cont_feats"])],
            }
            label_list.extend(data_batch["labels"])
            y_predict = sess.run(self.out, feed_dict=feed_dict)
            predict_list.extend(y_predict)

        return roc_auc_score(label_list, predict_list)

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
                "cont_feats:0": data_batch["cont_feats"],
                "cate_feats:0": data_batch["cate_feats"],
                "wide_feats:0": data_batch["wide_feats"],
                "input_data_len:0": [len(data_batch["cont_feats"])]
            }
            var = session.run("score:0", feed_dict=feed_dict_map)
            val_predict = var[:, 0]
            label_list.extend(data_batch["labels"])
            pred_list.extend(val_predict)
        print("val of auc:%.5f" % roc_auc_score(label_list, pred_list))
        sys.stdout.flush()
        print('---end---')
