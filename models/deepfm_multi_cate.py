#!/usr/bin/env python
# coding=utf-8

import os
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)


class DeepModel:
    def __init__(self, args, data_dict):
        self.hidden_units = args.hidden_units
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.model_pb = args.model_pb
        self.save_model_checkpoint = args.save_model_checkpoint
        self.restore_model_checkpoint = args.restore_model_checkpoint
        self.model_restore = args.model_restore
        self.decay_steps = args.learning_rate_decay_steps
        self.decay_rate = args.learning_rate_decay_rate
        self.l2_reg = args.l2_reg
        self.metric_type = "auc"
        self.random_seed = 2019
        self.dropout_keep_deep = [1, 1, 1, 1, 1]
        self.dropout_keep_fm = [1, 1]
        self.embedding_size = args.embedding_size
        self.single_cate_field_size = args.cate_field_size
        self.vector_feats_size = args.vector_feats_size
        self.cate_index_size = args.cate_feats_size
        self.multi_cate_feats_size = args.multi_feats_size
        self.multi_cate_field_size = args.multi_field_size
        self.multi_cate_range = args.multi_feats_range
        self.use_fm = True
        self.use_deep = True
        self.data_dict = data_dict

        self.__init__graph()

    def __init__graph(self):
        tf.set_random_seed(self.random_seed)

        self.input_data_size = tf.placeholder_with_default([self.batch_size], shape=[1], name='input_data_size')
        self.labels = tf.placeholder_with_default(self.data_dict['label'], shape=[None, 1], name='labels')
        self.cate_feats = tf.placeholder_with_default(self.data_dict['cate_feats'],
                                                      shape=[None,
                                                             self.single_cate_field_size + self.multi_cate_feats_size],
                                                      name='cate_feats')
        self.vector_feats = tf.placeholder_with_default(self.data_dict['vector_feats'],
                                                        shape=[None, self.vector_feats_size],
                                                        name='vector_feats')

        self.single_cate_feats = self.cate_feats[:, 0:self.single_cate_field_size]
        self.multi_cate_feats = self.cate_feats[:, self.single_cate_field_size:]

        self.global_step = tf.Variable(0, trainable=False)
        self.weights = {}
        self.biases = {}

        print(self.input_data_size)
        print(self.cate_feats)
        print(self.vector_feats)
        print(self.single_cate_feats)
        print(self.multi_cate_feats)

    def multi_cate_emb(self):
        # multi_category -> Embedding -> Reduce_Mean
        def nonzero_reduce_mean(emb):
            axis_2_sum = tf.reduce_sum(emb, axis=2)
            multi_cate_nonzero = tf.count_nonzero(axis_2_sum, 1, keepdims=True, dtype=float)
            multi_cate_sum = tf.reduce_sum(emb, axis=1)
            reduce_mean_emb = tf.div_no_nan(multi_cate_sum, multi_cate_nonzero)
            return reduce_mean_emb

        for item in self.multi_cate_range:
            index_start = int(item[0])
            index_end = int(item[1])
            feat_name = item[2]
            print("index_start:", index_start)
            print("index_end:", index_end)
            print("feat_name:", feat_name)

            # first_multi_cate_emb is for FM-first_order
            first_multi_cate_emb = tf.nn.embedding_lookup(self.weights['fm_first_order_emb'],
                                                          ids=self.multi_cate_feats[:, index_start:index_end])
            # multi_cate_emb is for FM-second_order and DNN
            multi_cate_emb = tf.nn.embedding_lookup(self.weights['feats_emb'],
                                                    ids=self.multi_cate_feats[:, index_start:index_end])

            first_multi_emb = nonzero_reduce_mean(first_multi_cate_emb)
            multi_emb = nonzero_reduce_mean(multi_cate_emb)

            print("first_multi_cate_emb_lookup:", feat_name, first_multi_cate_emb)
            print("first_multi_cate_reduce_mean:", feat_name, first_multi_emb)
            print("multi_cate_emb_lookup:", feat_name, multi_cate_emb)
            print("multi_cate_reduce_mean:", feat_name, multi_emb)

            if int(item[0]) == 0:
                self.all_first_multi_emb = first_multi_emb
                self.all_multi_cate_emb = multi_emb
            else:
                self.all_first_multi_emb = tf.concat([self.all_first_multi_emb, first_multi_emb], axis=1)
                self.all_multi_cate_emb = tf.concat([self.all_multi_cate_emb, multi_emb], axis=1)

            print("all_first_multi_emb:", self.all_first_multi_emb)
            print("all_multi_cate_emb:", self.all_multi_cate_emb)

    def deep_fm(self):
        index_max_size = self.cate_index_size
        self.weights['feats_emb'] = tf.Variable(
            tf.random_normal([index_max_size, self.embedding_size], 0.0, 0.01), name='feats_emb')
        self.weights['fm_first_order_emb'] = tf.Variable(tf.random_uniform([index_max_size, 1], 0.0, 1.0),
                                                         name='fm_first_order_emb')

        self.weights['feats_emb'] = tf.concat((tf.zeros(shape=[1, self.embedding_size]),
                                               self.weights['feats_emb'][1:]), 0)
        self.weights['fm_first_order_emb'] = tf.concat((tf.zeros(shape=[1, 1]),
                                                        self.weights['fm_first_order_emb'][1:]), 0)

        self.multi_cate_emb()

        with tf.name_scope('fm_part'):
            input_field_size = self.single_cate_field_size + self.multi_cate_field_size

            # FM_first_order [?, input_field_size]
            # single_cate
            first_single_cate_emb = tf.nn.embedding_lookup(self.weights['fm_first_order_emb'],
                                                           ids=self.single_cate_feats)
            first_single_cate_emb = tf.reshape(first_single_cate_emb, shape=[-1, self.single_cate_field_size])
            # concat & single_cate & multi_cate
            first_order_emb = tf.concat([first_single_cate_emb, self.all_first_multi_emb], axis=1)
            first_order = tf.nn.dropout(first_order_emb, self.dropout_keep_fm[0])
            print("fm_first_order:", first_order)

            # FM_second_order [?, embedding_size]
            # single_cate
            second_single_cate_emb = tf.nn.embedding_lookup(self.weights['feats_emb'], ids=self.single_cate_feats)
            # multi_cate
            second_multi_cate_emb = tf.reshape(self.all_multi_cate_emb,
                                               shape=[-1, self.multi_cate_field_size, self.embedding_size])
            # concat single_cate & multi_cate
            second_order_emb = tf.concat([second_single_cate_emb, second_multi_cate_emb], axis=1)

            sum_emb = tf.reduce_sum(second_order_emb, 1)
            sum_square_emb = tf.square(sum_emb)
            square_emb = tf.square(second_order_emb)
            square_sum_emb = tf.reduce_sum(square_emb, 1)
            second_order = 0.5 * tf.subtract(sum_square_emb, square_sum_emb)
            second_order = tf.nn.dropout(second_order, self.dropout_keep_fm[1])
            print("fm_second_order:", second_order)

            # FM_res [?, self.input_field_size + embedding_size]
            self.fm_res = tf.concat([first_order, second_order], axis=1)
            print("fm_res:", self.fm_res)

        with tf.name_scope('deep_part'):
            len_layers = len(self.hidden_units)
            # single_category -> Embedding
            self.single_cate_emb = tf.nn.embedding_lookup(self.weights['feats_emb'], ids=self.single_cate_feats)
            self.single_cate_emb = tf.reshape(self.single_cate_emb,
                                              shape=[-1, self.single_cate_field_size * self.embedding_size])

            # concat vector & single_category_embedding & multi_category_embedding -> Dense Vector
            dense_vector = tf.concat([self.vector_feats, self.single_cate_emb,
                                      self.all_multi_cate_emb],
                                     axis=1, name='dense_vector')
            single_cate_emb_size = self.single_cate_emb.shape.as_list()[1]
            multi_cate_emb_size = self.all_multi_cate_emb.shape.as_list()[1]
            input_size = self.vector_feats_size + single_cate_emb_size + multi_cate_emb_size
            print("vector_size = ", self.vector_feats_size)
            print("single_cate_emb_size = ", single_cate_emb_size)
            print("multi_cate_emb_size:", multi_cate_emb_size)
            print("deep_model_input_size = ", input_size)

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

            self.deep_res = tf.nn.dropout(dense_vector, self.dropout_keep_deep[0])
            for i in range(0, len_layers):
                self.deep_res = tf.add(tf.matmul(self.deep_res, self.weights['deep_%s' % i]),
                                       self.biases['deep_bias_%s' % i])
                self.deep_res = tf.nn.relu(self.deep_res)
                self.deep_res = tf.nn.dropout(self.deep_res, self.dropout_keep_deep[i + 1])

        with tf.name_scope('deep_fm'):
            if self.use_fm and self.use_deep:
                feats_input = tf.concat([self.fm_res, self.deep_res], axis=1)
                feats_input_size = input_field_size + self.embedding_size + self.hidden_units[-1]
            elif self.use_fm:
                feats_input = self.fm_res
                feats_input_size = input_field_size + self.embedding_size
            elif self.use_deep:
                feats_input = self.deep_res
                feats_input_size = self.hidden_units[-1]

            glorot = np.sqrt(2.0 / (feats_input_size + 1))
            self.weights['deep_fm_weight'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(feats_input_size, 1)), dtype=np.float32)
            self.biases['deep_fm_bias'] = tf.Variable(tf.random_normal([1]))

            out = tf.add(tf.matmul(feats_input, self.weights['deep_fm_weight']), self.biases['deep_fm_bias'])

        self.score = tf.nn.sigmoid(out, name='score')
        print(self.score)

    def model_optimizer(self):
        self.deep_fm()
        # loss
        loss = tf.losses.log_loss(self.labels, self.score)
        loss = tf.reduce_mean(loss)
        # l2 regularization on weights
        if self.l2_reg > 0:
            loss = loss + tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['deep_fm_weight'])
        learning_rate_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                         self.decay_rate, staircase=True)

        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_decay).minimize(loss,
                                                                                       global_step=self.global_step)

        return loss, optimizer, self.global_step

    def fit(self, print_num_batch, predict_data):
        op = self.model_optimizer()
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            saver = tf.train.Saver(max_to_keep=1)
            if self.model_restore == 1:
                try:
                    latest_checkpoint = tf.train.latest_checkpoint(self.restore_model_checkpoint)
                    saver.restore(sess=sess, save_path=latest_checkpoint)
                    print("@_@~ Old Model Restored Successfully!")
                except Exception as e:
                    print("=_=!! Error: There is no model checkpoint in %s" % self.restore_model_checkpoint)
                    print(e)
                    exit(-1)

            # 获取验证(预测)集数据
            val_data = self.get_val_data(sess, predict_data)

            print("Start of training")
            start_time = time.time()
            batch_count = 0
            i = 0
            try:
                while True:
                    loss, _, _ = sess.run(op)

                    if batch_count == print_num_batch:
                        batch_end_time = time.time()
                        auc = self.eval(sess, val_data)
                        print("[{}] val_auc:{}\t loss:{} time:{:.2f}s".format(i,
                                                                              auc,
                                                                              loss,
                                                                              batch_end_time - start_time))
                        sys.stdout.flush()
                        start_time = time.time()
                        batch_count = 0

                    batch_count += 1
                    i += 1

            except tf.errors.OutOfRangeError:
                print("--------------End of dataset-------------")

                print("--------------save checkpoint model-------------")
                checkpoint_path = os.path.join(self.save_model_checkpoint, 'model')
                saver.save(sess, checkpoint_path, global_step=self.global_step)

                print("--------------save pb model-------------")
                model_signature = signature_def_utils.build_signature_def(
                    inputs={
                        "cate_feats": utils.build_tensor_info(self.cate_feats),
                        "vector_feats": utils.build_tensor_info(self.vector_feats),
                        "input_data_size": utils.build_tensor_info(self.input_data_size)
                    },
                    outputs={"score": utils.build_tensor_info(self.score)},
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

    @staticmethod
    def get_val_data(sess, data):
        cate_index_list = []
        vector_value_list = []
        labels = []
        data_val = []
        try:
            while True:
                cate_index, vector_value, label = sess.run(
                    [
                        data["cate_feats"],
                        data["vector_feats"],
                        data["label"]
                    ])
                cate_index_list.append(cate_index)
                vector_value_list.append(vector_value)
                labels.append(label)
        except tf.errors.OutOfRangeError:
            data_val.append(labels)
            data_val.append(cate_index_list)
            data_val.append(vector_value_list)
            return data_val

    def eval(self, sess, val_data):
        """
            验证tfrecord类型的数据
        """
        predict_list = []
        label_list = []
        for i in range(len(val_data[0])):
            feed_dict_map = {
                self.cate_feats: val_data[1][i],
                self.vector_feats: val_data[2][i],
                self.input_data_size: [len(val_data[0][i])]
            }
            var = sess.run("score:0", feed_dict=feed_dict_map)
            val_score = var[:, 0]
            label_list.extend(val_data[0][i])
            predict_list.extend(val_score)
        return roc_auc_score(label_list, predict_list)


def predict(predict_data, model_pb):
    """
        加载pb模型,预测tfrecord类型的数据
    """
    with tf.Session() as sess:
        model_sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(model_sess, [tf.saved_model.tag_constants.SERVING], model_pb)
        predict_list = []
        label_list = []
        try:
            while True:
                cate_index, vector_value, label = sess.run(
                    [
                        predict_data["cate_feats"],
                        predict_data["vector_feats"],
                        predict_data["label"]
                    ])
                feed_dict_map = {
                    "cate_feats:0": cate_index,
                    "vector_feats:0": vector_value,
                    "input_data_size:0": [len(label)]
                }
                var = model_sess.run("score:0", feed_dict=feed_dict_map)
                predict_score = var[:, 0]
                label_list.extend(label)
                predict_list.extend(predict_score)
        except tf.errors.OutOfRangeError:
            print("-----------end of data_set-----------")
            print("val of auc:%.5f" % roc_auc_score(label_list, predict_list))
            sys.stdout.flush()
            print('---end---')
