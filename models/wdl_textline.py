#!/usr/bin/env python
# coding=utf-8

import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


class DeepModel:
    def __init__(self, args, data_dict):
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
        self.cate_index_size = args.cate_index_size
        self.embedding_size = args.embedding_size
        self.cate_feats_conf_path = args.cate_feats_conf_path
        # self.word2vec_embedding_path = args.word2vec_embedding_path
        # self.mul_cat_index_range, self.mul_cat_field_size = self.get_mul_index_range()
        self.data_dict = data_dict

        self.wide_feats_field_size = args.wide_field_size
        self.wide_feats_index_size = self.cate_index_size
        self.use_wide = True
        self.use_deep = True
        with tf.variable_scope("weight_matrix"):
            embeddings = tf.get_variable('weight_mat',
                                         dtype=tf.float32,
                                         shape=(self.cate_index_size, self.embedding_size),
                                         initializer=tf.contrib.layers.xavier_initializer())

            self.embeddings = tf.concat((tf.zeros(shape=[1, self.embedding_size]), embeddings[1:, :]), 0)

        self.__init_graph()
        self.model_optimizer()

    def __init_graph(self):
        tf.set_random_seed(self.random_seed)
        self.label = tf.identity(self.data_dict["labels"], name="labels")

        self.cont_feats = tf.identity(self.data_dict["cont_feats"], name="cont_feats")
        self.cate_feats = tf.identity(self.data_dict["cate_feats"], name="cate_feats")
        self.wide_feats = tf.identity(self.data_dict["wide_feats"], name="wide_feats")
        # self.mul_cate_feats = tf.identity(self.data_dict["mul_cate_feats"], name="mul_cate_feats")
        # self.mul_cate_feats_value = tf.identity(self.data_dict["mul_cate_feats_value"], name="mul_cate_feats_value")
        self.input_data_size = self.batch_size
        self.deep_feature_index = tf.tile(
            tf.Variable([[i for i in range(self.hidden_units[-1])]], trainable=False, name="deep_feature_index"),
            multiples=[self.input_data_size, 1])
        self.wide_feats_value = tf.ones(shape=[self.input_data_size, self.wide_feats_field_size], dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.weights = {}
        self.biases = {}

        print(self.wide_feats_value)
        print(self.cate_feats)
        print(self.cont_feats)
        print(self.wide_feats)
        print(self.input_data_size)
        print(self.deep_feature_index)
        # print(self.mul_cate_feats)
        # print(self.mul_cate_feats_value)

    def get_mul_index_range(self):
        mul_cat_index_list = list()
        if self.cate_feats_conf_path == '':
            print("feature_conf_path is None")
            return mul_cat_index_list

        index_start = 0
        with open(self.cate_feats_conf_path, 'r') as f:
            for line in f.readlines():
                line_data = line.strip()
                if line_data == '':
                    continue

                config_arr = line_data.split("\t")
                result_type = config_arr[6]
                if result_type != 'top':
                    continue

                result_parameter = config_arr[7]
                feature_name = config_arr[-1]
                topN = int(result_parameter.strip().split("=")[1])
                index_end = index_start + topN
                index_range = [index_start, index_end, feature_name]
                mul_cat_index_list.append(index_range)

                index_start = index_end

        mul_cat_field_size = index_start
        return mul_cat_index_list, mul_cat_field_size

    def wide_func(self):
        """
            LR模型部分
        """
        with tf.name_scope('wide_part'):
            if self.use_wide is True and self.use_deep is False:
                glorot = np.sqrt(2.0 / (self.wide_feats_index_size + 1))
                self.wide_weights = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.wide_feats_index_size, 1)), dtype=np.float32)
                # self.wide_weights = tf.concat((tf.zeros(shape=[1, 1]), self.wide_weights[1:, 1]), 0)

                self.weights["wide_w"] = tf.nn.embedding_lookup(self.wide_weights, ids=self.wide_feats)
                self.weights["wide_w"] = tf.reshape(self.weights["wide_w"], shape=[-1, self.wide_feats_field_size])
                self.weights["wide_b"] = tf.Variable(tf.random_normal([1]))
                self.wide_res = tf.add(
                    tf.reduce_sum(tf.multiply(self.weights["wide_w"], self.wide_feats_value), axis=1, keep_dims=True),
                    self.weights["wide_b"])
            # elif self.use_wide is True and self.use_deep is True:
            #     self.wide_feature_index_add = tf.add(self.wide_feature_index, self.hidden_units[-1])
            else:
                pass

    def deep_func(self):
        """
            DNN模型部分
        """
        with tf.name_scope('deep_part'):

            len_layers = len(self.hidden_units)

            # category -> Embedding
            self.cat_emb = tf.nn.embedding_lookup(self.embeddings, ids=self.cate_feats)
            self.cat_emb = tf.reshape(self.cat_emb, shape=[-1, self.cate_field_size * self.embedding_size])

            # # mul_category -> Embedding
            # print("load word2vec embedding")
            # emb = np.load(self.word2vec_embedding_path)
            # self.word2vecEmbedding = tf.constant(emb, dtype=np.float32, name='tags_emb')

            # if len(self.mul_cat_index_range) != 0:
            #     for item in self.mul_cat_index_range:
            #         index_start = int(item[0])
            #         index_end = int(item[1])
            #         feat_name = item[2]
            #         print("index_start:", index_start)
            #         print("index_end:", index_end)
            #         print("feat_name:", feat_name)
            #         if feat_name == 'tag':
            #             self.mul_cat_emb = tf.nn.embedding_lookup(self.word2vecEmbedding,
            #                                                       ids=self.mul_cate_feats[:, index_start:index_end])
            #             print("tag------word2vecEmbedding")
            #         else:
            #             self.mul_cat_emb = tf.nn.embedding_lookup(self.embeddings,
            #                                                       ids=self.mul_cate_feats[:, index_start:index_end])
            #
            #         print("mul_cat_emb_lookup:", item[-1], self.mul_cat_emb)
            #         axis_2_sum = tf.reduce_sum(self.mul_cat_emb, axis=2)
            #         mul_cat_nonzero = tf.count_nonzero(axis_2_sum, 1, keepdims=True, dtype=float)
            #
            #         mul_cat_emb_weight = tf.reshape(self.mul_cate_feats_value[:, index_start:index_end],
            #                                         [-1, index_end - index_start, 1])
            #         weighted_mul_cat_emb = tf.multiply(self.mul_cat_emb, mul_cat_emb_weight)
            #         print("mul_cat_emb_weight:", item[-1], mul_cat_emb_weight)
            #         print("weighted_mul_cat_emb:", item[-1], weighted_mul_cat_emb)
            #         mul_cat_sum = tf.reduce_sum(weighted_mul_cat_emb, axis=1)
            #         self.mul_cat_emb = tf.div_no_nan(mul_cat_sum, mul_cat_nonzero)
            #         # self.mul_cat_emb = tf.reduce_mean(self.mul_cat_emb, axis=1)
            #         print("mul_cat_reduce_mean:", item[-1], self.mul_cat_emb)
            #
            #         if index_start == 0:
            #             self.all_mul_cat_emb = self.mul_cat_emb
            #         else:
            #             self.all_mul_cat_emb = tf.concat([self.all_mul_cat_emb, self.mul_cat_emb], axis=1)
            #         print("all_mul_cat_emb:", self.all_mul_cat_emb)
            #
            # # concat Embedding Vector & continuous -> Dense Vector
            # self.dense_vector = tf.concat([self.cont_feats, self.cat_emb, self.all_mul_cat_emb], axis=1,
            #                               name='dense_vector')
            self.dense_vector = tf.concat([self.cont_feats, self.cat_emb], axis=1,
                                          name='dense_vector')

            cat_size = self.cat_emb.shape.as_list()[1]
            # mul_cat_size = self.all_mul_cat_emb.shape.as_list()[1]
            print("cat_emb size:", cat_size)
            # print("all_mul_cat_emb size:", mul_cat_size)
            # input_size = self.cont_field_size + cat_size + mul_cat_size
            input_size = self.cont_field_size + cat_size
            print("model_input_size = ", input_size)

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

            if self.use_deep and self.use_wide is False:
                glorot = np.sqrt(2.0 / (self.hidden_units[-1] + 1))
                self.weights['deep_res'] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.hidden_units[-1], 1)), dtype=np.float32)
                self.biases['deep_res_bias'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, 1)),
                                                           dtype=np.float32)
                self.deep_res = tf.add(tf.matmul(self.deep_res, self.weights['deep_res']),
                                       self.biases['deep_res_bias'])
            elif self.use_wide is True and self.use_deep is True:
                self.deep_feats_index_add = tf.add(self.deep_feature_index, self.wide_feats_field_size)
            else:
                pass

    def wide_and_deep(self):
        """
            wide and deep 合并
        """
        print("wide deep:", self.use_wide, self.use_deep)
        self.wide_func()
        self.deep_func()

        with tf.name_scope('wide_deep'):
            if self.use_wide and self.use_deep:
                self.wdl_feats_index_size = self.wide_feats_index_size + self.hidden_units[-1]
                glorot = np.sqrt(2.0 / self.wdl_feats_index_size)
                self.weights["wdl_weights"] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.wdl_feats_index_size, 1)), dtype=np.float32)
                # self.weights["wdl_weights"] = tf.concat((tf.zeros(shape=[1, 1]), self.weights["wdl_weights"][1:, 1]), 0)
                self.biases['wdl_bias'] = tf.Variable(tf.random_normal([1]))

                # self.wdl_feature_index = tf.concat([self.deep_feature_index, self.wide_feature_index_add], axis=1)
                # self.wdl_feature_value = tf.concat([self.deep_res, self.wide_feature_value], axis=1)
                self.wdl_feature_index = tf.concat([self.wide_feats, self.deep_feats_index_add], axis=1)
                self.wdl_feature_value = tf.concat([self.wide_feats_value, self.deep_res], axis=1)
                self.weights["wdl_weights"] = tf.nn.embedding_lookup(self.weights["wdl_weights"],
                                                                     ids=self.wdl_feature_index)
                self.weights["wdl_weights"] = tf.reshape(self.weights["wdl_weights"],
                                                         shape=[-1,
                                                                self.wide_feats_field_size + self.hidden_units[-1]])

                self.out = tf.add(
                    tf.reduce_sum(tf.multiply(self.weights["wdl_weights"], self.wdl_feature_value), axis=1,
                                  keep_dims=True), self.biases['wdl_bias'])
            elif self.use_wide and self.use_deep is False:
                self.out = self.wide_res
            elif self.use_wide is False and self.use_deep:
                self.out = self.deep_res

    def model_optimizer(self):
        self.wide_and_deep()

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

        self.out = tf.identity(self.out, name="score")
        # sys.stdout.flush()
        # self.auc = tf.metrics.auc(labels=self.label, predictions=self.out, name='auc')

    def fit(self, print_iter, num_batch_size):
        pred_list = []
        label_list = []
        count = 0
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
            losses = []
            start_time = time.time()
            for i in range(10000000):
                try:
                    label, y_pred, loss = sess.run([self.label, self.out, self.loss])
                    losses.append(loss)
                    if i % print_iter == 0 and i > 0:
                        count += 1
                        label_list.extend(label)
                        pred_list.extend(y_pred)
                    if count == num_batch_size:
                        total_loss = float(np.sum(losses) / (count * print_iter))
                        auc = roc_auc_score(label_list, pred_list)
                        end_time = time.time()
                        print(
                            "[{}] test auc:{}\t loss:{} time:{:.2f}s".format(i, auc, total_loss, end_time - start_time))
                        sys.stdout.flush()
                        start_time = time.time()
                        pred_list = []
                        label_list = []
                        losses = []
                        count = 0
                    else:
                        _, _ = sess.run([self.loss, self.optimizer])

                except tf.errors.OutOfRangeError:
                    print("--------------End of Dataset-------------")
                    # **************************保存为pb模型******************************
                    model_signature = signature_def_utils.build_signature_def(
                        inputs={
                            "cont_feats": utils.build_tensor_info(self.cont_feats),
                            "cate_feats": utils.build_tensor_info(self.cate_feats),
                            # "mul_cate_feats": utils.build_tensor_info(self.mul_cate_feats),
                            # "mul_cate_feats_value": utils.build_tensor_info(self.mul_cate_feats_value)
                            "wide_feats": utils.build_tensor_info(self.wide_feats)
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
                    break


def predict(data_out, model_pb):
    """
        加载pb模型
    """
    session = tf.Session(graph=tf.Graph())
    tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], model_pb)
    pred_list = []
    label_list = []
    for i in range(len(data_out)):
        data_batch = pickle.loads(data_out[i])
        feed_dict_map = {
            "cont_feats:0": data_batch["cont_feats"],
            "cate_feats:0": data_batch["cate_feats"],
            # "mul_cate_feats:0": data_batch["mul_cate_feats"],
            # "mul_cate_feats_value:0": data_batch["mul_cate_feats_value"],
            "wide_feats:0": data_batch["wide_feats"]
        }
        var = session.run("score:0", feed_dict=feed_dict_map)
        val_pred = var[:, 0]
        label_list.extend(data_batch["labels"])
        pred_list.extend(val_pred)
    print("val of auc:%.5f" % roc_auc_score(label_list, pred_list))
    sys.stdout.flush()

    print('---end---')
