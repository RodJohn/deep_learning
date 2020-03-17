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
        self.dropout_keep_fm = [1, 1]
        self.embedding_size = args.embedding_size
        self.cont_field_size = args.cont_field_size
        self.vector_field_size = args.vector_feats_size
        self.cate_field_size = args.cate_field_size
        self.cate_feats_size = args.cate_feats_size
        self.use_fm = True
        self.use_deep = True

        self.__init__graph()

    def __init__graph(self):
        tf.set_random_seed(self.random_seed)
        self.label = tf.placeholder(tf.float32, [None, 1], name='label')
        self.cont_feats = tf.placeholder(tf.float32, [None, self.cont_field_size], name='cont_feats')
        self.vector_feats = tf.placeholder(tf.float32, [None, self.vector_field_size], name='vector_feats')
        self.cate_feats = tf.placeholder(tf.int32, [None, self.cate_field_size], name='cate_feats')
        self.input_data_size = tf.placeholder(tf.int32, [1], name="input_data_size")

        self.global_step = tf.Variable(0, trainable=False)
        self.weights = {}
        self.biases = {}

        print(self.cont_feats)
        print(self.vector_feats)
        print(self.cate_feats)
        print(self.input_data_size)

    def deep_fm(self):
        index_max_size = self.cont_field_size + self.cate_feats_size
        self.weights['feats_emb'] = tf.Variable(tf.random_normal([index_max_size, self.embedding_size], 0.0, 0.01),
                                                name='feats_emb_weight')
        self.weights['feats'] = tf.Variable(tf.random_uniform([index_max_size, 1], 0.0, 1.0), name='feats_weight')

        with tf.name_scope('fm_part'):
            cont_feats_index = tf.tile(
                tf.Variable([[i for i in range(self.cont_field_size)]], trainable=False, name="cont_feats_index"),
                multiples=[self.input_data_size[0], 1])
            cont_feats_index_add = tf.add(cont_feats_index, self.cate_field_size, name="cont_feats_index_add")
            cat_feats_value = tf.ones(shape=[self.input_data_size[0], self.cate_field_size], dtype=tf.float32,
                                      name="cat_feats_value")
            print(cont_feats_index)
            print(cont_feats_index_add)
            print(cat_feats_value)

            input_feats_index = tf.concat([self.cate_feats, cont_feats_index_add], axis=1)
            input_feats_value = tf.concat([cat_feats_value, self.cont_feats], axis=1)
            input_feats_field_size = self.cate_field_size + self.cont_field_size

            # FM_first_order [?, self.input_feats_field_size]
            first_order_emb = tf.nn.embedding_lookup(self.weights['feats'], ids=input_feats_index)
            first_order_emb = tf.reshape(first_order_emb, shape=[-1, input_feats_field_size])
            first_order_mul = tf.multiply(first_order_emb, input_feats_value)
            first_order = tf.nn.dropout(first_order_mul, self.dropout_keep_fm[0])
            print("fm_first_order:", first_order)

            # FM_second_order [?, embedding_size]
            second_order_emb = tf.nn.embedding_lookup(self.weights['feats_emb'], ids=input_feats_index)
            input_feats_value_reshape = tf.reshape(input_feats_value, shape=[-1, input_feats_field_size, 1])
            second_order_emb = tf.multiply(second_order_emb, input_feats_value_reshape)
            sum_feats_emb = tf.reduce_sum(second_order_emb, 1)
            sum_square_feats_emb = tf.square(sum_feats_emb)
            square_feats_emb = tf.square(second_order_emb)
            square_sum_feats_emb = tf.reduce_sum(square_feats_emb, 1)
            second_order = 0.5 * tf.subtract(sum_square_feats_emb, square_sum_feats_emb)
            second_order = tf.nn.dropout(second_order, self.dropout_keep_fm[1])
            print("fm_second_order:", second_order)

        with tf.name_scope('deep_part'):
            # category -> Embedding
            cat_emb = tf.nn.embedding_lookup(self.weights['feats_emb'], ids=self.cate_feats)
            cat_emb = tf.reshape(cat_emb, shape=[-1, self.cate_field_size * self.embedding_size])
            dense_vector = tf.concat([self.cont_feats, self.vector_feats, cat_emb], axis=1, name='dense_vector')
            cat_size = cat_emb.shape.as_list()[1]
            input_size = self.cont_field_size + self.vector_field_size + cat_size
            print("cat_emb size:", cat_size)
            print("model_input_size = ", input_size)

            glorot = np.sqrt(2.0 / (input_size + self.hidden_units[0]))
            self.weights['deep_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(input_size, self.hidden_units[0])), dtype=np.float32)
            self.biases['deep_bias_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_units[0])), dtype=np.float32)

            len_layers = len(self.hidden_units)
            for i in range(1, len_layers):
                glorot = np.sqrt(2.0 / (self.hidden_units[i - 1] + self.hidden_units[i]))
                self.weights['deep_%s' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.hidden_units[i - 1], self.hidden_units[i])),
                    dtype=np.float32)
                self.biases['deep_bias_%s' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_units[i])),
                    dtype=np.float32)

            deep_res = tf.nn.dropout(dense_vector, self.dropout_keep_deep[0])
            for i in range(0, len_layers):
                deep_res = tf.add(tf.matmul(deep_res, self.weights['deep_%s' % i]), self.biases['deep_bias_%s' % i])
                deep_res = tf.nn.relu(deep_res)
                deep_res = tf.nn.dropout(deep_res, self.dropout_keep_deep[i + 1])

        with tf.name_scope('deep_fm'):
            if self.use_fm and self.use_deep:
                feats_input = tf.concat([first_order, second_order, deep_res], axis=1)
                feats_input_size = input_feats_field_size + self.embedding_size + self.hidden_units[-1]
            elif self.use_fm:
                feats_input = tf.concat([first_order, second_order], axis=1)
                feats_input_size = input_feats_field_size + self.embedding_size
            elif self.use_deep:
                feats_input = deep_res
                feats_input_size = self.hidden_units[-1]

            glorot = np.sqrt(2.0 / (feats_input_size + 1))
            self.weights['deep_fm_weight'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(feats_input_size, 1)), dtype=np.float32)
            self.biases['deep_fm_bias'] = tf.Variable(tf.random_normal([1]))
            self.out = tf.add(tf.matmul(feats_input, self.weights['deep_fm_weight']), self.biases['deep_fm_bias'])

        self.score = tf.nn.sigmoid(self.out, name='score')

    def model_optimizer(self):
        self.deep_fm()

        # loss
        self.loss = tf.losses.log_loss(self.label, self.score)
        self.loss = tf.reduce_mean(self.loss)

        # l2 regularization on weights
        if self.l2_reg > 0:
            self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['deep_fm_weight'])

        # optimizer
        self.learning_rate_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                              self.decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_decay).minimize(self.loss,
                                                                                                 global_step=self.global_step)

    def fit(self, train_data, val_data):
        self.model_optimizer()
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
                        self.vector_feats: data_batch["vector_feats"],
                        self.cate_feats: data_batch["cate_feats"],
                        self.input_data_size: [len(data_batch["cont_feats"])],
                        self.label: data_batch["labels"]
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
                    "cont_feats": utils.build_tensor_info(self.cont_feats),
                    "vector_feats": utils.build_tensor_info(self.vector_feats),
                    "cate_feats": utils.build_tensor_info(self.cate_feats),
                    "input_data_size": utils.build_tensor_info(self.input_data_size)
                },
                outputs={"output": utils.build_tensor_info(self.score)},
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
                self.cont_feats: data_batch["cont_feats"],
                self.vector_feats: data_batch["vector_feats"],
                self.cate_feats: data_batch["cate_feats"],
                self.input_data_size: [len(data_batch["cont_feats"])],
                self.label: data_batch["labels"]
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
        predict_list = []
        label_list = []
        for i in range(len(data_val)):
            data_batch = pickle.loads(data_val[i])
            feed_dict = {
                "cont_feats:0": data_batch["cont_feats"],
                "vector_feats:0": data_batch["vector_feats"],
                "cate_feats:0": data_batch["cate_feats"],
                "input_data_size:0": [len(data_batch["cont_feats"])]
            }
            var = session.run("score:0", feed_dict=feed_dict)
            val_predict = var[:, 0]
            label_list.extend(data_batch["labels"])
            predict_list.extend(val_predict)
        auc = roc_auc_score(label_list, predict_list)
        print("val of auc:%.5f" % auc)
        sys.stdout.flush()
