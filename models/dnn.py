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
        self.cont_field_size = args.cont_field_size + args.vector_feats_size
        self.cate_field_size = args.cate_field_size
        self.cate_feats_size = args.cate_feats_size
        self.embedding_size = args.embedding_size

        self.__init_graph()

    def __init_graph(self):
        tf.set_random_seed(self.random_seed)
        self.cate_feats = tf.placeholder(tf.int32, [None, self.cate_field_size], name='cate_feats')
        self.cont_feats = tf.placeholder(tf.float32, [None, self.cont_field_size], name='cont_feats')
        print(self.cate_feats)
        print(self.cont_feats)
        self.label = tf.placeholder(tf.float32, [None, 1], name='label')
        self.global_step = tf.Variable(0, trainable=False)

        weights = {}
        biases = {}

        with tf.name_scope('deep_part'):
            len_layers = len(self.hidden_units)
            embeddings = tf.get_variable('weight_mat',
                                         dtype=tf.float32,
                                         shape=(self.cate_feats_size, self.embedding_size),
                                         initializer=tf.contrib.layers.xavier_initializer())
            # category -> Embedding
            cate_emb = tf.nn.embedding_lookup(embeddings, ids=self.cate_feats)
            cate_emb = tf.reshape(cate_emb, shape=[-1, self.cate_field_size * self.embedding_size])
            # concat Embedding Vector & continuous -> Dense Vector
            dense_vector = tf.concat([self.cont_feats, cate_emb], axis=1, name='dense_vector')

            input_size = dense_vector.shape.as_list()[1]
            glorot = np.sqrt(2.0 / (input_size + self.hidden_units[0]))
            weights['deep_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, self.hidden_units[0])), dtype=np.float32)
            biases['deep_bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_units[0])), dtype=np.float32)

            for i in range(1, len_layers):
                glorot = np.sqrt(2.0 / (self.hidden_units[i - 1] + self.hidden_units[i]))
                weights['deep_%s' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.hidden_units[i - 1], self.hidden_units[i])),
                    dtype=np.float32)
                biases['deep_bias_%s' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_units[i])),
                    dtype=np.float32)

            deep_res = tf.nn.dropout(dense_vector, self.dropout_keep_deep[0])
            for i in range(0, len_layers):
                deep_res = tf.add(tf.matmul(deep_res, weights['deep_%s' % i]), biases['deep_bias_%s' % i])
                deep_res = tf.nn.relu(deep_res)
                deep_res = tf.nn.dropout(deep_res, self.dropout_keep_deep[i + 1])

            glorot = np.sqrt(2.0 / (self.hidden_units[-1] + 1))
            weights['deep_res'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.hidden_units[-1], 1)), dtype=np.float32)
            biases['deep_res_bias'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, 1)), dtype=np.float32)
            self.out = tf.add(tf.matmul(deep_res, weights['deep_res']), biases['deep_res_bias'])

        self.out = tf.nn.sigmoid(self.out)
        self.loss = tf.losses.log_loss(self.label, self.out)
        self.loss = tf.reduce_mean(self.loss)

        if self.l2_reg > 0:
            for i in range(len(self.hidden_units)):
                self.loss = self.loss + tf.contrib.layers.l1_regularizer(self.l2_reg)(weights['deep_%s' % i])
        self.learning_rate_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_decay).minimize(self.loss, global_step=self.global_step)
        self.out = tf.identity(self.out, name="score")
        # print(self.out)
        sys.stdout.flush()

    def fit(self, train_data, val_data):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            losses = []
            num_samples = 0
            for epoch in range(self.epochs):
                st = time.time()
                for i in range(len(train_data)):
                    data_batch = pickle.loads(train_data[i])
                    # data_batch = train_data[i]
                    feed_dict = {
                        self.cont_feats: data_batch["cont_feats"],
                        self.cate_feats: data_batch["cate_feats"],
                        self.label: data_batch["labels"]
                    }

                    loss_train, op = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
                    sys.stdout.flush()
                    losses.append(loss_train * self.batch_size)
                    num_samples += self.batch_size

                end_time = time.time()
                total_loss = float(np.sum(losses) / num_samples)
                valid_metric = self.evaluate(sess, val_data)
                print('[%s] valid-%s=%.5f\tloss=%.5f [%.1f s]' % (epoch + 1, self.metric_type, valid_metric, total_loss, end_time - st))
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                sys.stdout.flush()

            # **************************保存为pb模型******************************
            model_signature = signature_def_utils.build_signature_def(
                inputs={
                    "cont_feats": utils.build_tensor_info(self.cont_feats),
                    "cat_feats": utils.build_tensor_info(self.cate_feats),
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
            # data_batch = data_val[i]
            feed_dict = {
                self.cont_feats: data_batch["cont_feats"],
                self.cate_feats: data_batch["cate_feats"],
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
        predict_list = []
        label_list = []
        for i in range(len(data_val)):
            data_batch = pickle.loads(data_val[i])
            # data_batch = data_val[i]
            feed_dict_map = {
                "cont_feats:0": data_batch["cont_feats"],
                "cate_feats:0": data_batch["cate_feats"],
            }
            var = session.run("score:0", feed_dict=feed_dict_map)
            val_predict = var[:, 0]
            print(val_predict)
            label_list.extend(data_batch["labels"])
            predict_list.extend(val_predict)
        print("val of auc:%.5f" % roc_auc_score(label_list, predict_list))
        sys.stdout.flush()
