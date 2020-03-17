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
        self.use_attenton = args.use_attenton
        self.use_kg = args.use_kg
        self.use_bert = args.use_bert
        self.kernel_sizes = args.kernel_sizes
        self.out_filters = args.out_filters
        self.base_index = args.base_index
        self.cross_index = args.cross_index
        self.w2v_user_index = args.w2v_user_index
        self.w2v_item_index = args.w2v_item_index
        self.kg_user_index = args.kg_user_index
        self.kg_item_index = args.kg_item_index
        self.bert_user_index = args.bert_user_index
        self.bert_item_index = args.bert_item_index
        self.dropout_keep_deep = [1, 1, 1, 1, 1]
        self.metric_type = "auc"
        self.random_seed = 2019

        self.__init_graph()

    def __init_graph(self):
        tf.set_random_seed(self.random_seed)
        self.feature_values = tf.placeholder(tf.float32, [None, self.deep_feature_size], name='feature_values')
        self.input_data_size = tf.placeholder(tf.int32, [1], name="input_data_size")
        self.feature_indexs = tf.tile(
            tf.Variable([[i for i in range(self.hidden_units[-1])]], trainable=False, name="feature_indexs"),
            multiples=[self.input_data_size[0], 1])

        self.base_feature = self.feature_values[:, self.base_index[0]:self.base_index[1]]
        self.cross_feature = self.feature_values[:, self.cross_index[0]:self.cross_index[1]]
        self.w2v_user_embedding = self.feature_values[:, self.w2v_user_index[0]:self.w2v_user_index[1]]
        self.w2v_item_embedding = self.feature_values[:, self.w2v_item_index[0]:self.w2v_item_index[1]]
        self.kg_user_embedding = self.feature_values[:, self.kg_user_index[0]:self.kg_user_index[1]]
        self.kg_item_embedding = self.feature_values[:, self.kg_item_index[0]:self.kg_item_index[1]]
        self.bert_user_embedding = self.feature_values[:, self.bert_user_index[0]:self.bert_user_index[1]]
        self.bert_item_embedding = self.feature_values[:, self.bert_item_index[0]:self.bert_item_index[1]]

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

    def _attention(self,user_emb,item_emb):
        user_emb_cnn = self._kcnn(user_emb)
        item_emb_cnn = self._kcnn(item_emb)
        clicked_embeddings = tf.reshape(user_emb_cnn, shape=[-1, 1, self.out_filters * len(self.kernel_sizes)])
        news_embeddings_expanded = tf.expand_dims(item_emb_cnn, 1)
        attention_weights1 = tf.reduce_sum(clicked_embeddings * news_embeddings_expanded, axis=-1)
        attention_weights2 = tf.nn.softmax(attention_weights1, dim=-1)
        attention_weights_expanded = tf.expand_dims(attention_weights2, axis=-1)
        user_emb_att = tf.reduce_sum(clicked_embeddings * attention_weights_expanded, axis=1)
        item_emb_att = item_emb_cnn

        return user_emb_att, item_emb_att

    def deep_func(self):
        with tf.name_scope('deep_part'):
            bc_feat = tf.concat([self.base_feature, self.cross_feature], axis=-1)
            if self.use_kg and self.use_bert == 0:
                user_emb = tf.concat([self.w2v_user_embedding, self.kg_user_embedding], axis=-1)
                item_emb = tf.concat([self.w2v_item_embedding, self.kg_item_embedding], axis=-1)
            elif self.use_kg == 0 and self.use_bert:
                user_emb = tf.concat([self.w2v_user_embedding, self.bert_user_embedding], axis=-1)
                item_emb = tf.concat([self.w2v_item_embedding, self.bert_item_embedding], axis=-1)
            elif self.use_kg and self.use_bert:
                user_emb = tf.concat([self.w2v_user_embedding, self.kg_user_embedding,self.bert_user_embedding], axis=-1)
                item_emb = tf.concat([self.w2v_item_embedding, self.kg_item_embedding,self.bert_item_embedding], axis=-1)
            else:
                user_emb = self.w2v_user_embedding
                item_emb = self.w2v_item_embedding

            if self.use_attenton:
                user_emb_att, item_emb_att = self._attention(user_emb,item_emb)
                dnn_input = tf.concat([bc_feat, user_emb_att,item_emb_att], axis=-1)
            else:
                dnn_input = tf.concat([bc_feat, user_emb, item_emb], axis=-1)

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
