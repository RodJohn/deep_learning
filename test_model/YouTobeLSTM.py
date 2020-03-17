import tensorflow as tf
import os
import time
from sklearn.metrics import roc_auc_score
import utils.data_loader_test as data_loader
from tensorflow.contrib import rnn
import numpy as np
import sys


class LstmYouTobe(object):
    def __init__(self, args):
        self.lstm_dim = 200
        self.learning_rate = 0.001
        self.lr_decay = 0.9
        self.num_epochs = args.num_epochs
        self.batch_size = 1024
        self.vocab_size = 482
        self.embedding_size = 8
        self.metric_type = "auc"
        self.model_save_path = args.model_save_path
        self.aids_click = 5
        self.tags_count = 5
        self.lstm_layers = 2
        self.cont_feat_size = args.cont_feat_size
        self.cate_feat_size = args.cate_feat_size
        self.tag_feat_index_dict = args.tag_index_dict
        self.cat_feat_index_dict = args.cat_index_dict
        self.word2vec_embedding_path = args.word2vec_embedding_path

        self.hidden_units = [128, 64]
        self.dropout_keep_deep = (len(self.hidden_units) + 5) * [1]
        self.l2_reg = 0.00001
        self.learning_rate = args.learning_rate
        self.decay_steps = args.learning_rate_decay_steps
        self.decay_rate = args.learning_rate_decay_rate
        self.global_step = tf.Variable(0, trainable=False)

        with tf.variable_scope("weight_matrix"):
            embeddings = tf.get_variable('weight_mat',
                                         dtype=tf.float32,
                                         shape=(self.vocab_size, self.embedding_size),
                                         initializer=tf.contrib.layers.xavier_initializer())

            self.embeddings = tf.concat((tf.zeros(shape=[1, self.embedding_size]), embeddings[1:, :]), 0)
        print("load word2vec embedding")
        emb = np.load(self.word2vec_embedding_path)
        self.word2vecEmbedding = tf.constant(emb, dtype=np.float32, name='tags_emb')

        self.__init_graph()

    def __init_graph(self):
        self.cate_feat = tf.placeholder(tf.int32, shape=[None, self.cate_feat_size], name='cate_feat')
        self.cont_feat = tf.placeholder(tf.float32, shape=[None, self.cont_feat_size], name='cont_feat')
        self.u_seq_feat = tf.placeholder(tf.int32, shape=[None, self.aids_click, self.tags_count], name='u_seq_feat')
        self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
        self.u_click_len = tf.placeholder(tf.int32, shape=[None], name='u_click_len')

        user_emb_all = tf.nn.embedding_lookup(self.word2vecEmbedding, self.u_seq_feat)
        self.user_emb_all = tf.reduce_mean(user_emb_all, axis=2)
        lstm_vec = self.BiRNN(self.user_emb_all)

        self.cate_feature = {}
        self.cate_feature_emb = {}
        self.weights = {}
        self.biases = {}

        for index, col in enumerate(self.cat_feat_index_dict):
            print("cat feat:", col, self.cat_feat_index_dict[col][0], self.cat_feat_index_dict[col][1])
            self.cate_feature[col] = self.cate_feat[:,
                                     self.cat_feat_index_dict[col][0]:self.cat_feat_index_dict[col][1]]
            cate_embedding = tf.nn.embedding_lookup(self.embeddings, self.cate_feature[col])
            self.cate_feature_emb[col] = tf.reduce_mean(cate_embedding, axis=1)

            if index == 0:
                self.emb_all = self.cate_feature_emb[col]
            else:
                self.emb_all = tf.concat([self.emb_all, self.cate_feature_emb[col]], axis=1)

        for col in self.tag_feat_index_dict:
            print("tags feat:", col, self.tag_feat_index_dict[col][0], self.tag_feat_index_dict[col][1])
            self.cate_feature[col] = self.cate_feat[:,
                                     self.tag_feat_index_dict[col][0]:self.tag_feat_index_dict[col][1]]
            cate_embedding = tf.nn.embedding_lookup(self.word2vecEmbedding, self.cate_feature[col])
            self.cate_feature_emb[col] = tf.reduce_mean(cate_embedding, axis=1)
            self.emb_all = tf.concat([self.emb_all, self.cate_feature_emb[col]], axis=1)

        self.dense_vector = tf.concat([self.emb_all, self.cont_feat], axis=1)
        self.dense_vector = tf.concat([self.dense_vector, lstm_vec], axis=1)

    def BiRNN(self, user_emb_all):
        x = tf.unstack(user_emb_all, self.aids_click, 1)
        lstm_fw_cell = rnn.GRUCell(self.lstm_dim)
        lstm_bw_cell = rnn.GRUCell(self.lstm_dim)
        try:
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)
        except Exception:
            outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32)
        return outputs[-1]

    def lstm(self, user_emb_all):
        with tf.name_scope('LSTM'):
            layers = [tf.contrib.rnn.GRUCell(num_units=self.lstm_dim, activation=tf.nn.tanh)
                      for _ in range(self.lstm_layers)]
            multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
            self.outputs, self.states = tf.nn.dynamic_rnn(multi_layer_cell, user_emb_all, dtype=tf.float32,
                                                          sequence_length=self.u_click_len)
        return self.states[-1]

    def deep_func(self):
        with tf.name_scope('deep_part'):
            len_layers = len(self.hidden_units)
            input_size = self.dense_vector.shape.as_list()[1]
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
                self.loss = self.loss + tf.contrib.layers.l1_regularizer(self.l2_reg)(
                    self.weights['deep_%s' % i])
        self.learning_rate_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                              self.decay_rate,
                                                              staircase=True)
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_decay).minimize(self.loss,
                                                                                                 global_step=self.global_step)
        tf.identity(self.out, name="output")
        sys.stdout.flush()

    def fit(self, train_data, val_data):
        self.modelOptimizer()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            save_path = os.path.join(self.model_save_path, 'best_val')
            saver = tf.train.Saver(max_to_keep=10)
            try:
                latest_checkpoint = tf.train.latest_checkpoint(self.model_save_path)
                saver.restore(sess=sess, save_path=latest_checkpoint)
                print("restore model!")
            except:
                print("no model!")

            losses = []
            num_samples = 0
            for epoch in range(self.num_epochs):
                st = time.time()
                for i in range(0, len(train_data), self.batch_size):
                    batch_data = data_loader.DataProcess(train_data, i, i + self.batch_size)
                    batch_label = batch_data["label"]
                    batch_cate_feat = batch_data["cate_feat"]
                    batch_cont_feat = batch_data["cont_feat"]
                    batch_user_seq = batch_data["user_seq"]
                    batch_u_click_len = batch_data["u_click_len"]

                    feed_dict = {
                        self.cate_feat: batch_cate_feat,
                        self.cont_feat: batch_cont_feat,
                        self.u_seq_feat: batch_user_seq,
                        self.u_click_len: batch_u_click_len,
                        self.label: batch_label,
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
                saver.save(sess, save_path, global_step=self.global_step)

    def evaluate(self, sess, data_val):
        pred_list = []
        label_list = []
        for i in range(0, len(data_val), self.batch_size):
            batch_data = data_loader.DataProcess(data_val, i, i + self.batch_size)
            batch_label = batch_data["label"]
            batch_cate_feat = batch_data["cate_feat"]
            batch_cont_feat = batch_data["cont_feat"]
            batch_user_seq = batch_data["user_seq"]
            batch_u_click_len = batch_data["u_click_len"]

            feed_dict = {
                self.cate_feat: batch_cate_feat,
                self.cont_feat: batch_cont_feat,
                self.u_seq_feat: batch_user_seq,
                self.u_click_len: batch_u_click_len,
                self.label: batch_label,
            }
            label_list.extend(batch_data["label"])
            y_pred = sess.run(self.out, feed_dict=feed_dict)
            pred_list.extend(y_pred)
        return roc_auc_score(label_list, pred_list)
