#!/usr/bin/env python
# coding=utf-8
import sys
import time

import tensorflow as tf

import utils.data_loader as data_load
import utils.my_utils as my_utils


class ModelParams:
    def __init__(self):
        self.alg_name = sys.argv[1]
        self.action_type = sys.argv[2]
        self.epochs = int(sys.argv[3])
        self.embedding_size = int(sys.argv[4])
        self.cate_feats_size = int(sys.argv[5])
        self.num_batch_size = int(sys.argv[6])
        self.feat_conf_path = sys.argv[7]
        self.train_path = sys.argv[8]
        self.predict_path = sys.argv[9]
        self.model_pb = sys.argv[10]
        self.save_model_checkpoint = sys.argv[11]
        self.model_restore = int(sys.argv[12])
        self.restore_model_checkpoint = sys.argv[13]
        # self.summary_log = sys.argv[14]
        self.learning_rate = 0.001
        self.hidden_units = [512, 256, 128]
        self.dropout_keep_deep = [1, 1, 1, 1, 1]
        self.learning_rate_decay_steps = 10000000
        self.learning_rate_decay_rate = 0.9

        self.l2_reg = 0.00001
        self.batch_size = 1024

        self.cont_field_size, self.vector_feats_size, self.cate_field_size, self.multi_feats_size, \
            self.multi_field_size, self.multi_feats_range = my_utils.feat_size(self.feat_conf_path, self.alg_name)


def main(_):
    mp = ModelParams()

    for key, value in mp.__dict__.items():
        print(key, "=", value)

    if mp.alg_name == "deepfm_pipeline":
        import models.deepfm_pipeline as alg_model
    elif mp.alg_name == "deepfm_cate":
        import models.deepfm_cate as alg_model
    elif mp.alg_name == "deepfm_multi_cate":
        import models.deepfm_multi_cate as alg_model
    elif mp.alg_name == "deepfm_multi":
        import models.deepfm_multi as alg_model
    elif mp.alg_name == "dnn_cate":
        import models.dnn_cate as alg_model
    elif mp.alg_name == "dnn_pipeline":
        import models.dnn_pipeline as alg_model
    elif mp.alg_name == "dnn_multi_cate":
        import models.dnn_multi_cate as alg_model
    elif mp.alg_name == "dnn_multi":
        import models.dnn_multi as alg_model
    elif mp.alg_name == "dnn_tensorboard":
        import models.dnn_tensorboard as alg_model
    else:
        print("alg_name = %s is error" % mp.alg_name)
        exit(-1)
        import models.deepfm_pipeline as alg_model

    if mp.action_type == "train":
        start_time = time.time()
        train_data = data_load.load_input_file(mp, mp.train_path, 'train')
        predict_data = data_load.load_input_file(mp, mp.predict_path, 'pred')
        m = alg_model.DeepModel(mp, train_data, predict_data)
        print("--------------train------------")
        m.fit(mp.num_batch_size)
        end_time = time.time()
        print("model training time: %.2f s" % (end_time - start_time))
        alg_model.predict(predict_data, mp.model_pb)
    elif mp.action_type == "pred":
        print("--------------predict------------")
        # 预测tfrecord数据集
        predict_data = data_load.load_input_file(mp, mp.predict_path, 'pred')
        alg_model.predict(predict_data, mp.model_pb)
    else:
        print("action_type = %s is error !!!" % mp.action_type)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run(main=main)
