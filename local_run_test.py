#!/usr/bin/env python
# coding=utf-8
import sys
import time

import tensorflow as tf

import utils.data_loader as data_load
import utils.my_utils as my_utils

my_utils.venus_set_environ()
parse_dict = my_utils.arg_parse(sys.argv)


class ModelParams:
    def __init__(self):
        self.learning_rate_decay_steps = int(parse_dict.get("learning_rate_decay_steps", "10000000"))
        self.learning_rate_decay_rate = float(parse_dict.get("learning_rate_decay_steps", "0.9"))
        self.dropout_keep_deep = [1, 1, 1, 1, 1]
        self.hidden_units = [int(i) for i in parse_dict.get("hidden_units", "").split(",")]
        self.max_data_size = int(parse_dict.get("max_data_size", "8000000"))
        self.learning_rate = float(parse_dict.get("learning_rate", "0.001"))
        self.l2_reg = float(parse_dict.get("l2_reg", "0.00001"))
        self.batch_size = 1024
        self.hidden_units = [512, 256, 128]

        self.alg_name = parse_dict.get("alg_name", "dnn_tensorboard")
        self.action_type = parse_dict.get("action_type", "train")
        self.epochs = int(parse_dict.get("epochs", "4"))
        self.batch_size = int(parse_dict.get("batch_size", ""))
        self.embedding_size = int(parse_dict.get("embedding_size", "16"))
        self.cate_feats_size = int(parse_dict.get("cate_feats_size", "10000"))
        self.num_batch_size = int(parse_dict.get("num_batch_size", "100"))
        self.feat_conf_path = parse_dict.get("feat_conf_path", None)
        self.train_path = parse_dict.get("train_path", "")
        self.predict_path = parse_dict.get("predict_path", "")
        self.model_pb = parse_dict.get("model_pb", "")
        self.save_model_checkpoint = parse_dict.get("save_model_checkpoint", "")
        self.model_restore = parse_dict.get("model_restore", "0")
        self.restore_model_checkpoint = parse_dict.get("restore_model_checkpoint", "")

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
    else:
        print("alg_name = %s is error" % mp.alg_name)
        exit(-1)
        import models.deepfm_pipeline as alg_model

    if mp.action_type == "train":
        start_time = time.time()
        train_data = data_load.load_input_file(mp, mp.train_path, 'train')
        predict_data = data_load.load_input_file(mp, mp.predict_path, '')
        m = alg_model.DeepModel(mp, train_data)
        print("--------------train------------")
        m.fit(mp.num_batch_size, predict_data)
        end_time = time.time()
        print("model training time: %.2f s" % (end_time - start_time))
        alg_model.predict(predict_data, mp.model_pb)
    elif mp.action_type == "pred":
        print("--------------predict------------")
        # 预测tfrecord数据集
        predict_data = data_load.load_input_file(mp, mp.predict_path, '')
        alg_model.predict(predict_data, mp.model_pb)
    else:
        print("action_type = %s is error !!!" % mp.action_type)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run(main=main)
