#!/usr/bin/env python
# coding=utf-8
import sys

import tensorflow as tf

import utils.data_loader as data_load


# import time
# import utils.my_utils as my_utils
# my_utils.venus_set_environ()
# model_time = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
# parse_dict = my_utils.arg_parse(sys.argv)


class ModelParams:
    def __init__(self):
        # self.learning_rate_decay_steps = int(parse_dict.get("learning_rate_decay_steps", "10000000"))
        # self.learning_rate_decay_rate = float(parse_dict.get("learning_rate_decay_steps", "0.9"))
        # self.action_type = parse_dict.get("action_type", "train")
        # self.max_data_size = int(parse_dict.get("max_data_size", "8000000"))
        # self.alg_name = parse_dict.get("alg_name", "dnn_multi")
        # self.model_pb = parse_dict.get("model_pb", "model_pb/1")
        # self.train_path = parse_dict.get("train_path", "data/")
        # self.pred_path = parse_dict.get("pred_path", "data/")
        # self.epochs = int(parse_dict.get("epochs", "10"))
        # self.batch_size = int(parse_dict.get("batch_size", "1024"))
        # self.learning_rate = float(parse_dict.get("learning_rate", "0.0001"))
        # self.l2_reg = float(parse_dict.get("l2_reg", "0.000001"))
        # self.hidden_units = [int(i) for i in parse_dict.get("hidden_units", "1024,256,128").split(",")]
        # self.cate_index_size = int(parse_dict.get("cate_index_size", "450000"))
        # self.embedding_size = int(parse_dict.get("embedding_size", "16"))
        # self.cont_feat_conf_path = parse_dict.get("cont_feat_conf_path", "/data/webroot/zijingrong/deep_learning/dnn_mul/conf/dnn.conf")
        # self.cate_feat_conf_path = parse_dict.get("cate_feat_conf_path", "/data/webroot/zijingrong/deep_learning/dnn_mul/conf/lr.conf")
        # self.word2vec_embedding_path = parse_dict.get("word2vec_embedding_path",
        #                                               "/data/webroot/zijingrong/deep_learning/dnn_mul/bin/tags_embedding.npy"
        # self.cate_field_size, self.multi_cate_field_size = get_cate_field(self.cate_feat_conf_path)
        # self.cont_field_size = get_cont_field(self.cont_feat_conf_path)
        # self.wide_field_size = get_wide_field(self.cate_feat_conf_path)
        # self.print_iter = int(parse_dict.get("print_iter", "2"))
        # self.num_batch_size=int(parse_dict.get("num_batch_size", "2"))
        for i in range(len(sys.argv)):
            print(i, sys.argv[i])
        self.learning_rate_decay_steps = 10000000
        self.learning_rate_decay_rate = 0.9
        self.batch_size = 1024
        self.learning_rate = 0.001
        self.l2_reg = 0.00001
        self.hidden_units = [1024, 512, 256]
        self.max_data_size = 7000000
        self.alg_name = str(sys.argv[1])
        self.action_type = str(sys.argv[2])
        self.epochs = int(sys.argv[3])
        self.cate_index_size = int(sys.argv[4])
        self.embedding_size = int(sys.argv[5])
        self.train_path = str(sys.argv[6])
        self.pred_path = str(sys.argv[7])
        self.model_pb = str(sys.argv[8])
        self.cont_feats_conf_path = str(sys.argv[9])
        self.cate_feats_conf_path = str(sys.argv[10])
        self.word2vec_embedding_path = str(sys.argv[11])
        self.print_iter = int(sys.argv[12])
        self.num_batch_size = int(sys.argv[13])
        self.cont_field_size, self.vector_field_size = get_cont_field(self.cont_feats_conf_path)
        self.cate_field_size, self.multi_cate_field_size, self.wide_field_size = get_cate_field(
            self.cate_feats_conf_path)


def get_cont_field(path):
    cont_size = 0
    vector_size = 0
    vec_name_list = ["user_vec", "ruUserVec", "item_vec", "user_kgv", "item_kgv"]
    mid_vec_list = ["item_midv", "user_midv"]

    if path is None or path == '':
        print("cont_conf_path is None")
        return cont_size

    with open(path, 'r') as f:
        for line in f.readlines():
            line_data = line.strip()
            if line_data == '':
                continue

            config_arr = line_data.split("\t")
            col_name = config_arr[0]
            result_type = config_arr[2]

            if result_type == 'vector' or result_type == 'arr' or result_type == 'vec':
                if col_name in vec_name_list:
                    cont_size += 200
                    vector_size += 200
                elif col_name in mid_vec_list:
                    cont_size += 100
                    vector_size += 100
                else:
                    print("cont conf is error")
                    print(line_data)
                    exit(-1)
            else:
                cont_size += 1

    return cont_size, vector_size


def get_cate_field(path):
    dnn_cate_size = 0
    multi_cate_size = 0
    wide_cate_size = 0
    if path is None or path == '':
        print("cate_conf_path is None")
        return dnn_cate_size, multi_cate_size, wide_cate_size

    with open(path, 'r') as f:
        for line in f.readlines():
            line_data = line.strip()
            if line_data == '':
                continue

            config_arr = line_data.split("\t")
            feats_type = config_arr[3]
            result_type = config_arr[6]
            if result_type != 'top':
                if feats_type == 'wide':
                    wide_cate_size += 1
                elif feats_type == 'deep':
                    dnn_cate_size += 1
            else:
                result_parameter = config_arr[7]
                top_n = int(result_parameter.strip().split("=")[1])
                multi_cate_size += top_n

    return dnn_cate_size, multi_cate_size, wide_cate_size


def main(_):
    mp = ModelParams()
    for key, value in mp.__dict__.items():
        print(key, "=", value)

    if mp.alg_name == "dnn":
        print("----------dnn--------")
        import models.dnn as alg_model

    elif mp.alg_name == 'dnn_multi':
        print("------dnn_multi--------")
        import models.dnn_multi as alg_model
    elif mp.alg_name == 'dnn_multi_textline':
        print("------dnn_multi_textline------")
        import models.dnn_multi_textline as alg_model

    elif mp.alg_name == 'wdl':
        print("---------wdl---------")
        import models.wdl as alg_model
    elif mp.alg_name == 'wdl_textline':
        print("---------wdl_textline-------")
        import models.wdl_textline as alg_model

    elif mp.alg_name == 'deepfm':
        print("--------deepfm-------------")
        import models.deepfm as alg_model
    elif mp.alg_name == 'deepfm_textline':
        print("--------deepfm_textline---------")
        import models.deepfm_textline as alg_model

    else:
        print("alg_name = %s is error" % mp.alg_name)
        import models.dnn as alg_model
        exit()

    load_alg_name = ['dnn', 'dnn_multi', 'wdl', 'deepfm']
    textline_alg_name = ['dnn_multi_textline', 'wdl_textline', 'deepfm_textline']

    if mp.action_type == "train":
        print("--------------train------------")
        print("------load_data------")
        if mp.alg_name in load_alg_name:
            train_data = data_load.load_input_file(mp, mp.train_path, '')
            pred_data = data_load.load_input_file(mp, mp.pred_path, '')
            print("------fit_model------")
            m = alg_model.DeepModel(mp)
            m.fit(train_data, pred_data)
            m.predict(pred_data)
        elif mp.alg_name in textline_alg_name:
            train_data = data_load.load_input_file(mp, mp.train_path, "textline")
            print("------fit_model------")
            m = alg_model.DeepModel(mp, train_data)
            m.fit(print_iter=mp.print_iter, num_batch_size=mp.num_batch_size)
            pred_data = data_load.load_input_file(mp, mp.pred_path, "")
            m.predict(pred_data)
        else:
            pass
    elif mp.action_type == "pred":
        if mp.alg_name in load_alg_name:
            pred_data = data_load.load_input_file(mp, mp.pred_path, "")
            m = alg_model.DeepModel(mp)
            m.predict(pred_data)
        elif mp.alg_name in textline_alg_name:
            pred_data = data_load.load_input_file(mp, mp.pred_path, "")
            alg_model.predict(pred_data, mp.model_pb)
        else:
            pass
    else:
        print("action_type = %s is error !!!" % mp.action_type)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run(main=main)
