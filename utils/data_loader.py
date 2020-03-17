#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf


def _parse_function(example_proto, mp):
    cate_alg = ['deepfm_cate', 'dnn_cate', "dnn_multi_cate", "deepfm_multi_cate"]

    if mp.alg_name in cate_alg:
        print("-----------tf_cate-----------")
        features = {
            'cate_feats': tf.FixedLenFeature([mp.cate_field_size + mp.multi_feats_size], tf.int64),
            'label': tf.FixedLenFeature([1], tf.float32),
            'vector_feats': tf.FixedLenFeature([mp.vector_feats_size], tf.float32)
        }
    else:
        print("-----------tf_pipeline-----------")
        features = {
            'label': tf.FixedLenFeature([1], tf.float32),
            'cont_feats': tf.FixedLenFeature([mp.cont_field_size], tf.float32),
            'vector_feats': tf.FixedLenFeature([mp.vector_feats_size], tf.float32),
            'cate_feats': tf.FixedLenFeature([mp.cate_field_size + mp.multi_feats_size], tf.int64),
        }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features


def pipeline_process(mp, file_dir_list, action_type):
    data_set = tf.data.TFRecordDataset(file_dir_list, buffer_size=mp.batch_size * mp.batch_size) \
        .map(lambda x: _parse_function(x, mp), num_parallel_calls=10) \
        .shuffle(buffer_size=mp.batch_size * 10) \
        .batch(mp.batch_size, drop_remainder=True)

    if action_type == 'train':
        data_set = data_set.repeat(mp.epochs)

    iterator = data_set.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element


def load_input_file(mp, input_path, action_type):
    file_dir_list = get_file_list(input_path)
    data_out = pipeline_process(mp, file_dir_list, action_type)
    return data_out


def get_file_list(input_path):
    file_list = tf.gfile.ListDirectory(input_path)
    print("file_list_len:", len(file_list))
    file_dir_list = []
    for file in file_list:
        if file[:4] == "part":
            file_path = input_path + file
            file_dir_list.append(file_path)

    return file_dir_list


if __name__ == '__main__':
    a = dict()
    a['liu'] = 'an'
    for key, value in a.items():
        print(key, " = ", value)