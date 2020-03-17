#!/usr/bin/env python
# coding=utf-8
import pickle
import sys

import tensorflow as tf


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def get_libsvm_value(line):
    line_sp = line.split(" ")
    index_list = list()
    value_list = list()
    for j in range(len(line_sp)):
        split_data = line_sp[j].split("=")
        index_list.append(int(split_data[0]))
        value_list.append(float(split_data[1]))

    return index_list, value_list


def get_dense_value(line):
    return line.split(" ")


def read_data(mp, file_dir_list):
    print("*****read data******")
    sys.stdout.flush()
    read_data_all = []
    count = 0
    exit_flag = False
    for file in file_dir_list:
        try:
            inf = tf.gfile.FastGFile(file, "r")
        except:
            print("not file exception input file not find")
            continue
        for line in inf.readlines():
            if not line:
                continue
            read_data_all.append(line)
            count += 1
            if count > mp.max_data_size:
                exit_flag = True
                break
            if count % 200000 == 0:
                print(count)
                sys.stdout.flush()
        if exit_flag:
            break
        inf.close()

    # random.shuffle(read_data_all)
    # read_data_all =read_data_all[:TS.dataSizeSet]
    print("size of read data = ", len(read_data_all))
    return read_data_all


def load_process(mp, file_dir_list):
    data = read_data(mp, file_dir_list)
    data_split = data[0].strip('\n').split(",")
    cont_field_size = len(data_split[1].split(" "))
    cate_field_size = len(data_split[2].split(" "))
    print("data_cont_field_size = ", cont_field_size)
    print("data_cate_field_size = ", cate_field_size)
    if cont_field_size != mp.cont_field_size \
            or cate_field_size != mp.cate_field_size:
        print("feature size is error!!!")
        exit(-1)

    if mp.alg_name == "dnn_multi":
        multi_cate_field_size = len(data_split[3].split(" "))
        print("data_multi_cate_field_size = ", multi_cate_field_size)
        if multi_cate_field_size != mp.multi_cate_field_size:
            print("feature size is error!!!")
            exit(-1)

    if mp.alg_name == "wdl" or mp.alg_name == "wdl_textline":
        wide_field_size = len(data_split[3].split(" "))
        print("data_wide_field_size = ", wide_field_size)
        if wide_field_size != mp.wide_field_size:
            print("feature size is error!!!")
            exit(-1)

    # if mp.alg_name == "deep_fm":
    #     fm_field_size = len(data_split[1].split(" "))
    #     print("data_fm_field_size = ", fm_field_size)
    #     if fm_field_size != mp.fm_field_size:
    #         print("feature size is error!!!")
    #         exit(-1)

    data_out = []
    for i in range(0, len(data), mp.batch_size):
        batch_data = data[i: i + mp.batch_size]
        labels_batch = []
        cont_feats_list = []
        cate_feats_list = []
        multi_cate_feats_list = []
        multi_cate_feats_value_list = []
        wide_feats_list = []
        vector_feats_list = []

        pickle_data = {}
        for line in batch_data:
            line_sp = line.strip('\n').split(",")
            label = float(line_sp[0])
            labels_batch.append([label])

            cont_feats_values = get_dense_value(line_sp[1])
            cate_feats_index = get_dense_value(line_sp[2])
            index_vector = mp.cont_field_size - mp.vector_field_size
            vector_feats_values = cont_feats_values[index_vector:]
            vector_feats_list.append(vector_feats_values)
            cont_feats_values = cont_feats_values[:index_vector]
            cont_feats_list.append(cont_feats_values)
            cate_feats_list.append(cate_feats_index)

            if mp.alg_name == "dnn_multi":
                multi_cate_feats_index, multi_cate_feats_value = get_libsvm_value(line_sp[3])
                multi_cate_feats_list.append(multi_cate_feats_index)
                multi_cate_feats_value_list.append(multi_cate_feats_value)

            if mp.alg_name == "wide_deep" or mp.alg_name == "wdl_textline":
                wide_feats_index = get_dense_value(line_sp[3])
                wide_feats_list.append(wide_feats_index)

        pickle_data["cont_feats"] = cont_feats_list
        pickle_data["vector_feats"] = vector_feats_list
        pickle_data["cate_feats"] = cate_feats_list
        pickle_data["mul_cate_feats"] = multi_cate_feats_list
        pickle_data["mul_cate_feats_value"] = multi_cate_feats_value_list
        pickle_data["wide_feats"] = wide_feats_list
        pickle_data["labels"] = labels_batch
        pickle_data = pickle.dumps(pickle_data)
        data_out.append(pickle_data)

    return data_out


def load_test(mp, file_dir_list):
    data = read_data(mp, file_dir_list)
    data_out = []
    for i in range(0, len(data), mp.batch_size):
        batch_data = data[i: i + mp.batch_size]
        labels_batch = []
        cont_feats_index_list = []
        cont_feats_value_list = []
        cate_feats_index_list = []
        cate_feats_value_list = []
        vector_feats_value_list = []

        pickle_data = {}
        for line in batch_data:
            line_sp = line.strip('\n').split(",")
            label = float(line_sp[0])
            labels_batch.append([label])

            cont_feats_value = get_dense_value(line_sp[1])
            cont_feats_index = [i for i in range(0, mp.cont_field_size - mp.vector_field_size)]
            cate_feats_value = [i for i in range(0, mp.cate_field_size)]
            cate_feats_index = get_dense_value(line_sp[2])

            index_vector = mp.cont_field_size - mp.vector_field_size
            vector_feats_value = cont_feats_value[index_vector:]
            vector_feats_value_list.append(vector_feats_value)
            cont_feats_value = cont_feats_value[:index_vector]

            cont_feats_index_list.append(cont_feats_index)
            cont_feats_value_list.append(cont_feats_value)
            cate_feats_index_list.append(cate_feats_index)
            cate_feats_value_list.append(cate_feats_value)

        pickle_data["labels"] = labels_batch
        pickle_data["cont_feats_index"] = cont_feats_index_list
        pickle_data["cont_feats_value"] = cont_feats_value_list
        pickle_data["cate_feats_index"] = cate_feats_index_list
        pickle_data["cate_feats_value"] = cate_feats_value_list
        pickle_data["vector_feats_value"] = vector_feats_value_list
        pickle_data = pickle.dumps(pickle_data)
        data_out.append(pickle_data)

    return data_out


def decode_libsvm(line, mp):
    data_dict = dict()
    columns = tf.string_split([line], ',')
    labels = tf.string_to_number([columns.values[0]], out_type=tf.float32)

    cont_feats_values = 0
    vector_feats_values = 0
    cate_feats_values = 0
    mul_cate_feats_ids = 0
    mul_cate_feats_values = 0
    wide_feats_values = 0

    # -cont feat
    cont_feats = columns.values[1]
    cont_feats = tf.string_split([cont_feats], ' ')
    cont_feats_values = tf.string_to_number(cont_feats.values, out_type=tf.float32)

    if mp.alg_name == "deepfm_textline":
        vector_start_index = mp.cont_field_size - mp.vector_field_size
        vector_feats_values = cont_feats_values[vector_start_index:]
        cont_feats_values = cont_feats_values[:vector_start_index]

    # --cate feats
    cate_feats = columns.values[2]
    cate_feats = tf.string_split([cate_feats], ' ')
    cate_feats_values = tf.string_to_number(cate_feats.values, out_type=tf.int32)

    # --mul_cate feats
    if mp.alg_name == "dnn_multi_textline":
        mul_cate_feats = columns.values[3]
        mul_cate_feats = tf.string_split([mul_cate_feats], ' ')
        mul_cate_splits = tf.string_split(mul_cate_feats.values, '=')
        mul_cate_id_value = tf.reshape(mul_cate_splits.values, shape=mul_cate_splits.dense_shape)
        mul_cate_ids, mul_cate_values = tf.split(mul_cate_id_value, num_or_size_splits=2, axis=1)
        mul_cate_feats_ids = tf.squeeze(tf.string_to_number(mul_cate_ids, out_type=tf.int32), axis=1)
        mul_cate_feats_values = tf.squeeze(tf.string_to_number(mul_cate_values, out_type=tf.float32), axis=1)

    # --wide feats
    if mp.alg_name == "wdl_textline":
        wide_feats = columns.values[3]
        wide_feats = tf.string_split([wide_feats], ' ')
        wide_feats_values = tf.string_to_number(wide_feats.values, out_type=tf.int32)

    # --feature dict
    data_dict["labels"] = labels
    data_dict["cont_feats"] = cont_feats_values
    data_dict["vector_feats"] = vector_feats_values
    data_dict["cate_feats"] = cate_feats_values
    data_dict["mul_cate_feats"] = mul_cate_feats_ids
    data_dict["mul_cate_feats_value"] = mul_cate_feats_values
    data_dict["wide_feats_value"] = wide_feats_values

    return data_dict


def textline_process(mp, filenames):
    data_set = tf.data.TextLineDataset(filenames)
    data_set = data_set.map(lambda line: decode_libsvm(line, mp)) \
        .batch(mp.batch_size, drop_remainder=True).repeat(mp.epochs)
    iterator = data_set.make_one_shot_iterator()
    data_dict = iterator.get_next()
    return data_dict


def load_input_file(mp, input_path, read_file_type):
    file_dir_list = get_file_list(input_path)
    if read_file_type == "textline":
        data_out = textline_process(mp, file_dir_list)
    elif read_file_type == "tf_test":
        data_out = load_test(mp, file_dir_list)
    else:
        data_out = load_process(mp, file_dir_list)

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
