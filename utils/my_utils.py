#!/usr/bin/env python
# coding=utf-8

import os


def arg_parse(argv):
    parse_dict = dict()
    for i in range(1, len(argv)):
        line_parse = argv[i].split("=")
        key = line_parse[0].strip()
        value = line_parse[1].strip()
        parse_dict[key] = value
    return parse_dict


def venus_set_environ():
    # venus 参数设置
    os.environ['AWS_ACCESS_KEY_ID'] = "J2SU8BKYDQAKTKBHY1DV"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "4icFh3queHjR2jPk8U2j7qM1ekw7HpGQqkVPDgZ4"
    os.environ['S3_ENDPOINT'] = "s3szoffline.sumeru.mig"
    os.environ['S3_USE_HTTPS'] = '0'
    os.environ['AWS_DEFAULT_REGION'] = 'default'


def feat_size(path, alg_name):
    cont_size = 0
    vector_size = 0
    cate_size = 0
    multi_cate_size = 0
    multi_cate_field = 0
    multi_cate_range = []
    vec_name_list = ["user_vec", "ruUserVec", "item_vec", "user_kgv", "item_kgv"]
    mid_vec_list = ["item_midv", "user_midv"]

    pool_alg = ["deepfm_multi_cat", "deepfm_multi", "dnn_multi_cat", "dnn_multi"]

    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        if file != "dnn.conf" and file != "lr.conf":
            continue
        file_path = path + "/" + file
        print("----read %s----" % file_path)
        with open(file_path, 'r') as f:
            index_start = 0
            for line in f.readlines():
                line_data = line.strip()
                if line_data == '':
                    continue

                try:
                    config_arr = line_data.split("\t")
                    col_name = config_arr[0]
                    result_type = config_arr[2]

                    if result_type == 'vector' or result_type == 'vec':
                        if col_name in vec_name_list:
                            vector_size += 200
                        elif col_name in mid_vec_list:
                            vector_size += 100

                    elif result_type == 'arr':
                        result_parameter = config_arr[7]
                        feature_name = config_arr[-1]
                        top_n = int(result_parameter.strip().split("=")[1])
                        if alg_name not in pool_alg:
                            cate_size += top_n
                        else:
                            multi_cate_size += top_n
                            multi_cate_field += 1
                            index_end = index_start + top_n
                            index_range = [index_start, index_end, feature_name]
                            multi_cate_range.append(index_range)
                            index_start = index_end

                    elif result_type == 'string':
                        cate_size += 1
                    elif result_type == 'float':
                        cont_size += 1
                    else:
                        print("%s is error!!!" % line_data)
                except:
                    print("-----------feat_conf is Error!!!!-----------")
                    print(line_data)
                    exit(-1)

    return cont_size, vector_size, cate_size, multi_cate_size, multi_cate_field, multi_cate_range
