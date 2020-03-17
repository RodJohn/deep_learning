import tensorflow as tf


def decode_libsvm(line):
    data_dict = dict()
    columns = tf.string_split([line], ',')
    labels = tf.string_to_number([columns.values[0]], out_type=tf.float32)

    cont_feats = columns.values[1]
    cat_feats = columns.values[2]
    mul_cat_feats = columns.values[3]

    # -cont feats
    cont_feats = tf.string_split([cont_feats], ' ')
    cont_feats_value = tf.string_to_number(cont_feats.values, out_type=tf.float32)

    # --cat feats
    cat_feats = tf.string_split([cat_feats], ' ')
    cat_feats_values = tf.string_to_number(cat_feats.values, out_type=tf.int32)

    # --mul_cat feats
    mul_cat_feats = tf.string_split([mul_cat_feats], ' ')
    mul_cat_splits = tf.string_split(mul_cat_feats.values, '=')
    mul_cat_id_value = tf.reshape(mul_cat_splits.values, shape=mul_cat_splits.dense_shape)
    mul_cat_ids, mul_cat_values = tf.split(mul_cat_id_value, num_or_size_splits=2, axis=1)
    mul_cat_feats_ids = tf.squeeze(tf.string_to_number(mul_cat_ids, out_type=tf.int32), axis=1)
    mul_cat_feats_values = tf.squeeze(tf.string_to_number(mul_cat_values, out_type=tf.float32), axis=1)

    # --feature dict
    data_dict["labels"] = labels
    data_dict["cont_feats"] = cont_feats_value
    data_dict["cat_feats"] = cat_feats_values
    data_dict["mul_cat_feats"] = mul_cat_feats_ids
    data_dict["mul_cat_feats_value"] = mul_cat_feats_values

    return data_dict


def dnn_multi_process(filenames, batch_size, epochs):
    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.map(lambda line: decode_libsvm(line)).batch(batch_size).repeat(epochs)
    iterator = dataset.make_one_shot_iterator()
    data_dict = iterator.get_next()
    return data_dict


def file_process(mp, file_dir_list):
    data_out = None
    alg_name = mp.alg_name
    print("---------file_process alg_type = %s--------" % alg_name)
    if alg_name == "dnn_multi_textline":
        data_out = dnn_multi_process(file_dir_list, mp.batch_size, mp.epochs)
    elif alg_name == "dnn_multi":
        data_out = dnn_multi_process(file_dir_list, mp.batch_size, mp.epochs)
    else:
        exit(-1)

    return data_out


def get_file(input_path):
    dir_list = tf.gfile.ListDirectory(input_path)
    # print(dir_list)
    print("tf_dir:", len(dir_list))
    return dir_list


def load_input_file(mp, input_path):
    file_list = get_file(input_path)
    file_dir_list = []
    for file in file_list:
        if file[:4] == "part":
            filepath = input_path + file
            file_dir_list.append(filepath)
    data = file_process(mp, file_dir_list)
    return data


if __name__ == '__main__':
    train_files = "../../data/train/part-999"
    test_files = "../../data/pred/part-000"
    trainFeature = dnn_multi_process(train_files, batch_size=5, epochs=21)

    with tf.Session() as sess:
        for i in range(100):
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
            res1 = sess.run(trainFeature['cat_feats'])
            print("train:", res1)
