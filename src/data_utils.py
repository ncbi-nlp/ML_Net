"""
utilities functions of ML_Net
"""

import tensorflow as tf
import pickle

def _lm2lp(label_map, MAX_PAIRS):
    pos = tf.reshape(tf.where(label_map), [-1])
    neg = tf.reshape(tf.where(tf.logical_not(label_map)), [-1])

    neg_pos = tf.meshgrid(neg, pos, indexing='ij')
    neg_pos_mat = tf.reshape(tf.transpose(tf.stack(neg_pos)), [-1, 2])  # all the pairs for neg and pos by their index
    neg_pos_rand = tf.random_shuffle(neg_pos_mat)
    neg_pos_pad = tf.pad(neg_pos_rand, [[0, MAX_PAIRS], [0, 0]])
    neg_pos_res = tf.slice(neg_pos_pad, [0, 0], [MAX_PAIRS, -1])

    # MAX_PAIRS x 2
    return neg_pos_res


def batch_iter_eval(zip_text_mesh, batch_size):
    """
        Generates a batch iterator for a evaluation dataset.
    """
    text_list, mesh_list = zip(*zip_text_mesh)
    text_list = list(text_list)
    mesh_list = list(mesh_list)
    data_length = len(text_list)
    num_batches_per_epoch = int((data_length - 1) / batch_size) + 1

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        if (batch_num + 1) * batch_size >= data_length:
            yield text_list[data_length - batch_size:], mesh_list[data_length - batch_size:]
        else:
            end_index = (batch_num + 1) * batch_size
            yield text_list[start_index:end_index], mesh_list[start_index:end_index]


def save_obj(obj, file_address):
    with open(file_address, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_address):
    with open(file_address, 'rb') as f:
        return pickle.load(f)


def __parse_example_proto_with_elmo(example_serialized, NUM_CLASS, MAX_PAIRS):
    feature_map = {
        'raw_text': tf.FixedLenFeature([], tf.string),
        'labels': tf.VarLenFeature(dtype=tf.int64)
    }
    features = tf.parse_single_example(example_serialized, features=feature_map)

    raw_text = features['raw_text']

    labels = features['labels']
    label_map = tf.sparse_to_indicator(labels, NUM_CLASS)
    label_pairs = _lm2lp(label_map, MAX_PAIRS)
    label_map = tf.cast(label_map, tf.float32)
    label_map.set_shape([NUM_CLASS])

    label_dict = {}
    label_dict['label_pair'] = label_pairs
    label_dict['label_map'] = label_map
    label_dict['raw_text'] = raw_text
    return label_dict
