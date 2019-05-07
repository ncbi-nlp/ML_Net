"""
write tfrecords and pickle fies for model training and test
"""

import data_utils as data_utils
import tensorflow as tf
import string
import re

term_pattern = re.compile('[A-Za-z]+')

stopwords = set(list(string.punctuation))

def concatenateToken(tokens):
    sent = ""
    for token in tokens:
        sent = sent + " " + token
    return sent

def clean_notes(text):
    lines = text.split("[NEWLINE]")
    clean_lines = list()
    for line in lines[1:]:
        raw_dsum = re.sub(r'\[[^\]]+\]', ' ', line)
        raw_dsum = re.sub(r'admission date:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'discharge date:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'date of birth:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'sex:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'service:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'dictated by:.*$', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'completed by:.*$', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'signed electronically by:.*$', ' ', raw_dsum, flags=re.I)
        tokens = [token.lower() for token in re.findall(term_pattern, raw_dsum)]
        tokens = [token for token in tokens if token not in stopwords and len(token) > 1]
        if len(tokens) == 0:
            continue
        clean_lines.append(tokens)

    return clean_lines


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


MAX_ABS_LENGTH = 1500

icd_data_dir = "/"
output_data_dir = "/"

test_tf = output_data_dir + 'icd_test.tfrecords'  # address to save the TFRecords file
train_tf = output_data_dir + '0_icd_train.tfrecords'  # address to save the TFRecords file

test_writer = tf.python_io.TFRecordWriter(test_tf)
train_writer = tf.python_io.TFRecordWriter(train_tf)

raw_text_file = open(icd_data_dir + "/MIMIC_FILTERED_DSUMS", "r").readlines()
train_label_file = open(icd_data_dir + "/training_codes.data", "r").readlines()
test_label_file = open(icd_data_dir + "/testing_codes.data", "r").readlines()

train_len = len(train_label_file)
test_len = len(test_label_file)

label_list = list()
abs_dict_list = list()

for line_num, line in enumerate(raw_text_file[:train_len]):
    if line_num != 0 and line_num % 1280 == 0:
        train_writer.close()
        train_tf = output_data_dir + str(line_num) + "_icd_train.tfrecords"
        train_writer = tf.python_io.TFRecordWriter(train_tf)
        print('\r', line_num, end='', flush=True)

    abs_dict = dict()
    sents = clean_notes(line)

    raw_text = []

    labels = train_label_file[line_num].split("|")[1:]
    label_set = set(labels)
    for sent in sents:
        raw_text.extend(sent)

    raw_text = concatenateToken(raw_text[:MAX_ABS_LENGTH]).strip()

    labels_list = [int(x) for x in label_set.copy()]
    feature = {'labels': _int64_feature(labels_list.copy()),
               'raw_text': _bytes_feature(tf.compat.as_bytes(raw_text))
               }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    abs_dict["raw_text"] = raw_text
    abs_dict_list.append(abs_dict)
    label_list.append(labels_list)
    train_writer.write(example.SerializeToString())
    label_set.clear()
data_utils.save_obj(zip(abs_dict_list, label_list), output_data_dir + "train_abs_label_zip.pickle")
train_writer.close()

label_list.clear()
abs_dict_list.clear()

for line_num, line in enumerate(raw_text_file[train_len:]):
    if line_num % 10 == 0:
        print('\r', "test: ", line_num, end='', flush=True)
    abs_dict = dict()
    sents = clean_notes(line)
    raw_text = []

    labels = test_label_file[line_num].split("|")[1:]
    label_set = set(labels)
    for sent in sents:
        raw_text.extend(sent)

    raw_text = concatenateToken(raw_text[:MAX_ABS_LENGTH]).strip()

    labels_list = [int(x) for x in label_set.copy()]
    feature = {'labels': _int64_feature(labels_list.copy()),
               'raw_text': _bytes_feature(tf.compat.as_bytes(raw_text))
               }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    abs_dict["raw_text"] = raw_text
    abs_dict_list.append(abs_dict)
    label_list.append(labels_list)
    test_writer.write(example.SerializeToString())
    label_set.clear()
data_utils.save_obj(zip(abs_dict_list, label_list), output_data_dir + "test_abs_label_zip.pickle")
test_writer.close()

label_list.clear()
abs_dict_list.clear()
