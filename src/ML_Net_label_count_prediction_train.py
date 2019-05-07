import tensorflow as tf
import data_utils as data_utils
import time
import datetime
import os
import os.path as osp
import glob
from ML_Net import ML_elmo
import metrics as metric
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logs_path = "tensorflow_logs/label_count/"
data_dir = "tf_data/"
trained_model_path = ""
# Parameters
# ================================================== #

# Data loading params
tf.app.flags.DEFINE_integer("NUM_CLASSES", 7042, "NUM_CLASSES")
tf.app.flags.DEFINE_integer("MAX_LABELS_PERMITTED", 70, "maximum of labels permitted by label decision network ")
tf.app.flags.DEFINE_integer("MAX_PAIRS", 2000, "sample at most MAX_PAIRS from the Cartesian product (negative sampling)")

# Model Hyperparameters
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
tf.app.flags.DEFINE_integer("hidden_size", 50, "the hidden size of rnn unit")

# Training parameters
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch Size ")
tf.app.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs")

# Misc Parameters
tf.app.flags.DEFINE_integer("eval_model_step", 200, "evaluate every xx step")

FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))
print("")

num_preprocess_threads = 4
min_after_dequeue = 100 # 1000 per file
examples_queue = tf.RandomShuffleQueue(
    capacity=min_after_dequeue + 16 * FLAGS.batch_size,
    min_after_dequeue=min_after_dequeue,
    dtypes=[tf.string])
files = glob.glob(osp.join(data_dir, '*_train.tfrecords'))
filename_queue = tf.train.string_input_producer(files,num_epochs=FLAGS.num_epochs,shuffle=True, capacity=10)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
enqueue_ops = []
enqueue_ops.append(examples_queue.enqueue([serialized_example]))

tf.train.queue_runner.add_queue_runner(
    tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
example_serialized = examples_queue.dequeue()

outputs = []
keys = {}.keys()
for _ in range(num_preprocess_threads):
    data = data_utils.__parse_example_proto_with_elmo(example_serialized, FLAGS.NUM_CLASSES, FLAGS.MAX_PAIRS)
    keys = data.keys()
    outputs.append(list(data.values()))

res = tf.train.batch_join(outputs, batch_size=FLAGS.batch_size, capacity=2 * num_preprocess_threads * FLAGS.batch_size)
res_d = {}
for key, value in zip(keys, res):
    res_d[key] = value

session_conf = tf.ConfigProto(allow_soft_placement=True)
session_conf.gpu_options.allow_growth = True
sess = tf.Session(config=session_conf)

with sess.as_default():

    ML_net = ML_elmo(
        NUM_CLASSES=FLAGS.NUM_CLASSES,
        MAX_LABELS_PERMITTED=FLAGS.MAX_LABELS_PERMITTED,
        batch_size=FLAGS.batch_size,
        hidden_size=FLAGS.hidden_size,
        MAX_PAIRS=FLAGS.MAX_PAIRS
    )
    saver = tf.train.Saver(max_to_keep=15)
    current_loss = ML_net.lcnt_loss
    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads_and_vars = optimizer.compute_gradients(current_loss)
    train_op = tf.contrib.layers.optimize_loss(current_loss, global_step=global_step, learning_rate=FLAGS.learning_rate,
                                               optimizer="Adam")

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(logs_path, "runs", timestamp))

    print("Writing to {}\n".format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    para_writer = open(out_dir + "/parameters.txt", "w")
    for attr, value in sorted(FLAGS.__flags.items()):
        para_writer.write(data_dir + "\n")
        para_writer.write("{}={}".format(attr.upper(), value.value))
        para_writer.write("\n")
    para_writer.close()

    def eval_step():
        print("----\nprediction begin!\n----")
        # Tensors we want to evaluate
        prediction_logits_list = list()
        prediction_counts_list = list()
        eval_data = data_utils.load_obj("tf_data/test_abs_label_zip.pickle")

        train_x, train_y = zip(*eval_data)
        total_test_size = len(train_x)
        # print(total_test_size, "total_test_size")
        remain_test_size = total_test_size % FLAGS.batch_size
        total_epoch = int(total_test_size / FLAGS.batch_size)
        batches = data_utils.batch_iter_eval(zip(train_x, train_y), batch_size=FLAGS.batch_size)

        gold_labels_list = list()
        for x_batch, y_batch in batches:
            gold_labels_list.extend(y_batch)
            # print(y_batch, "gold_label")
            # print(np.where(y_batch == 1)[0],"gold label index")
            text_list = list()
            for abs_dict in x_batch:
                text_list.append(abs_dict["raw_text"])

            feed_dict = {
                ML_net.input_x: text_list,
                ML_net.dropout_keep_prob: 1
            }
            batch_logits, batch_count = sess.run([ML_net.logits, ML_net.predictions_count], feed_dict)
            prediction_logits_list.extend(batch_logits)
            prediction_counts_list.extend(batch_count)

        prediction_logits_list_update = prediction_logits_list[:total_epoch * FLAGS.batch_size].copy()
        prediction_logits_list_update.extend(prediction_logits_list[-remain_test_size:])
        prediction_counts_list_update = prediction_counts_list[:total_epoch * FLAGS.batch_size].copy()
        prediction_counts_list_update.extend(prediction_counts_list[-remain_test_size:])
        gold_labels_list_update = gold_labels_list[:total_epoch * FLAGS.batch_size].copy()
        gold_labels_list_update.extend(gold_labels_list[-remain_test_size:])

        prediction_logits_arr = np.asarray(prediction_logits_list_update)
        prediction_counts_list = np.asarray(prediction_counts_list_update)

        p, r, f, p_list, g_list = metric.get_p_r_f_jamia(logits=prediction_logits_arr, counts=prediction_counts_list,
                                        labels=gold_labels_list_update)
        print(global_step.eval(), "step")
        print(p, "\nprecision")
        print(r, "recall")
        print(f, "f-measure")


    def train_step(feed):
        """
        A single training step
        """
        x_feed = feed[0]
        y_feed = feed[1]
        feed_dict = {ML_net.input_x: x_feed[0],
                     ML_net.input_y_label_pairs: y_feed[0],
                     ML_net.input_y_label_map: y_feed[1],
                     ML_net.dropout_keep_prob: FLAGS.dropout_keep_prob
                     }

        _, step,  loss = sess.run(
            [train_op, global_step,  current_loss],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}".format(time_str, step, loss))

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    tf.tables_initializer().run()

    saver.restore(sess, trained_model_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    current_step = 0
    try:
        while not coord.should_stop():
            train_data = sess.run(res_d)
            text = tuple(train_data['raw_text'].flatten().tolist())
            text = [s.decode("utf-8").strip() for s in text]
            label_pairs = train_data['label_pair']
            label_map = train_data['label_map']

            x_feed = [text]
            y_feed = [label_pairs, label_map]
            feed = [x_feed, y_feed]
            train_step(feed)

            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.eval_model_step == 0:
                eval_step()

    except tf.errors.OutOfRangeError:
        print("Done training")
    finally:
        coord.request_stop()
    coord.join(threads)

    # print("Model saved in path: %s" % save_path)
    print("\nend!")
