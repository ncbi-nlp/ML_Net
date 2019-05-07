import tensorflow as tf
import ML_Net_components as ML_component
from tensorflow.contrib import rnn
import tensorflow_hub as hub

class ML_Net(object):

    def __init__(self, NUM_CLASSES, hidden_size, MAX_LABELS_PERMITTED, batch_size, MAX_PAIRS):
        """init all hyperparameter here"""
        # set hyperparamter

        self.NUM_CLASSES = NUM_CLASSES
        self.MAX_PAIRS = MAX_PAIRS
        self.MAX_LABELS_PERMITTED = MAX_LABELS_PERMITTED
        self.batch_size = batch_size
        self.input_x = tf.placeholder(tf.string, shape=(None), name='inputs')
        self.input_y_label_pairs = tf.placeholder(tf.int32, [None, self.MAX_PAIRS, 2],
                                                  name="input_y_label_pairs")  # convert to
        self.input_y_label_map = tf.placeholder(tf.float32, [None, self.NUM_CLASSES],
                                                name="input_y_label_map")  # convert to

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope("encoder"):
            module_url = "https://tfhub.dev/google/elmo/2"
            elmo_embed = hub.Module(module_url, trainable=True)
            embeddings = elmo_embed(self.input_x, signature="default", as_dict=True)["elmo"]

        with tf.name_scope("bidirectional_LSTM"):
            lstm_fw_cell = rnn.BasicLSTMCell(hidden_size)  # forward direction cell
            lstm_bw_cell = rnn.BasicLSTMCell(hidden_size)  # backward direction cell
            if self.dropout_keep_prob is not None:
                lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
            rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embeddings,
                                                                      dtype=tf.float32)

            self.output_rnn_bi = tf.concat(rnn_outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]

        with tf.name_scope("attention_layer"):

            hidden_size = self.output_rnn_bi.shape[2].value  # D value - hidden size of the RNN layer

            # Trainable parameters
            w_omega = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1))
            b_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
            u_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))

            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
                v = tf.tanh(tf.tensordot(self.output_rnn_bi, w_omega, axes=1) + b_omega)

            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
            alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

            # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
            self.sentence_level_output = tf.reduce_sum(self.output_rnn_bi * tf.expand_dims(alphas, -1), 1)

        with tf.name_scope("output"):
            sentence_level_output_norm = tf.contrib.layers.batch_norm(self.sentence_level_output, center=True, scale=True, is_training=True, epsilon=1e-12)
            self.logits = tf.contrib.layers.fully_connected(sentence_level_output_norm, self.NUM_CLASSES,activation_fn=tf.nn.relu)   # [batch_size,num_classes]
            # self.final_output = tf.nn.sigmoid(self.logits, name = "attention_final_output")

        with tf.name_scope("loss"):
            self.loss = ML_component.mll_exp(self.logits, self.input_y_label_pairs, self.batch_size,NUM_CLASSES)
            ##L2 normalization

        with tf.name_scope("count_loss"):
            num_bins = self.MAX_LABELS_PERMITTED
            sentence_level_output = tf.stop_gradient(sentence_level_output_norm)
            cnt_h1 = tf.contrib.layers.fully_connected(sentence_level_output, num_outputs = NUM_CLASSES, activation_fn=tf.nn.relu)
            # cnt_h1 = tf.contrib.layers.batch_norm(cnt_h1,center=True, scale=True, is_training=True, epsilon=1e-12)
            cnt_h2 = tf.contrib.layers.fully_connected(cnt_h1, num_outputs = NUM_CLASSES, activation_fn=tf.nn.relu)
            # cnt_h2 = tf.contrib.layers.batch_norm(cnt_h2, center=True, scale=True, is_training=True, epsilon=1e-12)
            cnt_h3 = tf.contrib.layers.fully_connected(cnt_h2, num_outputs=128, activation_fn=tf.nn.relu)
            # cnt_h3 = tf.contrib.layers.batch_norm(cnt_h3, center=True, scale=True, is_training=True, epsilon=1e-12)
            self.lcnt = tf.contrib.layers.fully_connected(cnt_h3,num_bins, activation_fn=None)
            self.predictions_count = tf.argmax(self.lcnt, axis=1, name="predictions_count")
            label_count = tf.reduce_sum(self.input_y_label_map, 1)
            tails = num_bins * tf.ones_like(label_count)
            bins = tf.where(label_count > num_bins, tails, label_count)
            labels = bins - 1
            labels = tf.cast(labels, tf.int64)

            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.lcnt, labels=labels)
            self.lcnt_loss = tf.reduce_mean(xent, name= 'count_loss')