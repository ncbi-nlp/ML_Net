"""
contains the components of ML-Net, including the loss functions
This script contains the codes from:
https://bitbucket.org/raingo-ur/mll-tf/src/master/
https://github.com/ematvey/hierarchical-attention-networks/
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


def bidirectional_rnn(cell_fw, cell_bw, inputs_embedded, input_lengths,
                      scope=None):
    """Bidirecional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "birnn") as scope:
        ((fw_outputs,
          bw_outputs),
         (fw_state,
          bw_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                            cell_bw=cell_bw,
                                            inputs=inputs_embedded,
                                            sequence_length=input_lengths,
                                            dtype=tf.float32,
                                            swap_memory=True,
                                            scope=scope))
        outputs = tf.concat((fw_outputs, bw_outputs), 2)

        def concatenate_state(fw_state, bw_state):
            if isinstance(fw_state, LSTMStateTuple):
                state_c = tf.concat(
                    (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
                state_h = tf.concat(
                    (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
                state = LSTMStateTuple(c=state_c, h=state_h)
                return state
            elif isinstance(fw_state, tf.Tensor):
                state = tf.concat((fw_state, bw_state), 1,
                                  name='bidirectional_concat')
                return state
            elif (isinstance(fw_state, tuple) and
                  isinstance(bw_state, tuple) and
                  len(fw_state) == len(bw_state)):
                # multilayer
                state = tuple(concatenate_state(fw, bw)
                              for fw, bw in zip(fw_state, bw_state))
                return state

            else:
                raise ValueError(
                    'unknown state type: {}'.format((fw_state, bw_state)))

        state = concatenate_state(fw_state, bw_state)
        return outputs, state


def task_specific_attention(inputs, output_size,
                            initializer=layers.xavier_initializer(),
                            activation_fn=tf.tanh, scope=None):
    """
    Performs task-specific attention reduction, using learned
    attention context vector (constant within task of interest).
    Args:
        inputs: Tensor of shape [batch_size, units, input_size]
            `input_size` must be static (known)
            `units` axis will be attended over (reduced from output)
            `batch_size` will be preserved
        output_size: Size of output's inner (feature) dimension
    Returns:
        outputs: Tensor of shape [batch_size, output_dim].
    """
    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn=activation_fn,
                                                  scope=scope)

        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keepdims=True)
        attention_weights = tf.nn.softmax(vector_attn, axis=1)
        weighted_projection = tf.multiply(input_projection, attention_weights)

        outputs = tf.reduce_sum(weighted_projection, axis=1)

        return outputs


"""
definition of the loss functions
"""


def _batch_gather(input, indices, batch_size):
    """
    output[i, ..., j] = input[i, indices[i, ..., j]]
    """
    shape_output = indices.get_shape().as_list()

    shape_input = input.get_shape().as_list()
    shape_input[0] = batch_size
    shape_output[0] = batch_size

    assert len(shape_input) == 2
    batch_base = shape_input[1] * np.arange(shape_input[0])
    batch_base_shape = [1] * len(shape_output)
    batch_base_shape[0] = shape_input[0]

    batch_base = batch_base.reshape(batch_base_shape)
    indices = batch_base + indices

    input = tf.reshape(input, [-1])
    return tf.gather(input, indices)


def _pairwise(label_pairs, logits, batch_size, NUM_CLASSES):
    mapped = _batch_gather(logits, label_pairs, batch_size)
    neg, pos = tf.split(mapped, 2, 2)
    delta = neg - pos

    neg_idx, pos_idx = tf.split(label_pairs, 2, 2)
    _, indices = tf.nn.top_k(tf.stop_gradient(logits), NUM_CLASSES)
    _, ranks = tf.nn.top_k(-indices, NUM_CLASSES)

    delta_nnz = tf.cast(tf.not_equal(neg_idx, pos_idx), tf.float32)
    return delta, delta_nnz


def mll_exp(logits, label_pairs, batch_size, NUM_CLASSES):
    # compute label pairs
    # # batch_size x num_pairs x 2
    # print(logits.get_shape(), "logit shape")
    # print(label_pairs.get_shape(), "label_pairs shape")
    delta, delta_nnz = _pairwise(label_pairs, logits, batch_size, NUM_CLASSES)

    delta_max = tf.reduce_max(delta, 1, keepdims=True)
    delta_max_nnz = tf.nn.relu(delta_max)

    inner_exp_diff = tf.exp(delta - delta_max_nnz)
    inner_exp_diff *= delta_nnz

    inner_sum = tf.reduce_sum(inner_exp_diff, 1, keepdims=True)

    loss = delta_max_nnz + tf.log(tf.exp(-delta_max_nnz) + inner_sum)
    return tf.reduce_mean(loss)
