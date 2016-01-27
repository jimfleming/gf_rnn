import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

class GFLSTMCell(rnn_cell.RNNCell):
    def __init__(self, num_layers, num_blocks):
        self._num_layers = num_layers
        self._num_blocks = num_blocks

    @property
    def input_size(self):
        return self._num_blocks

    @property
    def output_size(self):
        return self._num_blocks

    @property
    def state_size(self):
        return 2 * self._num_blocks

    def __call__(self, inputs, hs_prev, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            initializer = tf.random_uniform_initializer(-0.1, 0.1)

            def get_variable(name, shape):
                return tf.get_variable(name, shape, initializer=initializer, dtype=inputs.dtype)

            c_prev, h_prev = tf.split(1, 2, state)

            W_c = get_variable("W_c", [self.input_size, self.output_size])
            W_i = get_variable("W_i", [self.input_size, self.output_size])
            W_f = get_variable("W_f", [self.input_size, self.output_size])
            W_o = get_variable("W_o", [self.input_size, self.output_size])

            U_c = get_variable("U_c", [self.output_size, self.output_size])
            U_i = get_variable("U_i", [self.output_size, self.output_size])
            U_f = get_variable("U_f", [self.output_size, self.output_size])
            U_o = get_variable("U_o", [self.output_size, self.output_size])

            gs = []
            for i in self._num_layers:
                w_g = get_variable("w_g", [self.input_size, self.output_size])
                u_g = get_variable("u_g", [self.output_size, self.output_size])
                g = tf.sigmoid(tf.matmul(inputs, w_g) + tf.matmul(hs_prev, u_g))
                gf = g * tf.matmul(h_prev, U_c)
                gs.append(gf)
            gfa = tf.concat(gs)

            # c_next = tf.tanh(tf.matmul(inputs, W_c) + tf.matmul(h_prev, U_c)) # original
            c_next = tf.tanh(tf.matmul(inputs, W_c) + tf.reduce_sum(gfa))

            i = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(h_prev, U_i))
            f = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(h_prev, U_f))
            c = tf.mul(f, c_prev) + tf.mul(i, c_next)
            o = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(h_prev, U_o))
            h = tf.mul(tf.tanh(c), o)

            return h, tf.concat(1, [c, h])
