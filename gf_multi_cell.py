import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

class GFMultiRNNCell(rnn_cell.RNNCell):
    def __init__(self, cells):
        if not cells:
            raise ValueError("Must specify at least one cell for GFMultiRNNCell.")

        for i in xrange(len(cells) - 1):
            if cells[i + 1].input_size != cells[i].output_size:
                raise ValueError("In GFMultiRNNCell, the input size of each next"
                                " cell must match the output size of the previous one."
                                " Mismatched output size in cell %d." % i)

        self._cells = cells

    @property
    def input_size(self):
        return self._cells[0].input_size

    @property
    def output_size(self):
        return self._cells[-1].output_size

    @property
    def state_size(self):
        return sum([cell.state_size for cell in self._cells])

    def __call__(self, inputs, hs_prev, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):    # "GFMultiRNNCell"
            cur_state_pos = 0
            cur_inp = inputs
            new_states = []
            new_hs = []
            for i, cell in enumerate(self._cells):
                with tf.variable_scope("Cell%d" % i):
                    cur_state = tf.slice(state, [0, cur_state_pos], [-1, cell.state_size])
                    cur_state_pos += cell.state_size
                    cur_inp, new_state = cell(cur_inp, hs_prev, cur_state)
                    new_states.append(new_state)
                    new_hs.append(cur_inp)
        return cur_inp, tf.concat(1, new_hs), tf.concat(1, new_states)
