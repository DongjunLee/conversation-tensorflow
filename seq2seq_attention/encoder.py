
import tensorflow as tf


__all__ = [
    "Encoder"
]



class Encoder:
    """Encoder class is Mutil-layer Recurrent Neural Networks
    The 'Encoder' usually encode the sequential input vector.
    """

    UNI_ENCODER_TYPE = "uni"
    BI_ENCODER_TYPE = "bi"

    RNN_GRU_CELL = "gru"
    RNN_LSTM_CELL = "lstm"
    RNN_LAYER_NORM_LSTM_CELL = "layer_norm_lstm"
    RNN_NAS_CELL = "nas"

    def __init__(self, encoder_type="uni", num_layers=4,
                 cell_type="GRU", num_units=512, dropout=0.8,
                 dtype=tf.float32):
        """Contructs an 'Encoder' instance.
        * Args:
            encoder_type: rnn encoder_type (uni, bi)
            num_layers: number of RNN cell composed sequentially of multiple simple cells.
            input_vector: RNN Input vectors.
            sequence_length: batch element's sequence length
            cell_type: RNN cell types (lstm, gru, layer_norm_lstm, nas)
            num_units: the number of units in cell
            dropout: set prob operator adding dropout to inputs of the given cell.
            dtype: the dtype of the input
        * Returns:
            Encoder instance
        """

        self.encoder_type = encoder_type
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.num_units = num_units
        self.dropout = dropout
        self.dtype = dtype

    def build(self, input_vector, sequence_length, scope=None):
        if self.encoder_type == self.UNI_ENCODER_TYPE:
            self.cells = self._create_rnn_cells()

            return self.unidirectional_rnn(input_vector, sequence_length, scope=scope)
        elif self.encoder_type == self.BI_ENCODER_TYPE:

            self.num_layers = int(self.num_layers / 2)
            if self.num_layers == 0:
                self.num_layers = 1

            self.cells_fw = self._create_rnn_cells(is_list=True)
            self.cells_bw = self._create_rnn_cells(is_list=True)

            return self.bidirectional_rnn(input_vector, sequence_length, scope=scope)
        else:
            raise ValueError(f"Unknown encoder_type {self.encoder_type}")

    def unidirectional_rnn(self, input_vector, sequence_length, scope=None):
        return tf.nn.dynamic_rnn(
                self.cells,
                input_vector,
                sequence_length=sequence_length,
                dtype=self.dtype,
                time_major=False,
                swap_memory=True,
                scope=scope)

    def bidirectional_rnn(self, input_vector, sequence_length, scope=None):
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                self.cells_fw,
                self.cells_bw,
                input_vector,
                sequence_length=sequence_length,
                dtype=self.dtype,
                scope=scope)

        if self.cell_type == self.RNN_LSTM_CELL:
            encoder_final_state_c = tf.concat((output_state_fw[-1].c, output_state_bw[-1].c), axis=1)
            encoder_final_state_h = tf.concat((output_state_fw[-1].h, output_state_bw[-1].h), axis=1)
            encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)
        else:
            encoder_final_state = tf.concat((output_state_fw[-1], output_state_bw[-1]), axis=1)

        return outputs, encoder_final_state

    def _create_rnn_cells(self, is_list=False):
        """Contructs stacked_rnn with num_layers
        * Args:
            is_list: flags for stack bidirectional. True=stack bidirectional, False=unidirectional
        * Returns:
            stacked_rnn
        """

        stacked_rnn = []
        for _ in range(self.num_layers):
            single_cell = self._rnn_single_cell()
            stacked_rnn.append(single_cell)

        if is_list:
            return stacked_rnn
        else:
            return tf.nn.rnn_cell.MultiRNNCell(
                    cells=stacked_rnn,
                    state_is_tuple=True)

    def _rnn_single_cell(self):
        """Contructs rnn single_cell"""

        if self.cell_type == self.RNN_GRU_CELL:
            single_cell = tf.contrib.rnn.GRUCell(
                self.num_units,
                reuse=tf.get_variable_scope().reuse)
        elif self.cell_type == self.RNN_LSTM_CELL:
            single_cell = tf.contrib.rnn.BasicLSTMCell(
                self.num_units,
                forget_bias=1.0,
                reuse=tf.get_variable_scope().reuse)
        elif self.cell_type == self.RNN_LAYER_NORM_LSTM_CELL:
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.num_units,
                forget_bias=1.0,
                layer_norm=True,
                reuse=tf.get_variable_scope().reuse)
        elif self.cell_type == self.RNN_NAS_CELL:
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.num_units)
        else:
            raise ValueError(f"Unknown rnn cell type. {self.cell_type}")

        if self.dropout > 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - self.dropout))

        return single_cell
