
import tensorflow as tf


__all__ = [
    "Attention", "Decoder"
]


class Attention:
    """Attention class"""

    BAHDANAU_MECHANISM = "bahdanau"
    NORMED_BAHDANAU_MECHANISM = "normed_bahdanau"

    LUONG_MECHANISM = "luong"
    SCALED_LUONG_MECHANISM = "scaled_luong"

    def __init__(self,
                 attention_mechanism="bahdanau",
                 encoder_type="bi",
                 num_units=512,
                 memory=None,
                 memory_sequence_length=None):

        assert memory is not None
        assert memory_sequence_length is not None

        self.attention_mechanism = attention_mechanism
        self.encoder_type = encoder_type
        self.num_units = num_units
        self.memory = memory
        self.memory_sequence_length = memory_sequence_length

    def wrap(self, decoder_cell, alignment_history=True):
        with tf.variable_scope("attention") as scope:
            attention_layer_size = self.num_units
            return tf.contrib.seq2seq.AttentionWrapper(
                        decoder_cell,
                        self._create_mechanism(),
                        attention_layer_size=attention_layer_size,
                        alignment_history=alignment_history,
                        name=f"{self.attention_mechanism}-mechanism")

    def _create_mechanism(self):

        if self.attention_mechanism == "bahdanau":
            return tf.contrib.seq2seq.BahdanauAttention(
                    self.num_units,
                    self.memory,
                    memory_sequence_length=self.memory_sequence_length)

        elif self.attention_mechanism == "normed_bahdanau":
            return tf.contrib.seq2seq.BahdanauAttention(
                    self.num_units,
                    self.memory,
                    memory_sequence_length=self.memory_sequence_length,
                    normalize=True)

        elif self.attention_mechanism == "luong":
            return tf.contrib.seq2seq.LuongAttention(
                    self.num_units,
                    self.memory,
                    memory_sequence_length=self.memory_sequence_length)

        elif self.attention_mechanism == "scaled_luong":
            return tf.contrib.seq2seq.LuongAttention(
                    self.num_units,
                    self.memory,
                    memory_sequence_length=self.memory_sequence_length,
                    scale=True)

        else:
            raise ValueError(f"Unknown attention mechanism {self.attention_mechanism}")


class Decoder:
    """Decoder class"""

    UNI_ENCODER_TYPE = "uni"
    BI_ENCODER_TYPE = "bi"

    RNN_GRU_CELL = "gru"
    RNN_LSTM_CELL = "lstm"
    RNN_LAYER_NORM_LSTM_CELL = "layer_norm_lstm"
    RNN_NAS_CELL = "nas"

    def __init__(self,
                 cell_type="lstm",
                 dropout=0.8,
                 encoder_type="uni",
                 num_layers=None,
                 num_units=None,
                 sampling_probability=0.4,
                 mode=tf.estimator.ModeKeys.TRAIN,
                 dtype=tf.float32):

        self.cell_type = cell_type
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.num_layers = num_layers
        self.num_units = num_units
        self.sampling_probability = sampling_probability

        if encoder_type == self.BI_ENCODER_TYPE:
            self.num_units *= 2
            self.num_layers = int(self.num_layers / 2)
            if self.num_layers == 0:
                self.num_layers = 1
        self.mode = mode
        self.dtype = dtype

    def set_attention_then_project(self,
                                   attention_mechanism="bahdanau",
                                   beam_width=0,
                                   memory=None,
                                   memory_sequence_length=None,
                                   vocab_size=None):

        self.beam_width = beam_width

        cells = self._create_rnn_cells()

        attention = Attention(
                        attention_mechanism=attention_mechanism,
                        encoder_type=self.encoder_type,
                        num_units=self.num_units,
                        memory=memory,
                        memory_sequence_length=memory_sequence_length)
        alignment_history = (self.mode == tf.estimator.ModeKeys.PREDICT
                and self.beam_width == 0)

        attn_cell = attention.wrap(cells, alignment_history=alignment_history)
        self.out_cell = tf.contrib.rnn.OutputProjectionWrapper(
            attn_cell, vocab_size)

        self.maximum_iterations = tf.round(tf.reduce_max(memory_sequence_length) * 2)

    def set_initial_state(self, batch_size, encoder_final_state):
        if self.mode == tf.estimator.ModeKeys.PREDICT and self.beam_width > 0:
            decoder_start_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, self.beam_width)
            self.decoder_initial_state = self.out_cell.zero_state(batch_size * self.beam_width, self.dtype)
            self.decoder_initial_state = self.decoder_initial_state.clone(cell_state=decoder_start_state)
        else:
            self.decoder_initial_state = self.out_cell.zero_state(batch_size, self.dtype)
            self.decoder_initial_state = self.decoder_initial_state.clone(cell_state=encoder_final_state)

    def build(self,
            inputs=None,
            sequence_length=None,
            embedding=None,
            start_tokens=None,
            end_token=None,
            length_penalty_weight=1.0):

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            assert inputs is not None
            assert sequence_length is not None

            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs=inputs,
                    sequence_length=sequence_length,
                    embedding=embedding,
                    sampling_probability=self.sampling_probability)

            return self._basic_decoder(helper)

        else:
            assert embedding is not None
            assert start_tokens is not None
            assert end_token is not None

            if self.mode == tf.estimator.ModeKeys.PREDICT and self.beam_width > 0:
                return self._beam_search_decoder(
                        embedding, start_tokens, end_token, length_penalty_weight)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        embedding=embedding,
                        start_tokens=start_tokens,
                        end_token=end_token)
                return self._basic_decoder(helper)

    def _basic_decoder(self, helper):
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.out_cell,
            helper=helper,
            initial_state=self.decoder_initial_state)

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=True,
                swap_memory=True)
        else:
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.maximum_iterations)

        return outputs

    def _beam_search_decoder(self, embedding, start_tokens, end_token, length_penalty_weight):
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=self.out_cell,
            embedding=embedding,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=self.decoder_initial_state,
            beam_width=self.beam_width,
            length_penalty_weight=length_penalty_weight)

        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            output_time_major=False,
            impute_finished=False,
            maximum_iterations=self.maximum_iterations)
        return outputs

    def _create_rnn_cells(self):
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

        if self.num_layers == 1:
            return stacked_rnn[0]
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


