from __future__ import print_function


from hbconfig import Config
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.layers import core as layers_core



class Seq2Seq:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32

        self.mode = mode
        self.params = params

        self._init_placeholder(features, labels)
        self.build_graph()

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"prediction": self.prediction})
        else:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=self.decoder_train_pred,
                loss=self.loss,
                train_op=self.train_op
            )

    def _init_placeholder(self, features, labels):
        self.encoder_input = features
        if type(features) == dict:
            self.encoder_input = features["input_data"]

        self.encoder_input_lengths = tf.reduce_sum(
            tf.to_int32(tf.not_equal(self.encoder_input, Config.data.PAD_ID)), 1,
            name="encoder_input_lengths")

        if self.mode != tf.estimator.ModeKeys.PREDICT:

            self.decoder_input = labels
            self.decoder_input_lengths = tf.reduce_sum(
                tf.to_int32(tf.not_equal(self.decoder_input, Config.data.PAD_ID)), 1,
                name="decoder_input_lengths")

            decoder_input_shift_1 = tf.slice(self.decoder_input,
                                           [0, 1], [Config.train.batch_size, Config.data.max_seq_length-1])
            pad_tokens = tf.zeros([Config.train.batch_size, 1], dtype=tf.int32)

            self.targets = tf.concat([decoder_input_shift_1, pad_tokens], axis=1)

    def build_graph(self):
        self._build_embed()
        self._build_seq2seq_helper()
        self._build_encoder()
        # self._build_projection()
        self._build_decoder()

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss()
            self._build_optimizer()

    def _build_embed(self):
        with tf.variable_scope ("embeddings", dtype=self.dtype) as scope:

            if Config.model.embed_share:
                embedding = tf.get_variable(
                    "embedding_share", [Config.data.vocab_size, Config.model.embed_dim], self.dtype)

                self.embedding_encoder = embedding
                self.embedding_decoder = embedding
            else:
                self.embedding_encoder = tf.get_variable(
                    "embedding_encoder", [Config.data.vocab_size, Config.model.embed_dim], self.dtype)
                self.embedding_decoder = tf.get_variable(
                    "embedding_decdoer", [Config.data.vocab_size, Config.model.embed_dim], self.dtype)

            self.encoder_emb_inp = tf.nn.embedding_lookup(
                self.embedding_encoder, self.encoder_input)

            if self.mode != tf.estimator.ModeKeys.PREDICT:
                self.decoder_emb_inp = tf.nn.embedding_lookup(
                    self.embedding_decoder, self.decoder_input)

    def _build_seq2seq_helper(self):
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=self.embedding_decoder,
                    start_tokens=tf.fill([Config.train.batch_size], Config.data.START_ID),
                    end_token=Config.data.EOS_ID)
        else:
            self.train_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self.decoder_emb_inp,
                    sequence_length=self.decoder_input_lengths)

    def _build_encoder(self):
        with tf.variable_scope('encoder'):
            cells = self._build_rnn_cells()
            self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                    cells,
                    self.encoder_emb_inp,
                    sequence_length=self.encoder_input_lengths,
                    dtype=tf.float32,
                    time_major=False)

    def _build_projection(self):
        # Projection
        with tf.variable_scope("decoder/output_projection"):
            self.output_layer = layers_core.Dense(
                Config.data.vocab_size, use_bias=False, name="output_projection")

    def _build_decoder(self):

        def decode(helper, scope):
            with tf.variable_scope(scope):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=Config.model.num_units, memory=self.encoder_outputs,
                    memory_sequence_length=self.encoder_input_lengths)

                cells = self._build_rnn_cells()

                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cells,
                    attention_mechanism,
                    attention_layer_size=Config.model.num_units,
                    alignment_history=True,
                    name="attention")

                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, Config.data.vocab_size)

                decoder_initial_state = out_cell.zero_state(Config.train.batch_size, self.dtype)
                decoder_initial_state.clone(cell_state=self.encoder_final_state)

                if self.mode == tf.estimator.ModeKeys.PREDICT:
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=out_cell,
                        helper=helper,
                        initial_state=(decoder_initial_state))

                    outputs = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder,
                        output_time_major=False,
                        impute_finished=True,
                        maximum_iterations=tf.round(tf.reduce_max(self.encoder_input_lengths) * 2)
                    )

                else:
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=out_cell,
                        helper=helper,
                        initial_state=(decoder_initial_state))

                    outputs = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder,
                        output_time_major=False,
                        swap_memory=True)

                return outputs[0]

        with tf.variable_scope('decoder'):

            if self.mode == tf.estimator.ModeKeys.PREDICT:
                self.decoder_pred_outputs = decode(self.pred_helper, 'decode')
                self.prediction = self.decoder_pred_outputs.sample_id
                # self.decoder_train_pred = tf.argmax(self.decoder_train_logits[0], axis=1, name='train/pred_0')
                # self.prediction = tf.argmax(self.decoder_pred_outputs.rnn_output, axis=1, name='prediction')
            else:
                self.decoder_train_outputs = decode(self.train_helper, 'decode')
                self.decoder_train_logits = self.decoder_train_outputs.rnn_output

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.decoder_train_pred = tf.argmax(self.decoder_train_logits[0], axis=1, name='train/pred_0')

    def _build_rnn_cells(self):
        stacked_rnn = []

        for _ in range(Config.model.num_layers):
            single_cell = self._single_cell(Config.model.cell_type, Config.model.dropout)
            stacked_rnn.append(single_cell)

        return tf.nn.rnn_cell.MultiRNNCell(
                cells=stacked_rnn,
                state_is_tuple=True)

    def _single_cell(self, cell_type, dropout):
        if cell_type == "GRU":
            single_cell = tf.contrib.rnn.GRUCell(
                Config.model.num_units,
                forget_bias=1.0)
        elif cell_type == "LSTM":
            single_cell = tf.contrib.rnn.BasicLSTMCell(
                Config.model.num_units,
                forget_bias=1.0)
        elif cell_type == "LAYER_NORM_LSTM":
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                Config.model.num_units,
                forget_bias=1.0,
                layer_norm=True)
        elif cell_type == "NAS":
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                Config.model.num_units)
        else:
            raise ValueError(f"Unknown rnn cell type. {cell_type}")

        if dropout > 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - dropout))

        return single_cell

    def _build_loss(self):
        pad_num = Config.data.max_seq_length - tf.shape(self.decoder_train_logits)[1]
        zero_padding = tf.zeros(
                [Config.train.batch_size, pad_num, Config.data.vocab_size],
                dtype=tf.float32)

        zero_padding_logits = tf.concat([self.decoder_train_logits, zero_padding], axis=1)
        #logits = self.output_layer(zero_padding_logits)

        weight_masks = tf.sequence_mask(
                lengths=self.decoder_input_lengths,
                maxlen=Config.data.max_seq_length,
                dtype=tf.float32, name='masks')

        self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=zero_padding_logits,
                targets=self.targets,
                weights=weight_masks,
                name="loss")

    def _build_optimizer(self):
        self.train_op = layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer='Adam',
            learning_rate=Config.train.learning_rate,
            summaries=['loss', 'learning_rate'],
            name="train_op")
