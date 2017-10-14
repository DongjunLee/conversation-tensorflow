from __future__ import print_function


from hbconfig import Config
import tensorflow as tf
from tensorflow.contrib import layers


class Seq2Seq:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.mode = mode
        self.params = params

        self.input_data = features
        if type(features) == dict:
            self.input_data = features["input_data"]
        self.input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.input_data, Config.data.PAD_ID)), 1)

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.build_graph()

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"prediction": self.prediction})
        else:
            self.outputs = labels
            self.output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.outputs, Config.data.PAD_ID)), 1)

            self.build_graph()

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=self.decoder_pred_outputs.sample_id,
            loss=self.loss,
            train_op=self.train_op
        )

    def build_graph(self):
        self._build_embed()
        self._build_seq2seq_helper()
        self._build_encoder()
        self._build_decoder()

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss()
            self._build_optimizer()

    def _build_embed(self):
        self.input_embed = layers.embed_sequence(
            self.input_data,
            vocab_size=Config.data.vocab_size,
            embed_dim=Config.model.embed_dim,
            scope='embed')

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.output_embed = layers.embed_sequence(
                self.outputs,
                vocab_size=Config.data.vocab_size,
                embed_dim=Config.model.embed_dim,
                scope='embed', reuse=True)

    def _build_seq2seq_helper(self):
        with tf.variable_scope('embed', reuse=True):
            embeddings = tf.get_variable('embeddings')

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=embeddings,
                    start_tokens=tf.fill([Config.train.batch_size], Config.data.START_ID),
                    end_token=Config.data.EOS_ID)
        else:
            self.train_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self.output_embed,
                    sequence_length=self.output_lengths)

    def _build_encoder(self):
        with tf.variable_scope('encoder'):
            cells = self._build_rnn_cells()
            self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                    cells,
                    self.input_embed,
                    sequence_length=self.input_lengths,
                    dtype=tf.float32,
                    time_major=False)

    def _build_decoder(self):

        def decode(helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=Config.model.num_units, memory=self.encoder_outputs,
                    memory_sequence_length=self.input_lengths)

                cells = self._build_rnn_cells()

                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cells, attention_mechanism, attention_layer_size=Config.model.num_units / 2)

                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, Config.data.vocab_size, reuse=reuse
                )

                decoder_initial_state = out_cell.zero_state(Config.train.batch_size, tf.float32)
                decoder_initial_state.clone(cell_state=self.encoder_final_state)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=(decoder_initial_state))

                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=tf.round(tf.reduce_max(self.input_lengths) * 2)
                )
                return outputs[0]

        with tf.variable_scope('decoder'):

            if self.mode == tf.estimator.ModeKeys.PREDICT:
                self.decoder_pred_outputs = decode(self.pred_helper, 'decode')
                self.prediction = self.decoder_pred_outputs.sample_id
            else:
                self.decoder_train_outputs = decode(self.train_helper, 'decode')
                self.decoder_train_logits = self.decoder_train_outputs.rnn_output

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            tf.argmax(self.decoder_train_logits[0], axis=1, name='training/pred_0')

    def _build_rnn_cells(self):
        stacked_rnn = []
        for _ in range(Config.model.num_layers):
            single_cell = tf.contrib.rnn.GRUCell(num_units=Config.model.num_units)
            stacked_rnn.append(single_cell)
        return tf.nn.rnn_cell.MultiRNNCell(
                cells=stacked_rnn,
                state_is_tuple=True)

    def _build_loss(self):
        pad_num = Config.data.max_seq_length - tf.shape(self.decoder_train_logits)[1]
        zero_padding = tf.zeros(
                [Config.train.batch_size, pad_num, Config.data.vocab_size],
                dtype=tf.float32)

        logits = tf.concat([self.decoder_train_logits, zero_padding], axis=1)

        weight_masks = tf.sequence_mask(
                lengths=self.output_lengths,
                maxlen=Config.data.max_seq_length,
                dtype=tf.float32, name='masks')

        self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=self.outputs,
                weights=weight_masks,
                name="loss")

    def _build_optimizer(self):
        self.train_op = layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer='Adam',
            learning_rate=Config.train.learning_rate,
            summaries=['loss', 'learning_rate'])
