from __future__ import print_function


from hbconfig import Config
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


class Seq2Seq:

    def __init__(self):
        pass

    def make_estimator_spec(self, mode, features, labels, params):
        self._create_placeholder(features)
        self.build_graph()

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=self.pred_outputs.sample_id,
            loss=self.loss,
            train_op=self.train_op
        )

    def _create_placeholder(self, features):
        self.X = features['input']
        self.y = features['output']

    def build_graph(self):
        self._create_embed()
        self._create_seq2seq_helper()
        self._create_encoder()
        self._create_decoder()
        self._create_loss()
        self._create_optimizer()
        self._create_predictions()

    def _create_embed(self):
        self.start_tokens = tf.zeros([Config.model.BATCH_SIZE], dtype=tf.int64)
        self.train_output = tf.concat([tf.expand_dims(self.start_tokens, 1), self.y], 1)

        self.input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.X, 1)), 1)
        self.output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.train_output, 1)), 1)

        self.input_embed = layers.embed_sequence(
            self.X,
            vocab_size=Config.model.VOCAB_SIZE,
            embed_dim=Config.model.EMBED_DIM,
            scope='embed')
        self.output_embed = layers.embed_sequence(
            self.train_output,
            vocab_size=Config.model.VOCAB_SIZE,
            embed_dim=Config.model.EMBED_DIM,
            scope='embed', reuse=True)

    def _create_seq2seq_helper(self):
        with tf.variable_scope('embed', reuse=True):
            embeddings = tf.get_variable('embeddings')

        self.train_helper = tf.contrib.seq2seq.TrainingHelper(
            self.output_embed, self.output_lengths)
        self.pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embeddings,
            start_tokens=tf.to_int32(self.start_tokens),
            end_token=1)

    def _create_encoder(self):
        cells = self._create_rnn_cells()
        self.encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cells, self.input_embed, dtype=tf.float32)

    def _create_decoder(self):

        def decode(helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=Config.model.NUM_UNITS, memory=self.encoder_outputs,
                    memory_sequence_length=self.input_lengths)

                cells = self._create_rnn_cells()

                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cells, attention_mechanism, attention_layer_size=Config.model.NUM_UNITS / 2)

                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, Config.model.VOCAB_SIZE, reuse=reuse
                )

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(
                        dtype=tf.float32, batch_size=Config.model.BATCH_SIZE))

                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=False,
                    impute_finished=True, maximum_iterations=Config.model.OUTPUT_MAX_LENGTH
                )
                return outputs[0]

        self.train_outputs = decode(self.train_helper, 'decode')
        self.pred_outputs = decode(self.pred_helper, 'decode', reuse=True)

        tf.identity(self.train_outputs.sample_id[0], name='train_pred')

    def _create_rnn_cells(self):
        stacked_rnn = []
        for _ in range(Config.model.NUM_LAYERS):
            single_cell = tf.contrib.rnn.GRUCell(num_units=Config.model.NUM_UNITS)
            stacked_rnn.append(single_cell)
        return tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)

    def _create_loss(self):
        weights = tf.to_float(tf.not_equal(self.train_output[:, :-1], 1))
        self.loss = tf.contrib.seq2seq.sequence_loss(
            self.train_outputs.rnn_output, self.y, weights=weights)

    def _create_optimizer(self):
        self.train_op = layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer='Adam',
            learning_rate=Config.model.LEARNING_RATE,
            summaries=['loss', 'learning_rate'])

    def _create_predictions(self):
        tf.identity(self.pred_outputs.sample_id[0], name='predictions')
