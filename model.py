from __future__ import print_function


from hbconfig import Config
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.layers import core as layers_core

from seq2seq_attention.encoder import Encoder
from seq2seq_attention.decoder import Decoder



class Conversation:

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
        # set beam_width config or default is 0
        self.beam_width = Config.predict.get('beam_width', 0)

        self._build_embed()
        self._build_encoder()
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
                    "embedding_decoder", [Config.data.vocab_size, Config.model.embed_dim], self.dtype)

            self.encoder_emb_inp = tf.nn.embedding_lookup(
                self.embedding_encoder, self.encoder_input)

            if self.mode != tf.estimator.ModeKeys.PREDICT:
                self.decoder_emb_inp = tf.nn.embedding_lookup(
                    self.embedding_decoder, self.decoder_input)

    def _build_encoder(self):
        with tf.variable_scope('encoder'):
            encoder = Encoder(
                    encoder_type=Config.model.encoder_type,
                    num_layers=Config.model.num_layers,
                    cell_type=Config.model.cell_type,
                    num_units=Config.model.num_units,
                    dropout=Config.model.dropout)

            self.encoder_outputs, self.encoder_final_state = encoder.build(
                    input_vector=self.encoder_emb_inp,
                    sequence_length=self.encoder_input_lengths)

            if self.mode == tf.estimator.ModeKeys.PREDICT and self.beam_width > 0:
                self.encoder_outputs = tf.contrib.seq2seq.tile_batch(
                        self.encoder_outputs, self.beam_width)
                self.encoder_input_lengths = tf.contrib.seq2seq.tile_batch(
                        self.encoder_input_lengths, self.beam_width)

    def _build_decoder(self):

        with tf.variable_scope('decoder'):

            decoder = Decoder(
                        cell_type=Config.model.cell_type,
                        dropout=Config.model.dropout,
                        encoder_type=Config.model.encoder_type,
                        num_layers=Config.model.num_layers,
                        num_units=Config.model.num_units,
                        mode=self.mode,
                        dtype=self.dtype)

            decoder.set_attention_then_project(
                        attention_mechanism=Config.model.attention_mechanism,
                        beam_width=self.beam_width,
                        memory=self.encoder_outputs,
                        memory_sequence_length=self.encoder_input_lengths,
                        vocab_size=Config.data.vocab_size)
            decoder.set_initial_state(Config.train.batch_size, self.encoder_final_state)

            decoder_outputs = decoder.build(
                                inputs=self.decoder_emb_inp,
                                sequence_length=self.decoder_input_lengths,
                                embedding=self.embedding_decoder,
                                start_tokens=tf.fill([Config.train.batch_size], Config.data.START_ID),
                                end_token=Config.data.EOS_ID,
                                length_penalty_weight=Config.predict.length_penalty_weight)

            if self.mode == tf.estimator.ModeKeys.PREDICT:
                if self.beam_width > 0:
                    self.prediction = decoder_outputs.predicted_ids
                else:
                    self.prediction = decoder_outputs.sample_id
            else:
                self.decoder_train_logits = decoder_outputs.rnn_output
        self.decoder_train_pred = tf.argmax(self.decoder_train_logits[0], axis=1, name='train/pred_0')

    def _build_loss(self):
        pad_num = Config.data.max_seq_length - tf.shape(self.decoder_train_logits)[1]
        zero_padding = tf.zeros(
                [Config.train.batch_size, pad_num, Config.data.vocab_size],
                dtype=tf.float32)

        zero_padding_logits = tf.concat([self.decoder_train_logits, zero_padding], axis=1)

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
