
from hbconfig import Config
import tensorflow as tf

from .encoder import Encoder
from .decoder import Decoder


class Graph:

    def __init__(self,
            encoder_inputs=None,
            decoder_inputs=None,
            dtype=tf.float32):

        self.encoder_inputs = encoder_inputs
        self.encoder_input_lengths = tf.reduce_sum(
            tf.to_int32(tf.not_equal(self.encoder_inputs, Config.data.PAD_ID)), 1,
            name="encoder_input_lengths")

        self.decoder_inputs = decoder_inputs
        self.decoder_input_lengths = tf.reduce_sum(
            tf.to_int32(tf.not_equal(self.decoder_inputs, Config.data.PAD_ID)), 1,
            name="decoder_input_lengths")

        self.dtype = dtype

    def build(self, mode):
        self.beam_width = Config.predict.get('beam_width', 0)
        self.mode = mode

        self._build_embed()
        self._build_encoder()
        self._build_decoder()

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
                self.embedding_encoder, self.encoder_inputs)

            if self.mode != tf.estimator.ModeKeys.PREDICT:
                self.decoder_emb_inp = tf.nn.embedding_lookup(
                    self.embedding_decoder, self.decoder_inputs)

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
                    self.predictions = decoder_outputs.predicted_ids
                else:
                    self.predictions = decoder_outputs.sample_id
            else:
                self.decoder_train_logits = decoder_outputs.rnn_output

        self.train_predictions = tf.argmax(self.decoder_train_logits, axis=2)
        tf.identity(self.train_predictions[0], name='train/pred_0')

        # make logits
        pad_num = Config.data.max_seq_length - tf.shape(self.decoder_train_logits)[1]
        zero_padding = tf.zeros(
                [Config.train.batch_size, pad_num, Config.data.vocab_size],
                dtype=self.dtype)

        self.logits = tf.concat([self.decoder_train_logits, zero_padding], axis=1)
        self.weight_masks = tf.sequence_mask(
            lengths=self.decoder_input_lengths,
            maxlen=Config.data.max_seq_length,
            dtype=self.dtype, name='masks')
