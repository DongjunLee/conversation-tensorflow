from __future__ import print_function


from hbconfig import Config
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.layers import core as layers_core

import seq2seq_attention



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
                predictions={"prediction": self.predictions})
        else:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=self.train_predictions,
                loss=self.loss,
                train_op=self.train_op
            )

    def _init_placeholder(self, features, labels):
        self.encoder_inputs = features
        if type(features) == dict:
            self.encoder_inputs = features["input_data"]

        if self.mode != tf.estimator.ModeKeys.PREDICT:

            self.decoder_inputs = labels
            decoder_input_shift_1 = tf.slice(self.decoder_inputs, [0, 1],
                    [Config.train.batch_size, Config.data.max_seq_length-1])
            pad_tokens = tf.zeros([Config.train.batch_size, 1], dtype=tf.int32)

            # make target (right shift 1 from decoder_inputs)
            self.targets = tf.concat([decoder_input_shift_1, pad_tokens], axis=1)

    def build_graph(self):
        graph = seq2seq_attention.Graph(
                    encoder_inputs=self.encoder_inputs,
                    decoder_inputs=self.decoder_inputs)
        graph.build(self.mode)

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.predictions = graph.predictions
        else:
            self.train_predictions = graph.train_predictions
            self._build_loss(graph.logits, graph.weight_masks)
            self._build_optimizer()

    def _build_loss(self, logits, weight_masks):
        self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
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
