from __future__ import print_function


from hbconfig import Config
import nltk
import tensorflow as tf

import seq2seq_attention



class Conversation:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32
        self.mode = mode

        self.loss, self.train_op, self.metrics, self.predictions = None, None, None, None
        self._init_placeholder(features, labels)
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self.metrics,
            predictions={"prediction": self.predictions})

    def _init_placeholder(self, features, labels):
        self.encoder_inputs = features
        if type(features) == dict:
            self.encoder_inputs = features["input_data"]

        batch_size = tf.shape(self.encoder_inputs)[0]

        if self.mode == tf.estimator.ModeKeys.TRAIN or self.mode == tf.estimator.ModeKeys.EVAL:
            self.decoder_inputs = labels
            decoder_input_shift_1 = tf.slice(self.decoder_inputs, [0, 1],
                    [batch_size, Config.data.max_seq_length-1])
            pad_tokens = tf.zeros([batch_size, 1], dtype=tf.int32)

            # make target (right shift 1 from decoder_inputs)
            self.targets = tf.concat([decoder_input_shift_1, pad_tokens], axis=1)
        else:
            self.decoder_inputs = None

    def build_graph(self):
        graph = seq2seq_attention.Graph(mode=self.mode)
        graph.build(encoder_inputs=self.encoder_inputs,
                    decoder_inputs=self.decoder_inputs)

        self.predictions = graph.predictions
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(graph.logits, graph.weight_masks)
            self._build_optimizer()
            self._build_metric()

    def _build_loss(self, logits, weight_masks):
        self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=self.targets,
                weights=weight_masks,
                name="loss")

    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer='Adam',
            learning_rate=Config.train.learning_rate,
            summaries=['loss', 'learning_rate'],
            name="train_op")

    def _build_metric(self):

        def blue_score(labels, predictions,
                       weights=None, metrics_collections=None,
                       updates_collections=None, name=None):

            def _nltk_blue_score(labels, predictions):

                # slice after <eos>
                predictions = predictions.tolist()
                for i in range(len(predictions)):
                    prediction = predictions[i]
                    if Config.data.EOS_ID in prediction:
                        predictions[i] = prediction[:prediction.index(Config.data.EOS_ID)+1]

                labels = [
                    [[w_id for w_id in label if w_id != Config.data.PAD_ID]]
                    for label in labels.tolist()]
                predictions = [
                    [w_id for w_id in prediction]
                    for prediction in predictions]

                return float(nltk.translate.bleu_score.corpus_bleu(labels, predictions))

            score = tf.py_func(_nltk_blue_score, (labels, predictions), tf.float64)
            return tf.metrics.mean(score * 100)

        self.metrics = {
            "accuracy": tf.metrics.accuracy(self.targets, self.predictions),
            "bleu": blue_score(self.targets, self.predictions)
        }
