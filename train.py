from __future__ import print_function

import logging
import os

from hbconfig import Config
import tensorflow as tf

import data
from model import Seq2Seq
import utils


def train():
    vocab = utils.load_vocab("vocab")
    Config.model.VOCAB_SIZE = len(vocab)

    seq2seq = Seq2Seq()
    est = tf.estimator.Estimator(
            model_fn=seq2seq.make_estimator_spec,
            model_dir=Config.model.CPT_PATH)

    input_fn, feed_fn = utils.make_input_fn(
        Config.model.BATCH_SIZE,
        "input", "output",
        vocab, Config.model.MAX_SENTENCE_LENGTH)

    # Make hooks to print examples of inputs/predictions.
    print_inputs = tf.train.LoggingTensorHook(
        ['input_0', 'output_0'], every_n_iter=1,
        formatter=utils.get_formatter(['input_0', 'output_0'], vocab))

    print_predictions = tf.train.LoggingTensorHook(
        ['predictions', 'decoder/train_pred'], every_n_iter=1,
        formatter=utils.get_formatter(['predictions', 'decoder/train_pred'], vocab))

    est.train(
        input_fn=input_fn,
        hooks=[tf.train.FeedFnHook(feed_fn), print_inputs, print_predictions],
        steps=10000)


def main():
    if not os.path.isdir(Config.data.PROCESSED_PATH):
        data.prepare_raw_data()
        data.process_data()
    print('Data ready!')
    # create checkpoints folder if there isn't one already
    data.make_dir(Config.model.CPT_PATH)

    train()

if __name__ == '__main__':
    tf.logging._logger.setLevel(logging.INFO)
    main()
