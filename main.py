#-- coding: utf-8 -*-

import argparse
import logging

from hbconfig import Config
import tensorflow as tf

import data_loader
import dataset
from model import Conversation
import hook



def experiment_fn(run_config, params):

    conversation = Conversation()
    estimator = tf.estimator.Estimator(
            model_fn=conversation.model_fn,
            model_dir=Config.train.model_dir,
            params=params,
            config=run_config)

    vocab = data_loader.load_vocab("vocab")
    Config.data.vocab_size = len(vocab)

    train_X, test_X, train_y, test_y = data_loader.make_train_and_test_set()
    Config.eval.batch_size = len(test_y)

    train_input_fn, train_input_hook = dataset.get_train_inputs(train_X, train_y)
    test_input_fn, test_input_hook = dataset.get_test_inputs(test_X, test_y)

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        train_steps=Config.train.train_steps,
        min_eval_frequency=Config.train.min_eval_frequency,
        train_monitors=[
            train_input_hook,
            hook.print_variables(
                variables=['train/enc_0', 'train/dec_0', 'train/pred_0'],
                vocab=vocab,
                every_n_iter=Config.train.check_hook_n_iter)],
        eval_hooks=[test_input_hook]
    )
    return experiment


def main(mode):
    params = tf.contrib.training.HParams(**Config.model.to_dict())

    run_config = tf.contrib.learn.RunConfig(
            model_dir=Config.train.model_dir,
            save_checkpoints_steps=Config.train.save_checkpoints_steps)

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=mode,
        hparams=params
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode (train/test/train_and_evaluate)')
    args = parser.parse_args()

    tf.logging._logger.setLevel(logging.INFO)

    Config(args.config)
    print("Config: ", Config)

    main(args.mode)
