import logging
import os

from hbconfig import Config
import tensorflow as tf


class Logger(object):
    class __Logger:
        def __init__(self):
            self._logger = logging.getLogger("crumbs")
            self._logger.setLevel(logging.INFO)

            formatter = logging.Formatter(
                '[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(formatter)

            self._logger.addHandler(streamHandler)

    instance = None

    def __init__(self):
        if not Logger.instance:
            Logger.instance = Logger.__Logger()

    def get_logger(self):
        return self.instance._logger


def tokenize_and_map(line, vocab):
    return [vocab.get(token, Config.data.UNK_ID) for token in line.split(' ')]


def make_input_fn(
        batch_size, input_filename, output_filename, vocab,
        max_sentence_length,
        input_process=tokenize_and_map, output_process=tokenize_and_map):

    def input_fn():
        inp = tf.placeholder(tf.int64, shape=(None, None), name='input')
        output = tf.placeholder(tf.int64, shape=(None, None), name='output')
        tf.identity(inp[0], 'input_0')
        tf.identity(output[0], 'output_0')
        return {
            'input': inp,
            'output': output,
        }, None

    def sampler():
        input_path = os.path.join(Config.data.PROCESSED_PATH, input_filename)
        output_path = os.path.join(Config.data.PROCESSED_PATH, output_filename)

        while True:
            with open(input_path) as finput:
                with open(output_path) as foutput:
                    for in_line in finput:
                        out_line = foutput.readline()

                        if in_line.startswith("\n") or out_line.startswith("\n"):
                            pass
                        else:
                            in_lines = in_line.split(' ')
                            out_lines = out_line.split(' ')
                            if len(in_lines) < Config.data.MAX_SENTENCE_LENGTH and \
                                len(out_lines) + 2 < Config.data.MAX_SENTENCE_LENGTH:

                                yield {
                                    'input': in_lines,
                                    'output': out_lines
                                }

    sample_me = sampler()

    def feed_fn():
        inputs, outputs = [], []

        # Pad me right with pad_id
        for i in range(batch_size):
            rec = next(sample_me)
            inputs.append(rec['input'])
            outputs.append(rec['output'])

            inputs[i] += [Config.data.PAD_ID] * (max_sentence_length - len(inputs[i]))
            outputs[i] = [Config.data.START_ID] + outputs[i] + [Config.data.EOS_ID] + ([Config.data.PAD_ID] * (max_sentence_length - 2 - len(outputs[i])))
        return {
            'input:0': inputs,
            'output:0': outputs
        }

    return input_fn, feed_fn


def load_vocab(filename):
    vocab = {}
    vocab_path = os.path.join(Config.data.PROCESSED_PATH, filename)
    with open(vocab_path) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    return vocab


def get_rev_vocab(vocab):
    return {idx: key for key, idx in vocab.items()}


def get_formatter(keys, vocab):
    rev_vocab = get_rev_vocab(vocab)

    def to_str(sequence):
        tokens = [
            rev_vocab.get(x, "<unk>") for x in sequence if x != Config.data.PAD_ID]
        return ' '.join(tokens)

    def format(values):
        res = []
        for key in keys:
            res.append("%s = %s" % (key, to_str(values[key])))
        return '\n - ' + '\n - '.join(res)
    return format
