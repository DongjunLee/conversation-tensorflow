# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os
import random
import re

from nltk.tokenize import TweetTokenizer
from hbconfig import Config
import numpy as np
import tensorflow as tf
from tqdm import tqdm



tokenizer = TweetTokenizer()

def get_lines():
    id2line = {}
    file_path = os.path.join(Config.data.base_path, Config.data.line_fname)
    with open(file_path, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.decode('iso-8859-1').split(' +++$+++ ')
            if len(parts) == 5:
                if parts[4][-1] == '\n':
                    parts[4] = parts[4][:-1]
                id2line[parts[0]] = parts[4]
    return id2line


def get_convos():
    """ Get conversations from the raw data """
    file_path = os.path.join(Config.data.base_path, Config.data.conversation_fname)
    convos = []
    with open(file_path, 'rb') as f:
        for line in f.readlines():
            parts = line.decode('iso-8859-1').split(' +++$+++ ')
            if len(parts) == 4:
                convo = []
                for line in parts[3][1:-2].split(', '):
                    convo.append(line[1:-1])
                convos.append(convo)

    return convos


def cornell_question_answers(id2line, convos):
    """ Divide the dataset into two sets: questions and answers. """
    questions, answers = [], []
    for convo in convos:
        for index, line in enumerate(convo[:-1]):
            questions.append(id2line[convo[index]])
            answers.append(id2line[convo[index + 1]])
    assert len(questions) == len(answers)
    return questions, answers


def twitter_question_answers():
    """ Divide the dataset into two sets: questions and answers. """
    file_path = os.path.join(Config.data.base_path, Config.data.line_fname)

    twitter_corpus = []
    with open(file_path, 'rb') as f:
        for line in f.readlines():
            line = line.decode('utf-8')

            if line[-1] == '\n':
                twitter_corpus.append(line[:-1].lower())
            else:
                twitter_corpus.append(line.lower())

    questions = twitter_corpus[::2] # even is question
    answers = twitter_corpus[1::2] # odd is answer

    assert len(questions) == len(answers)
    return questions, answers


def prepare_dataset(questions, answers):
    # create path to store all the train & test encoder & decoder
    make_dir(Config.data.base_path + Config.data.processed_path)

    # random convos to create the test set
    test_ids = random.sample([i for i in range(len(questions))], Config.data.testset_size)

    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    files = []
    for filename in filenames:
        files.append(open(os.path.join(Config.data.base_path, Config.data.processed_path, filename), 'wb'))

    for i in tqdm(range(len(questions))):

        question = questions[i]
        answer = answers[i]

        if i in test_ids:
            files[2].write((question + "\n").encode('utf-8').lower())
            files[3].write((answer + '\n').encode('utf-8').lower())
        else:
            files[0].write((question + '\n').encode('utf-8').lower())
            files[1].write((answer + '\n').encode('utf-8').lower())

    for file in files:
        file.close()


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words


def build_vocab(in_fname, out_fname, normalize_digits=True):
    print("Count each vocab frequency ...")

    vocab = {}
    def count_vocab(fname):
        with open(fname, 'rb') as f:
            for line in tqdm(f.readlines()):
                line = line.decode('utf-8')
                for token in tokenizer.tokenize(line):
                    if not token in vocab:
                        vocab[token] = 0
                    vocab[token] += 1

    in_path = os.path.join(Config.data.base_path, Config.data.processed_path, in_fname)
    out_path = os.path.join(Config.data.base_path, Config.data.processed_path, out_fname)

    count_vocab(in_path)
    count_vocab(out_path)

    print("total vocab size:", len(vocab))
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)

    dest_path = os.path.join(Config.data.base_path, Config.data.processed_path, 'vocab')
    with open(dest_path, 'wb') as f:
        f.write(('<pad>' + '\n').encode('utf-8'))
        f.write(('<unk>' + '\n').encode('utf-8'))
        f.write(('<s>' + '\n').encode('utf-8'))
        f.write(('<\s>' + '\n').encode('utf-8'))
        index = 4
        for word in tqdm(sorted_vocab):
            if vocab[word] < Config.data.word_threshold:
                break

            f.write((word + '\n').encode('utf-8'))
            index += 1


def load_vocab(vocab_fname):
    print("load vocab ...")
    with open(os.path.join(Config.data.base_path, Config.data.processed_path, vocab_fname), 'rb') as f:
        words = f.read().decode('utf-8').splitlines()
        print("vocab size:", len(words))
    return {words[i]: i for i in range(len(words))}


def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in tokenizer.tokenize(line)]


def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab'
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    vocab = load_vocab(vocab_path)
    in_file = open(os.path.join(Config.data.base_path, Config.data.processed_path, in_path), 'rb')
    out_file = open(os.path.join(Config.data.base_path, Config.data.processed_path, out_path), 'wb')

    lines = in_file.read().decode('utf-8').splitlines()
    for line in tqdm(lines):
        if mode == 'dec':  # we only care about '<s>' and </s> in decoder
            ids = [vocab['<s>']]
        else:
            ids = []

        sentence_ids = sentence2id(vocab, line)
        ids.extend(sentence_ids)
        if mode == 'dec':
            ids.append(vocab['<\s>'])

        out_file.write(b' '.join(str(id_).encode('cp1252') for id_ in ids) + b'\n')


def prepare_raw_data():
    print('Preparing raw data into train set and test set ...')

    data_type = Config.data.get('type', 'cornell-movie')
    if data_type == "cornell-movie":
        id2line = get_lines()
        convos = get_convos()
        questions, answers = cornell_question_answers(id2line, convos)
    elif data_type == "twitter":
        questions, answers = twitter_question_answers()
    elif data_type == "all":
        # cornell-movie
        Config.data.base_path = "data/cornell_movie_dialogs_corpus/"
        Config.data.line_fname = "movie_lines.txt"
        Config.data.conversation_fname = "movie_conversations.txt"

        id2line = get_lines()
        convos = get_convos()
        co_questions, co_answers = cornell_question_answers(id2line, convos)

        #twitter
        Config.data.base_path = "data/"
        Config.data.line_fname = "twitter_en.txt"

        tw_questions, tw_answers = twitter_question_answers()

        questions = co_questions + tw_questions
        answers = co_answers + tw_answers
    else:
        raise ValueError(f"Unknown data_type, {data_type}")

    prepare_dataset(questions, answers)

def process_data():
    print('Preparing data to be model-ready ...')

    build_vocab('train.enc', 'train.dec')

    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')


def make_train_and_test_set(shuffle=True, bucket=True):
    print("make Training data and Test data Start....")

    train_X, train_y = load_data('train_ids.enc', 'train_ids.dec')
    test_X, test_y = load_data('test_ids.enc', 'test_ids.dec')

    assert len(train_X) == len(train_y)
    assert len(test_X) == len(test_y)

    print(f"train data count : {len(train_X)}")
    print(f"test data count : {len(test_X)}")

    if shuffle:
        print("shuffle dataset ...")
        train_p = np.random.permutation(len(train_y))
        test_p = np.random.permutation(len(test_y))

        train_X, train_y = train_X[train_p], train_y[train_p]
        test_X, test_y = test_X[test_p], test_y[test_p]

    if bucket:
        print("sorted by inputs length and outputs length ...")
        train_X, train_y = zip(*sorted(zip(train_X, train_y), key=lambda x: len(x[0]) + len([x[1]])))
        test_X, test_y = zip(*sorted(zip(test_X, test_y), key=lambda x: len(x[0]) + len([x[1]])))

    return train_X, test_X, train_y, test_y

def load_data(enc_fname, dec_fname):
    enc_input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, enc_fname), 'r')
    dec_input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, dec_fname), 'r')

    enc_data, dec_data = [], []
    for e_line, d_line in tqdm(zip(enc_input_data.readlines(), dec_input_data.readlines())):
        e_ids = [int(id_) for id_ in e_line.split()]
        d_ids = [int(id_) for id_ in d_line.split()]

        if len(e_ids) == 0 or len(d_ids) == 0:
            continue

        if len(e_ids) <= Config.data.max_seq_length and len(d_ids) < Config.data.max_seq_length:

            if abs(len(d_ids) - len(e_ids)) / (len(e_ids) + len(d_ids)) < Config.data.sentence_diff:
                enc_data.append(_pad_input(e_ids, Config.data.max_seq_length))
                dec_data.append(_pad_input(d_ids, Config.data.max_seq_length))

    print(f"load data from {enc_fname}, {dec_fname}...")
    return np.array(enc_data, dtype=np.int32), np.array(dec_data, dtype=np.int32)


def _pad_input(input_, size):
    return input_ + [Config.data.PAD_ID] * (size - len(input_))


def set_max_seq_length(dataset_fnames):

    max_seq_length = Config.data.get('max_seq_length', 10)

    for fname in dataset_fnames:
        input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, fname), 'r')

        for line in input_data.readlines():
            ids = [int(id_) for id_ in line.split()]
            seq_length = len(ids)

            if seq_length > max_seq_length:
                max_seq_length = seq_length

    Config.data.max_seq_length = max_seq_length
    print(f"Setting max_seq_length to Config : {max_seq_length}")


def make_batch(data, buffer_size=10000, batch_size=64, scope="train"):

    class IteratorInitializerHook(tf.train.SessionRunHook):
        """Hook to initialise data iterator after Session is created."""

        def __init__(self):
            super(IteratorInitializerHook, self).__init__()
            self.iterator_initializer_func = None

        def after_create_session(self, session, coord):
            """Initialise the iterator after the session has been created."""
            self.iterator_initializer_func(session)


    def get_inputs():

        iterator_initializer_hook = IteratorInitializerHook()

        def train_inputs():
            with tf.name_scope(scope):

                X, y = data

                # Define placeholders
                input_placeholder = tf.placeholder(
                    tf.int32, [None, Config.data.max_seq_length])
                output_placeholder = tf.placeholder(
                    tf.int32, [None, Config.data.max_seq_length])

                # Build dataset iterator
                dataset = tf.data.Dataset.from_tensor_slices(
                    (input_placeholder, output_placeholder))

                if scope == "train":
                    dataset = dataset.repeat(None)  # Infinite iterations
                else:
                    dataset = dataset.repeat(1)  # 1 Epoch
                # dataset = dataset.shuffle(buffer_size=buffer_size)
                dataset = dataset.batch(batch_size)

                iterator = dataset.make_initializable_iterator()
                next_X, next_y = iterator.get_next()

                tf.identity(next_X[0], 'enc_0')
                tf.identity(next_y[0], 'dec_0')

                # Set runhook to initialize iterator
                iterator_initializer_hook.iterator_initializer_func = \
                    lambda sess: sess.run(
                        iterator.initializer,
                        feed_dict={input_placeholder: X,
                                   output_placeholder: y})

                # Return batched (features, labels)
                return next_X, next_y

        # Return function and hook
        return train_inputs, iterator_initializer_hook

    return get_inputs()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    prepare_raw_data()
    process_data()
