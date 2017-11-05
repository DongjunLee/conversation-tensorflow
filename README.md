# A Neural Conversational Model

TensorFLow Sequence-to-Sequence Models for Conversation

## Requirements

- Python 3.6
- TensorFlow 1.4
- hb-config

## Features

- Using Higher-APIs in TensorFlow
	- [Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
	- [Experiment](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment)
	- [Dataset](https://www.tensorflow.org/api_docs/python/tf/contrib/data/Dataset)


## Todo

- apply Beam Search (tensorflow error..)
- need to preprocessing data.
- train with other dataset.
- make dataset Korean dialog corpus like [Cornell_Movie-Dialogs_Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

## Config

example: cornell-movie-dialogs.yml

```yml
data:
  base_path: 'data/cornell_movie_dialogs_corpus/'
  conversation_fname: 'movie_conversations.txt'
  line_fname: 'movie_lines.txt'
  processed_path: 'processed_cornell_movie_dialogs_data'
  word_threshold: 2
  max_seq_length: 200
  testset_size: 25000

  PAD_ID: 0
  UNK_ID: 1
  START_ID: 2
  EOS_ID: 3

model:
  num_layers: 3
  num_units: 512
  embed_dim: 256
  embed_share: true (true or false)
  cell_type: GRU  (LSTM, GRU, LAYER_NORM_LSTM, NAS)
  beam_width: 0  (0: GreedyEmbeddingHelper)
  dropout: 0.2

train:
  batch_size: 32
  learning_rate: 0.001
  train_steps: 100000
  model_dir: 'logs/cornell_movie_dialogs'
  save_every: 1000
  loss_hook_n_iter: 1000
  check_hook_n_iter: 1000
  min_eval_frequency: 1000
```


## Usage

Install requirements.

```pip install -r requirements.txt```

First, check if the model is valid.

```python main.py --config check_tiny --mode train```

Then, download [Cornell_Movie-Dialogs_Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) and train it.

```
sh prepare_Cornell_Movie-Dialogs_Corpus
python main.py --config cornell-movie-dialogs --mode train_and_evaluate
```

After training, start chatting.

```python chat --config cornell-movie-dialogs```



## Conversation Example

\> (Human)

```
> hi.
i ' m sorry . <\s>
> why?
i ' m not interested .
> hmm.. are you human?
nnnnn . <\s>
> you are you?
i ' m not going to marry you . <\s>
> what? me also!
i ' m sorry . <\s>
> anyway, what is your name?
i ' m not going to kill you . <\s>
> i'm asking your name.
i ' m sorry . <\s>
> yeah, so.. where are you from?
i ' m not sure . <\s>
> Have you lost your memory?
i ' m not sure i ' m not going to you . <\s>
> Have you ever been to Florida?
i ' m not sure i ' m not a little . <\s>
> there there.
no , no , no , no ,
> Are you alright?
i ' m not going to die . <\s>
> alsoulutely, you are very healthy.
i don ' t know . <\s>
> you know that? you are robot.
i ' m not sure . <\s>
>  i don ' t know what to say.
i ' m not sure i ' m not a little stupid . <\s>
> what do you like to do in your free time?
i ' m not sure i ' m not gonna take you a little more . <\s>
> are you in love?
i ' m not sure i ' m not stupid . <\s>
> i know.
i ' m not going to let you
> what's your name?
i ' m not sure i ' m not a good guy . <\s>
> oh god..
i ' m not going to do that . <\s>
> let's talk about something else.
i ' m not sure . <\s>
> i have to figure out what is the problem.
i ' m not sure . <\s>
> because of you are stupid.
i ' m not going to get you . <\s>
> florida?
he is a homosexual - -
> really?
i ' m not sure i
> can you tell me about more detail about him?
i ' m not a cop . <\s>
> okay.
i ' m not going to
> you always say 'i....'
i ' m not sure i ' m not a murderer . <\s>
> hmm.....
i ' m not sure i ' m not a little popular . <\s>
> you are not popular.
i ' m not sure i ' m not a girl .
> are you boy?
i ' m not sure . <\s>
> are you girl?
i ' m not sure . <\s>
> who are you?
i ' m sorry . <\s>
> i'm sorry.
i ' m not crowding . <\s>
> just kidding.
i ' m going to be a writer
> great!
i don ' t know .
> what do you mean?
i ' m sorry , i ' m sorry . <\s>
> omg
no . <\s>
> florida.
he hasn ' t got any
> music is my life.
you ' re a fool , you ' re a man !
> wow!
i ' m not kidding .
> sorry.
i ' m not kidding .
> yeah, i know it.
i ' m not sure . <\s>
> california?
what ? <\s>
> haha.
it ' s a real competition
```

Currently, my model is very stupid. Most of the words begin with ‘ I'm ... ’. 


## Reference

- [stanford-tensorflow-tutorials](https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/assignments/chatbot) by Chip Huyen
- [TensorFlow — Sequence to Sequence](https://medium.com/@ilblackdragon/tensorflow-sequence-to-sequence-3d9d2e238084) - Medium Illia Polosukhin
- [TensorFlow Neural Machine Translation Tutorial](https://github.com/tensorflow/nmt) - Tensorflow
- [tf-seq2seq](https://github.com/JayParks/tf-seq2seq) by JayParks