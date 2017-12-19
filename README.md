# A Neural Conversational Model

TensorFlow implementation of Conversation Models.

1. **Model**

	- `seq2seq_attention` : Seq2Seq model with attentional decoder based on '[Neural machine translation by jointly learning to align and translate](https://arxiv.org/pdf/1409.0473.pdf)' (2015) by Dzmitry Bahdanau
		- Encoder
			- Unidirectional RNN
			- Stack Bidirectional RNN
		- Attention
			- Bahdanau Attention (option Norm)
			- Luong Attention (option Scale)
		- Decoder
			- Greedy (beam_width = 0)
			- Beam Search (beam_width > 0)

2. **Dataset**

	- [Cornell_Movie-Dialogs_Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
	- [Twitter chat_corpus](https://github.com/Marsan-Ma/chat_corpus)

## Requirements

- Python 3.6
- TensorFlow 1.4
- nltk
- hb-config
- tqdm

## Project Structure

    .
    ├── config                  # Config files (.yml, .json) using with hb-config
    ├── seq2seq_attention       # seq2seq_attention architecture graphs (from input to logits)
    ├── data_loader.py          # raw_date -> precossed_data -> generate_batch (using Dataset)
    ├── hook.py                 # training or test hook feature (eg. print_variables)
    ├── main.py                 # define experiment_fn
    └── model.py                # define EstimatorSpec      

Reference : [hb-config](https://github.com/hb-research/hb-config), [Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator), [experiments_fn](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment), [EstimatorSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec)


## Todo

- need to preprocessing data.
- train with other dataset. ([Twitter chat_corpus](https://github.com/Marsan-Ma/chat_corpus))
- make dataset Korean dialog corpus like [Cornell_Movie-Dialogs_Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- Implements CopyNet
	- [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393) by J, Gu 2016.
- Common Challenge
	- Incorporating Context
		- [Building End-To-End Dialogue Systems
Using Generative Hierarchical Neural Network Models](https://arxiv.org/pdf/1507.04808.pdf) by IV Serban, 2015.
	- Coherent Personality Challenge
		- [A Persona-Based Neural Conversation Model](https://arxiv.org/abs/1603.06155) by J Li, 2015.
	- Intention and Diversity
		- [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/abs/1510.03055) by J Li, 2015.
	- Evalutaion Metrics
		- [How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation](https://arxiv.org/abs/1603.08023) by CW Liu, 2016

## Config

Can control all **Experimental environment**.

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
  num_layers: 4
  num_units: 512
  embed_dim: 256
  embed_share: true (true or false)
  cell_type: gru  (lstm, gru, layer_norm_lstm, nas)
  beam_width: 0  (0: GreedyEmbeddingHelper)
  dropout: 0.2
  encoder_type: bi  (uni / bi)
  attention_mechanism: normed_bahdanau (bahdanau, normed_bahdanau, luong, scaled_luong)

train:
  batch_size: 32
  learning_rate: 0.001
  train_steps: 100000
  model_dir: 'logs/cornell_movie_dialogs'
  save_every: 1000
  loss_hook_n_iter: 1000
  check_hook_n_iter: 1000
  min_eval_frequency: 1000

eval:
  batch_size: -1 (all test dataset)

predict:
  beam_width: 5  (0: GreedyEmbeddingHelper, 1>=: BeamSearchDecoder)
  length_penalty_weight: 1.0
```


## Usage

Install requirements.

```pip install -r requirements.txt```

First, check if the model is valid.

```python main.py --config check_tiny --mode train```

Then, download [Cornell_Movie-Dialogs_Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) and train it.

```
sh scripts/prepare_Cornell_Movie-Dialogs_Corpus
python main.py --config cornell-movie-dialogs --mode train_and_evaluate
```

### Experiments modes

- `evaluate` : Evaluate on the evaluation data.
- `extend_train_hooks` : Extends the hooks for training.
- `reset_export_strategies` : Resets the export strategies with the new_export_strategies.
- `run_std_server` : Starts a TensorFlow server and joins the serving thread.
- `test` : Tests training, evaluating and exporting the estimator for a single step.
- `train` : Fit the estimator using the training data.
- `train_and_evaluate` : Interleaves training and evaluation.

---

After training, start chatting.

```python chat.py --config cornell-movie-dialogs```


### Tensorboard

```tensorboard --logdir logs```


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


## Reference

- [stanford-tensorflow-tutorials](https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/assignments/chatbot) by Chip Huyen
- [TensorFlow Neural Machine Translation Tutorial](https://github.com/tensorflow/nmt) - Tensorflow
- [Deep Learning for Chatbots, Part 1 – Introduction](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/)
- [Neural Text Generation: A Practical Guide](https://arxiv.org/abs/1711.09534) (2017) by Ziang Xie

## Author

Dongjun Lee (humanbrain.djlee@gmail.com)

### Contributors

- junbeomlee ([github](https://github.com/junbeomlee))
