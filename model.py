from __future__ import print_function


from hbconfig import Config
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.layers import core as layers_core



class Seq2Seq:

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
            if Config.model.encoder_type == "uni":

                cells = self._build_rnn_cells()
                self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                        cells,
                        self.encoder_emb_inp,
                        sequence_length=self.encoder_input_lengths,
                        dtype=tf.float32,
                        time_major=False)

            elif Config.model.encoder_type == "bi":
                cells_fw = self._build_rnn_cells(Config.model.num_units,is_list=True)
                cells_bw = self._build_rnn_cells(Config.model.num_units,is_list=True)
                
                outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                        cells_fw,
                        cells_bw,
                        self.encoder_emb_inp,
                        initial_states_fw=None,
                        initial_states_bw=None,
                        sequence_length=self.encoder_input_lengths,
                        dtype=tf.float32)

                if Config.model.cell_type == "LSTM":
                    encoder_final_state_c = tf.concat((output_state_fw[-1].c,output_state_bw[-1].c),1)
                    encoder_final_state_h = tf.concat((output_state_fw[-1].h,output_state_bw[-1].h),1)
                    encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c = encoder_final_state_c, h = encoder_final_state_h)
                else:
                    encoder_final_state = tf.concat((output_state_fw[-1],output_state_bw[-1]),1)

                self.encoder_outputs = outputs
                self.encoder_final_state = encoder_final_state
            

            beam_width = Config.predict.get('beam_width', 0)
            if self.mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0 :
                self.encoder_outputs = tf.contrib.seq2seq.tile_batch(self.encoder_outputs, beam_width)
                self.encoder_input_lengths = tf.contrib.seq2seq.tile_batch(self.encoder_input_lengths, beam_width)

    def _build_projection(self):
        # Projection
        with tf.variable_scope("decoder/output_projection"):
            self.output_layer = layers_core.Dense(
                Config.data.vocab_size, use_bias=False, name="output_projection")

    def _build_decoder(self):

        def decode(helper=None, scope="decode"):

            with tf.variable_scope(scope):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=Config.model.num_units, memory=self.encoder_outputs,
                    memory_sequence_length=self.encoder_input_lengths)

                if Config.model.encoder_type == "uni":
                    cells = self._build_rnn_cells(Config.model.num_units)
                    attention_layer_size = Config.model.num_units
                elif Config.model.encoder_type == "bi":
                    cells = self._build_rnn_cells(Config.model.num_units*2)
                    attention_layer_size = Config.model.num_units*2

                beam_width = Config.predict.get('beam_width', 0)
                alignment_history = (self.mode == tf.estimator.ModeKeys.PREDICT and
                         beam_width == 0)

                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cells,
                    attention_mechanism,
                    attention_layer_size=attention_layer_size,
                    alignment_history=alignment_history,
                    name="attention")

                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, Config.data.vocab_size)

                if self.mode == tf.estimator.ModeKeys.PREDICT:

                    maximum_iterations = tf.round(tf.reduce_max(self.encoder_input_lengths) * 2)

                    if helper is None:
                        decoder_start_state = tf.contrib.seq2seq.tile_batch(self.encoder_final_state, beam_width)
                        decoder_initial_state = out_cell.zero_state(Config.train.batch_size * beam_width, self.dtype)
                        decoder_initial_state.clone(cell_state=decoder_start_state)

                        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                            cell=out_cell,
                            embedding=self.embedding_decoder,
                            start_tokens=tf.fill([Config.train.batch_size], Config.data.START_ID),
                            end_token=Config.data.EOS_ID,
                            initial_state=(decoder_initial_state),
                            beam_width=Config.predict.beam_width,
                            length_penalty_weight=Config.predict.length_penalty_weight)

                        outputs = tf.contrib.seq2seq.dynamic_decode(
                            decoder=decoder,
                            output_time_major=False,
                            impute_finished=False,
                            maximum_iterations=maximum_iterations)

                    else:
                        decoder_initial_state = out_cell.zero_state(Config.train.batch_size, self.dtype)
                        decoder_initial_state.clone(cell_state=self.encoder_final_state)

                        decoder = tf.contrib.seq2seq.BasicDecoder(
                            cell=out_cell,
                            helper=helper,
                            initial_state=(decoder_initial_state))

                        outputs = tf.contrib.seq2seq.dynamic_decode(
                            decoder=decoder,
                            output_time_major=False,
                            impute_finished=True,
                            maximum_iterations=maximum_iterations)

                else:
                    decoder_initial_state = out_cell.zero_state(Config.train.batch_size, self.dtype)
                    decoder_initial_state.clone(cell_state=self.encoder_final_state)
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=out_cell,
                        helper=helper,
                        initial_state=(decoder_initial_state))

                    outputs = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder,
                        output_time_major=False,
                        swap_memory=True)

                return outputs[0]

        with tf.variable_scope('decoder'):

            if self.mode == tf.estimator.ModeKeys.PREDICT:

                beam_width = Config.predict.get('beam_width', 0)
                if beam_width > 0 :
                    self.decoder_pred_outputs = decode()
                    self.prediction = self.decoder_pred_outputs.predicted_ids
                else:
                    self.pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        embedding=self.embedding_decoder,
                        start_tokens=tf.fill([Config.train.batch_size], Config.data.START_ID),
                        end_token=Config.data.EOS_ID)

                    self.decoder_pred_outputs = decode(helper=self.pred_helper)
                    self.prediction = self.decoder_pred_outputs.sample_id

            else:
                self.train_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self.decoder_emb_inp,
                    sequence_length=self.decoder_input_lengths)

                self.decoder_train_outputs = decode(self.train_helper, 'decode')
                self.decoder_train_logits = self.decoder_train_outputs.rnn_output

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.decoder_train_pred = tf.argmax(self.decoder_train_logits[0], axis=1, name='train/pred_0')

    def _build_rnn_cells(self, num_units, is_list=False):
        stacked_rnn = []
        for _ in range(Config.model.num_layers):
            single_cell = self._single_cell(Config.model.cell_type, Config.model.dropout,num_units)
            stacked_rnn.append(single_cell)

        if(is_list):
            return stacked_rnn
        else:
            return tf.nn.rnn_cell.MultiRNNCell(
                    cells=stacked_rnn,
                    state_is_tuple=True)
        

    def _single_cell(self, cell_type, dropout, num_units):
        if cell_type == "GRU":
            single_cell = tf.contrib.rnn.GRUCell(
                num_units)
        elif cell_type == "LSTM":
            single_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units,
                forget_bias=1.0)
        elif cell_type == "LAYER_NORM_LSTM":
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units,
                forget_bias=1.0,
                layer_norm=True)
        elif cell_type == "NAS":
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units)
        else:
            raise ValueError(f"Unknown rnn cell type. {cell_type}")

        if dropout > 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - dropout))

        return single_cell

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
