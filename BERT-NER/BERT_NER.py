#! usr/bin/env python3
# -*- coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from bert import modeling
from bert import optimization
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from absl import logging

from my_utils import pad_and_cut, tag2label, read_vocab, data_augmentation
from my_utils import write_to_file, valid, get_sent_length_label, mask


flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_float("drop_prob", 0.1, "dropout rate")

flags.DEFINE_integer("lstm_layer", 1, "num of lstm layer.")

flags.DEFINE_bool("is_mask", False, "Whether to mask data.")

flags.DEFINE_integer("data_dup", 2, "num of data duplicate.")

flags.DEFINE_bool("data_augment", True, "Whether to do data augment.")

flags.DEFINE_bool(
    "use_lstm", False, "whether to use lstm")

flags.DEFINE_integer(
    "layer", -1, "layer to fine tune, fine-tune the layer which is in [FLAGS.layer, end], \
    layer=-10 means to fine-tune last 10 layers")

flags.DEFINE_integer(
    "hidden_dim", 768, "hidden_dim")

flags.DEFINE_string(
    "train_data", "./data/train.txt","train_data")

flags.DEFINE_string(
    "eval_data", "./data/new_valid.txt","eval_data")

flags.DEFINE_string(
    "test_data", "./data/test.txt","test_data")

flags.DEFINE_string(
    "output_dir", "./output/",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "result_file", "./result/result.txt",
    "The result file directory.")

flags.DEFINE_string(
    "eval_result_file", "./result/eval_result.txt",
    "The eval result file directory.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 200,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_float("num_train_epochs", 50.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("crf", 0, "crf/softmax")

flags.DEFINE_integer("train_batch_size", 48, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 48, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 48, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_string("gpu_id", "3", "gpu_id")

flags.DEFINE_string(
    "bert_config_file", "DG_BERT/dg_bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")


flags.DEFINE_string("vocab_file", "DG_BERT/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_id, length, label_id=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_id = text_id
    self.label_id = label_id
    self.length = length

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               mask,
               segment_ids,
               label_ids,
               is_real_example=True):
    self.input_ids = input_ids
    self.mask = mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.is_real_example = is_real_example

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, vocab):
        self.vocab = vocab

    def get_train_examples(self, data_dir, is_augment, is_mask):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_train_data(self, file_path, maxlen):
        """Read a BIO data!"""

        train_data = []

        logging.info("loading data: {}......".format(file_path))
        with open(file_path, mode='r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                sent = []
                label = []
                line = line.strip()
                line = line.split(' ')
                for piece in line:
                    # 过滤空的:''
                    if len(piece) == 0:
                        continue
                    assert len(piece) > 2
                    # str2int = [int(str) for str in piece[:-2].split('_')]
                    str2int = [str for str in piece[:-2].split('_')]
                    sent.extend(str2int)
                    try:
                        tag = tag2label[piece[-2:]]
                    except:
                        logging.info("tag not found in tag2id.", line)
                        break
                    if tag == 0:
                        label.extend([tag] * len(str2int))
                    else:
                        label.append(tag * 2 - 1)
                        if len(str2int) > 1:
                            label.extend([tag * 2] * (len(str2int) - 1))

                length = min(len(label), maxlen)
                # sent2id
                sent = self.sent2id(sent)

                # padding and cut
                sent = pad_and_cut(sent, pad_token=0, seq_len=maxlen)
                label = pad_and_cut(label, pad_token=0, seq_len=maxlen)
                train_data.append((sent, label, length))
        return train_data

    def _read_test_data(self, file_path, maxlen):
        test_data = []

        logging.info("loading data: {}......".format(file_path))
        with open(file_path, mode='r', encoding='utf-8') as fr:
            lines = fr.readlines()

            for line in lines:
                # line:str
                line = line.strip()
                # sent:list
                sent = line.split('_')

                # sent2id
                sent2id = self.sent2id(sent)
                length = min(len(sent2id), maxlen)
                # padding and cut
                sent2id = pad_and_cut(sent2id, pad_token=0, seq_len=maxlen)

                # 0代表label
                test_data.append((sent2id, [0], length))

        return test_data


    def sent2id(self, sent):
        sent2id = []
        for w in sent:
            if w in self.vocab:
                sent2id.append(self.vocab[w])
            else:
                sent2id.append(1)
        return sent2id

class NerProcessor(DataProcessor):
    def __init__(self, vocab, max_len):
        super(NerProcessor, self).__init__(vocab)
        self.max_len = max_len

    def get_train_data(self, data_dir, is_augment, data_dup):
        train_data = self._read_train_data(data_dir, self.max_len)

        if is_augment:
            logging.info("before data augment, num of data:{}".format(len(train_data)))
            train_data = data_augmentation_v2(train_data, self.max_len, data_dup)
            logging.info("After data augment, num of data:{}".format(len(train_data)))

        return train_data

    def get_train_examples(self, data_dir, is_augment, is_mask):
        train_data = self._read_train_data(data_dir, self.max_len)

        if is_mask:
            logging.info("++++++++++++++++++++++++++++mask5%++++++++++++++++++++++++++++++++")
            train_data = mask(train_data)

        if is_augment:
            logging.info("before data augment, num of data:{}".format(len(train_data)))
            train_data = data_augmentation(train_data, self.max_len, FLAGS.data_dup)
            logging.info("After data augment, num of data:{}".format(len(train_data)))

        return self._create_example(
            train_data, "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_train_data(data_dir, self.max_len), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_test_data(data_dir, self.max_len), "test"
        )

    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """
        return ["O", "B-A", "I-A", "B-B", "I-B", "B-C", "I-C"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_id=line[0], label_id=line[1], length=line[2]))
        return examples


def print_example(input_ids, mask, segment_ids, label_ids):
    logging.info("*** Example ***")
    logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
    logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))


def convert_examples_features(examples, output_file, maxlen):
    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        input_ids = example.text_id
        label_ids = example.label_id
        mask = example.length * [1]

        while len(label_ids)<maxlen:
            label_ids.append(0)

        while len(mask)<maxlen:
            mask.append(0)
        segment_ids = [0]*maxlen

        if ex_index<3:
            print_example(input_ids, mask, segment_ids, label_ids)
        features = collections.OrderedDict()

        features["input_ids"] = create_int_feature(input_ids)
        features["mask"] = create_int_feature(mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["label_ids"] = create_int_feature(label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        # sentence token in each batch
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }
    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn

# all above are related to data preprocess
# Following i about the model

def bi_lstm(sequence_output, mask, lstm_layer):
    logging.info("+++++++++++++++++++++++++++{} lstm layer++++++++++++++++++++++++++++++++".format(lstm_layer))
    mask2len = tf.reduce_sum(mask, axis=1)
    with tf.variable_scope("bi-lstm"):
        cells_fw = [LSTMCell(FLAGS.hidden_dim) for _ in range(lstm_layer)]
        cells_bw = [LSTMCell(FLAGS.hidden_dim) for _ in range(lstm_layer)]
        output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=sequence_output,
            sequence_length=mask2len,
            dtype=tf.float32)
        return output


def hidden2tag(hiddenlayer,numclass):
    linear = tf.keras.layers.Dense(numclass,activation=None)
    return linear(hiddenlayer)

def crf_loss(logits,labels,mask,num_labels,mask2len):
    """
    :param logits:
    :param labels:
    :param mask2len:each sample's length
    :return:
    """
    #TODO
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
                "transition",
                shape=[num_labels,num_labels],
                initializer=tf.contrib.layers.xavier_initializer()
        )
    
    log_likelihood,transition = tf.contrib.crf.crf_log_likelihood(logits,labels,transition_params=trans ,sequence_lengths=mask2len)
    # loss = tf.math.reduce_mean(-log_likelihood)
    loss = tf.reduce_mean(-log_likelihood)

    return loss,transition

def softmax_layer(logits, labels, mask):

    # shape:[batch_size, see_len]-->[batch_size]
    mask2len = tf.reduce_sum(mask, axis=1)

    # shape: [batch_size, seq_len]
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=labels)
    mask2 = tf.sequence_mask(mask2len)

    losses = tf.boolean_mask(losses, mask2)
    loss = tf.reduce_mean(losses)

    labels_softmax_ = tf.argmax(logits, axis=-1)
    predict = tf.cast(labels_softmax_, tf.int32)

    return loss, predict


def create_model(bert_config, is_training, input_ids, mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings, use_lstm):
    model = modeling.BertModel(
        config = bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
        )

    #shape: [batch_size, sequence_length, hidden_dim]
    output_layer = model.get_sequence_output()
    if is_training:
        output_layer = tf.keras.layers.Dropout(rate=FLAGS.drop_prob)(output_layer)

    if use_lstm:
        logging.info("using bilstm...")
        output_layer = bi_lstm(output_layer, mask, FLAGS.lstm_layer)
    if is_training:
        output_layer =  tf.nn.dropout(output_layer, 1-FLAGS.drop_prob)
    logits = hidden2tag(output_layer,num_labels)
    # TODO test shape
    logits = tf.reshape(logits,[-1,FLAGS.max_seq_length,num_labels])
    if FLAGS.crf==0:
        mask2len = tf.reduce_sum(mask,axis=1)
        loss, trans = crf_loss(logits,labels,mask,num_labels,mask2len)
        predict,viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
        return (loss, logits,predict)
    elif FLAGS.crf==1:
        loss,predict = softmax_layer(logits, labels, mask)
        return (loss, logits, predict)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        mask = features["mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, logits,predicts) = create_model(bert_config, is_training, input_ids,
                                                        mask, segment_ids, label_ids,num_labels,
                                                        use_one_hot_embeddings, FLAGS.use_lstm)

        tvars = tf.trainable_variables()

        scaffold_fn = None
        initialized_variable_names=None
        if init_checkpoint:

            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:

                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            logging.info("**** Trainable Variables ****")

            logging.info("=======================================variables to fine tune=============================================")
            tvars = tvars[FLAGS.layer:]
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, FLAGS.layer)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, logits,num_labels,mask):
                predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
                cm = metrics.streaming_confusion_matrix(label_ids, predictions, num_labels-1, weights=mask)
                return {
                    "confusion_matrix":cm
                }
                #
            eval_metrics = (metric_fn, [label_ids, logits, num_labels, mask])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn



def main():
    logging.set_verbosity(logging.INFO)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    vocab = read_vocab(FLAGS.vocab_file)
    processor = NerProcessor(max_len=FLAGS.max_seq_length, vocab=vocab)

    label_list = processor.get_labels()

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.train_data, FLAGS.data_augment, is_mask=FLAGS.is_mask)

        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)


    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")

        convert_examples_features(train_examples, train_file, FLAGS.max_seq_length)
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.eval_data)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        convert_examples_features(eval_examples, eval_file, FLAGS.max_seq_length)

        logging.info("***** Running evaluation *****")
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=eval_input_fn)

        logging.info("===================predicting=========================")
        preds = [row for row in result]
        sents, lengths, _ = get_sent_length_label(eval_examples)
        write_to_file(preds=preds, lengths=lengths, sent2ids=sents, file=FLAGS.eval_result_file, vocab=vocab)
        valid(result_file=FLAGS.eval_result_file, label_file=FLAGS.eval_data)


    if FLAGS.do_predict:

        predict_examples = processor.get_test_examples(FLAGS.test_data)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")

        convert_examples_features(predict_examples, predict_file, FLAGS.max_seq_length)
        logging.info("***** Running prediction*****")
        logging.info("  Num examples = %d", len(predict_examples))
        logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)
        logging.info("===================predicting=========================")
        preds = [row for row in result]
        sents, lengths, _ = get_sent_length_label(predict_examples)

        write_to_file(preds=preds, lengths=lengths, sent2ids=sents, file=FLAGS.result_file, vocab=vocab)


if __name__ == "__main__":
    main()
