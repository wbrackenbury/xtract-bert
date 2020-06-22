import os
import logging
import argparse
import pprint
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from gensim import corpora, models
from gensim.models import KeyedVectors, Word2Vec, ldamodel
from official.nlp.bert import tokenization
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.snowball import SnowballStemmer

from txt_xtract import process_text
from utils import grouper, get_ext

HUB_URL = "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1"
MAX_SEQ_LEN = 128
MAX_GROUP_SIZE = 100
WORD_VECTORS_TO_READ = 500000
BERT_NAME = 'bert'
#W2V_MODEL_PATH = "/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin.gz"
W2V_MODEL_PATH = "/google_news_vec.bin.gz"

def tok_and_stem(doc):

    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    snowball_stemmer = SnowballStemmer('english')

    raw = doc.lower()
    tokens = tokenizer.tokenize(raw)

    stopped_tokens = [it for it in tokens if not it in en_stop]
    stemmed_tokens = [snowball_stemmer.stem(it) for it in stopped_tokens]
    return stemmed_tokens

def doc_vec(model, doc):
    """
    Create a makeshift document vector by taking the means of all word vectors
    """

    tok_and_stemmed_doc = tok_and_stem(doc)
    clean_doc = [word for word in tok_and_stemmed_doc if word in model.vocab]
    if clean_doc:
        return np.mean(model[clean_doc], axis = 0)
    else:
        example_vector = model.get_vector("garbage")
        return np.zeros(example_vector.shape)

def load_w2v_model():
    """
    Load the word2vec pretrained model from the data file
    """
    model = KeyedVectors.load_word2vec_format(W2V_MODEL_PATH, binary = True, \
                                              limit = WORD_VECTORS_TO_READ)
    model.init_sims(replace = True)

    return model

def get_bert_tokenizer(bert_layer):

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    return tokenizer

def get_bert_layer(HUB_URL, max_seq_length):


    in_shape = tf.convert_to_tensor((max_seq_length,))
    input_word_ids = tf.keras.layers.Input(shape=in_shape, dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=in_shape, dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=in_shape, dtype=tf.int32,
                                        name="segment_ids")
    bert_layer = hub.KerasLayer(HUB_URL,
                                trainable=False)
    return bert_layer

def bert_input(text, tokenizer, max_seq_length):

  tokens_a = tokenizer.tokenize(text)
  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  return input_ids, input_mask, segment_ids

def get_bert_rep(text, bert_layer, bert_token):

    full_out = []
    input_ids, input_mask, segment_ids = [], [], []

    mod_txt = text.replace('\n', '.')
    sentences = mod_txt.split('.')

    max_seq_length = MAX_SEQ_LEN  # Your choice here.

    for s in sentences:
        if len(s) >= max_seq_length:
            continue

        iid, mask, seg = bert_input(s, bert_token, max_seq_length)
        input_ids.append(iid)
        input_mask.append(mask)
        segment_ids.append(seg)

        if input_ids:
            temp_out, _ = bert_layer([tf.convert_to_tensor(input_ids),
                                      tf.convert_to_tensor(input_mask),
                                      tf.convert_to_tensor(segment_ids)])
            full_out.append(temp_out)

    if full_out:
        full_out = np.vstack(full_out)
        full_out = np.mean(full_out, axis=0)
        return full_out
    else:
        return []

def rep_base(path, model_choice):
    rep = {'path': path,
           'model': model_choice,
           'error': False,
           'error_reason': '',
           'rep': []}
    return rep

def get_rep_and_ext(path, model_choice):
    rep = rep_base(path, model_choice)
    ext = get_ext(path)
    if ext is None:
        rep['error'] = True
        rep['error_reason'] = "Cannot determine extension"
        rep['ext'] = ''
        return rep
    rep['ext'] = ext
    return rep

def extract_from_path(path, model_choice, model_items):

    rep = get_rep_and_ext(path, model_choice)
    if rep['error']:
        return rep
    with open(path, 'rb') as of:
        content = of.read()

    try:
        txt = process_text(content, rep['ext'])
        if model_choice == BERT_NAME:
            bert_layer, bert_token = model_items
            text_rep = get_bert_rep(txt, bert_layer, bert_token)
        else:
            text_rep = doc_vec(model_items, txt)
        rep['rep'] = text_rep
    except Exception as e:
        rep['error'] = True
        rep['error_reason'] = e

    return rep

def walk_paths(dir_path, model_choice, model_items):

    ret_blobs = []
    for root, dirs, fs in os.walk(dir_path):
        for f in fs:
            path = os.path.join(dir_path, f)
            logging.info("Extracting from {}...".format(f))
            rep = extract_from_path(path, model_choice, model_items)
            ret_blobs.append(rep)

    return ret_blobs

def load_bert_tools():

    max_seq_length = MAX_SEQ_LEN  # Your choice here.

    bert_layer = get_bert_layer(HUB_URL, tf.constant(max_seq_length))
    bert_token = get_bert_tokenizer(bert_layer)

    return bert_layer, bert_token


def load_model(model_choice):
    if model_choice == "bert":
        return load_bert_tools()
    else:
        return load_w2v_model()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path', help='Path to directory',
#                         required=False, type=str)
#     parser.add_argument('--model', help='bert | w2v',
#                         required=True, type=str)
#
#     args = parser.parse_args()
#
#     logging.info("Parsed args, now loading {} model...".format(args.model))
#     model_items = load_model(args.model)
#     logging.info("{} model loaded. Beginning walk and extraction.".format(args.model))
#
#     rep_jsons = walk_paths(args.path, args.model, model_items)
#     pprint.pprint(rep_jsons)
