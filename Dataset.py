# Adapted from the following tutorial:
# https://www.tensorflow.org/text/tutorials/nmt_with_attention

import numpy as np
import typing
from typing import Any, Tuple
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pathlib

# Download the file
path_to_zip = tf.keras.utils.get_file('spa-eng.zip',
                                      origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
                                      extract=True)

path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'
print(f"Downloaded dataset: {path_to_file}")

# Load into memory


def load_data(path):
    text = path_to_file.read_text(encoding='utf-8')

    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]

    return [targ for targ, inp in pairs], [inp for targ, inp in pairs]


eng, spa = load_data(path_to_file)

# Create tf.data.Dataset
#batch_size = 64
#dataset = tf.data.Dataset.from_tensor_slices((eng, spa)).shuffle(len(eng))
#dataset = dataset.batch(batch_size)
#
# def standardize(text):
#    # Normalize the data onto UTF-8 and add [START] and [END] tokens.
#
#    # Split accecented characters.
#    text = tf_text.normalize_utf8(text, 'NFKD')
#    text = tf.strings.lower(text)
#    # Keep space, a to z, and select punctuation.
#    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
#    # Add spaces around punctuation.
#    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
#    # Strip whitespace.
#    text = tf.strings.strip(text)
#    # Append start and end tokens
#    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
#    return text
#
#
# Build vocabularies
# input_text_processor = preprocessing.TextVectorization(
#    standardize=standardize, max_tokens=5000)
# output_text_processor = preprocessing.TextVectorization(
#    standardize=standardize, max_tokens=5000)
