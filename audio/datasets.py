import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

from . import shift

def get_data(DATASET_PATH):
    data_dir = pathlib.Path(DATASET_PATH)
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    return data_dir, commands

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def prepare_data(data_dir):
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=64,
        validation_split=0.2, 
        seed=0, 
        output_sequence_length=16000, 
        subset='both') 
    label_names = np.array(train_ds.class_names)
    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)
    return train_ds, val_ds, label_names

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (shift.get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

def shift_data(train_ds, val_ds, test_ds):
    train_spectrogram_ds = make_spec_ds(train_ds)
    val_spectrogram_ds = make_spec_ds(val_ds)
    test_spectrogram_ds = make_spec_ds(test_ds)
    
    train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_spectrogram_ds, val_spectrogram_ds, test_spectrogram_ds