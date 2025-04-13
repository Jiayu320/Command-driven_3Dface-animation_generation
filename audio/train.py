import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

from . import datasets
from . import model
from . import shift

# Set the seed value for experiment reproducibility. 设置实验可重复性的种子值。
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


DATASET_PATH = 'data/command_ness'
data_dir, commands = datasets.get_data(DATASET_PATH)
train_ds, val_ds, label_names = datasets.prepare_data(data_dir)
train_spectrogram_ds, val_spectrogram_ds, test_spectrogram_ds = datasets.shift_data(train_ds, val_ds, val_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

model = model.createModel(example_spectrograms, label_names, train_spectrogram_ds)

EPOCHS = 20
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    #在损失不再改善（连续patience次验证周期没有提升）的情况下提前停止训练过程，防止过拟合
    #verbose=1表示显示详细的日志信息
)

class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    # YOu could add additional signatures for a single wave, or a ragged-batch.
    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
       x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))


  @tf.function
  def __call__(self, x):
    # If they pass a string, load the file and decode it.
    if x.dtype == tf.string:
      x = tf.io.read_file(x)
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
      x = tf.squeeze(x, axis=-1)
      x = x[tf.newaxis, :]

    x = shift.get_spectrogram(x)
    result = self.model(x, training=False)

    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(label_names, class_ids)
    return class_names

export = ExportModel(model)
tf.saved_model.save(export, "saved")