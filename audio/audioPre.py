from .vad_ex import EnergyVAD
import torchaudio
import torch

import librosa
import soundfile as sf

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

def resample_rate_test(path,new_sample_rate = 16000):
    signal, sr = librosa.load(path, sr=None)
    wavfile = path.split('/')[-1]
    wavfile = wavfile.split('.')[0]
    file_name = wavfile + '_new.wav'
    if sr is not None and sr != new_sample_rate:
        new_signal = librosa.resample(signal, orig_sr=sr, target_sr=new_sample_rate)
    folder_path = 'trans_test_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    sf.write('trans_test_data/' + file_name, new_signal, new_sample_rate)
    file_name = 'trans_test_data/' + file_name
    return file_name

def Energy_waveform(path, voice_activity):
    audio, sample_rate = torchaudio.load(str(path))
    energy_waveform = []
    for i in range(len(voice_activity)):
        if voice_activity[i] == 1:
            energy_waveform.append(audio.numpy()[0, i * 320 : (i + 1) * 320])
    energy_waveform = np.concatenate(energy_waveform)
    # convert to tensor
    energy_waveform = torch.from_numpy(energy_waveform)
    energy_waveform = energy_waveform.unsqueeze(0)
    # save the new audio
    wavfile = path.split('/')[-1]
    wavfile = wavfile.split('.')[0]
    folder_path = 'energy_test_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    energy_waveform_path = 'energy_test_data/' + wavfile + '_Energy_new.wav'
    torchaudio.save(str(energy_waveform_path), energy_waveform, sample_rate)
    #print(energy_waveform_path)
    return energy_waveform, energy_waveform_path

def vadEnergy_audio(path, sample_rate=16000):
    audio, sample_rate = torchaudio.load(str(path))
    vad = EnergyVAD(
        sample_rate= sample_rate,
        frame_length= 25, # in milliseconds
        frame_shift= 20, # in milliseconds
        energy_threshold= 0.05, # you may need to adjust this value
        pre_emphasis= 0.95,
        ) # default values are used here
    voice_activity = vad(audio) # returns a boolean array indicating whether a frame is speech or not
    energy_waveform, energy_waveform_path = Energy_waveform(path, voice_activity)
    x = tf.audio.encode_wav(energy_waveform, sample_rate=16000, name=None)
    return voice_activity, energy_waveform, energy_waveform_path



