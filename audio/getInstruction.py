import tensorflow as tf
from . import audioPre


def getInstruction(audio_path):
    audio_path = audioPre.resample_rate_test(audio_path,new_sample_rate = 16000)
    voice_activity, energy_waveform, energy_waveform_path = audioPre.vadEnergy_audio(str(audio_path), sample_rate=16000)
    imported = tf.saved_model.load("saved")
    result = imported(tf.constant(str(energy_waveform_path))).numpy()
    
    result = result[0].decode('utf-8')
    
    return result