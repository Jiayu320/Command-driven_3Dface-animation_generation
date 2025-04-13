import pathlib
import numpy as np
import tensorflow as tf

import re
from translate import Translator   

def get_data(DATASET_PATH):
    data_dir = pathlib.Path(DATASET_PATH)
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    return data_dir, commands

def remove_de_in_chinese(text):
    result = re.sub(r'的', '', text)
    result = re.sub(r'地', '', result)
    result = re.sub(r'得', '', result)
    return result

def creatEmotionDict(DATASET_PATH):
    emotionComDict = {}
    translator = Translator(from_lang='english', to_lang='chinese')
    data_dir, emotionCommands = get_data(DATASET_PATH)
    
    for index, value in enumerate(emotionCommands):
        if value != '.ipynb_checkpoints':
            result = translator.translate(value)
            emotionComDict[value] = remove_de_in_chinese(result)
    return emotionComDict