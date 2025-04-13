from . import prepareDict as prep
from tqdm import tqdm
import speech_recognition as sr

def find_keys_by_value(dictionary, value):
    keys = [key for key, val in dictionary.items() if val == value]
    return keys[0]

def check_string_contains_dict_text(input_string, my_dict, lang):
    all_items = list(my_dict.keys()) + list(my_dict.values())
    for item in all_items:
        if item in input_string:
            if lang == 'Chinese':
                return find_keys_by_value(my_dict, item)
            else:
                return item
    return False

def prepareDict(DATASET_PATH):
    emotionComDict = prep.creatEmotionDict(DATASET_PATH)
    return emotionComDict

def createPoseDict():
    PoseDict = {
        "left": "左",
        "right": "右",
        "up": "上",
        "down": "下",
        "shake": "摇",
        "nod": "点"
    }
    return PoseDict

def createLevelsDict():
    LevelsDict = {
        "slight": "微",
        "large": "大",
    }
    return LevelsDict

def audio2Text(path, lang):
    r = sr.Recognizer()
    harvard = sr.AudioFile(path)
    with harvard as source:
        audio = r.record(source)
    if lang == 'Chinese':
        text = r.recognize_sphinx(audio, language='zh-CN')
    else:
        text = r.recognize_sphinx(audio) 
    return text

def getInstrustion(user_input, motionDict, lang):
    result = check_string_contains_dict_text(user_input, motionDict, lang)

    if result:
        return result
    else:
        result = 'Fixed'
        return result