import json
import whisper
import os
import time
from whisper.utils import get_writer

model = whisper.load_model(r"C:\Users\nigama\.cache\whisper\base.pt")



def match_language(lang_code: str) -> str:
    """
    Method to match the language code detected by Whisper to full name of the language
    """
    with open("backend_app/lang.json", "rb") as f:
        lang_data = json.load(f)

    return lang_data[lang_code].capitalize()


def transcribe(filename):
    """

    :param filename:
    :return:
    """
    filename = os.path.join(os.getcwd()+"/recordings",filename).replace("\\","/")
    audio = whisper.load_audio(filename)
    # audio = whisper.pad_or_trim(audio)
    # Pass the audio file to the model and generate transcripts
    print("--------------------------------------------")
    print("Attempting to generate transcripts ...")
    data = {}
    result = model.transcribe(audio)
    data['text'] = result["text"].strip()
    data['lang'] = result["language"]
    print("Succesfully generated transcripts")
    return json.dumps(data)