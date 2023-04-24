from typing import Union
from fastapi import FastAPI, File, UploadFile
from test import *
import json
from types import SimpleNamespace
import whisper
from whisper.utils import get_writer
from recognition import *
from translation import *
from summarization import *
from dbSpeech import *
import datetime

model = whisper.load_model(r"C:\Users\nigama\.cache\whisper\base.pt")
data = dict()
folder_path = os.getcwd()

app = FastAPI()


@app.post("/api/audiofile/")
async def upload(file: UploadFile = File(...)):
    """
    Upload a file
    :param file:
    :return:
    """
    try:
        print("Attempting to generate transcripts ...")
        contents = file.file.read()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(folder_path, f"audio_{timestamp}.wav")
        print(filename)
        with open(filename, 'wb') as f:
            f.write(contents)
        result = transcribe(filename=filename)
        x = json.loads(result, object_hook=lambda d: SimpleNamespace(**d))
        message = insert_transcript(str(file.filename), str(timestamp), str(x.lang), str(x.text))
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return message


@app.post("/api/translation")
def getTranslation(text: str, from_code: str, to_code: str):
    return translate(text, from_code, to_code)


@app.post("/api/summarize")
def getSummarize(text: str):
    return summarizer(text=text)

# run command : uvicorn main:app --reload
