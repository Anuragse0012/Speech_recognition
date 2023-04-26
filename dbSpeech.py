import json
import sqlite3


def insert_transcript(filename:str, timestamp:str,  language:str, transcript:str):
    try:
        conn = sqlite3.connect('SpeechRecognition.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO speech_transcript VALUES(?,?,?,?)", (filename, timestamp, language, transcript))
        conn.commit()
        conn.close()
    except Exception: return {"message": "failure"}
    return {"message": "success"}

def select_transcript(filename:str):
    try:
        conn = sqlite3.connect('SpeechRecognition.db')
        cursor = conn.cursor()
        rows = cursor.execute("SELECT language,transcript FROM speech_transcript").fetchall()
        data = {'transcript': rows[0][1], 'language': rows[0][0]}
        conn.commit()
        conn.close()
    except Exception: return {"message": "failure"}
    return json.dumps(data)


