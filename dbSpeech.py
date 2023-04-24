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



