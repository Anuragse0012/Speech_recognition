from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer

model = AutoModelWithLMHead.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

def summarizer(text:str):
    """

    :param text:
    :return:
    """
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf")
    #summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
    return (summarizer(text, max_length=130, min_length=30))

