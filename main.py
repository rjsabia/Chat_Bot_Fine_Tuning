import os
import openai
import requests

import pandas as pd

# Environmental variables

# OLD KEY NOT Working
# openai.api_key = os.getenv('OPEN_AI_KEY_01')
openai.api_key = os.getenv('OPEN_AI_KEY_03')



# --------------------------------------------

qa_df = pd.read_csv("python_qa.csv")

questions , answers = qa_df['Body'] , qa_df['Answer']
qa_openai_format =  [ {"prompt":q, "completion":a} for q,a in zip(questions,answers)]

# testing output
# print(qa_openai_format[4])
# qa_openai_format[4]['prompt']


response = openai.Completion.create(
            model="text-babbage-001",
            prompt = qa_openai_format[4]['prompt'],
            max_tokens = 250,
            temperature = 0
)

# response['choices'][0]['text']
print(response['choices'][0]['text'])














