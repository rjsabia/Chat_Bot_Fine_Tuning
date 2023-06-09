import os
import openai
import requests

import pandas as pd

import tiktoken
import json

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

# Control due to Davinci to test fine tuned model against
# response = openai.Completion.create(
            # model="text-davinci-003",
            # prompt = qa_openai_format[4]['prompt'],
            # max_tokens = 250,
            # temperature = 0
# )


response = openai.Completion.create(
            model="text-babbage-001",
            prompt = qa_openai_format[4]['prompt'],
            max_tokens = 250,
            temperature = 0
)

# response['choices'][0]['text']
# !!!!!!!! Need to un comment below to work !!!!!!!!!!!
# print(response['choices'][0]['text'])

# Get length of data frame
print('###########----->>>>\n','Length of Dataframe: ',len(qa_df))


def num_tokens_from_string(string,encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# set amount of lines to use in dataset
dataset_size = 500  # 500 out of the 4429 lines

# This will write the dataset to a new json file
with open("example_training_data.json","w") as f:
    for entry in qa_openai_format[:dataset_size]:
        f.write(json.dumps(entry))
        f.write('\n')

# set up counter for tokens
token_counter = 0

for prompt_completion in qa_openai_format:
    # for prompt,completion in element.items():
    for prompt,completion in prompt_completion.items():
        token_counter += num_tokens_from_string(prompt,'gpt2')
        token_counter += num_tokens_from_string(completion,'gpt2')

print(token_counter) # for babbage $0.0006 per 1000 tokens (training) * 4 epochs































