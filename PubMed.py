#!/usr/bin/env python
# coding: utf-8

# # ELI5 - Healthcare Research Papers from PubMed
# Explain Like I'm 5, a research paper from PubMed.

# ### Install Libraries

# In[1]:


#!pip -q install langchain openai tiktoken
#!pip install rouge
#!pip install wordcloud


# In[9]:


#!pip install pubmed_parser


# In[1]:


from dotenv import load_dotenv
load_dotenv()


# ### Scrape Abstract Text From PubMed URL

# In[15]:


url = 'https://pubmed.ncbi.nlm.nih.gov/33755728/'


# In[16]:


import requests
from bs4 import BeautifulSoup

# Fetch the HTML content from the URL
response = requests.get(url)
html_content = response.text

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

# Find the div with class 'abstract' and id 'abstract'
abstract_div = soup.find('div', {"class": "abstract", "id": "abstract"})

# Extract and print the text from the div if it's found
if abstract_div:
    abstract_text = abstract_div.get_text(strip=True)
    print(abstract_text)
else:
    print("Abstract not found.")


# In[167]:


from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

model_name = "gpt-3.5-turbo"

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    model_name=model_name
)

texts = text_splitter.split_text(abstract_text)

docs = [Document(page_content=t) for t in texts]
print(len(docs))


# #### Create API Key as Environment File

# In[24]:


env_content = "OPENAI_API_KEY='sk-abcd'"
#Replace this with your own API KI

#with open('.env', 'w') as file:
#    file.write(env_content)
#print("File '.env' has been created and the API key information has been written into it.")


# In[25]:


import sys
import os
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


# In[26]:


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name=model_name)


# ### Prompt Engineering

# In[48]:


from langchain.prompts import PromptTemplate

prompt_template = """

Read the {text} carefully, then explain the extract, line by line, in an easy to understand language.

"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


# In[96]:


prompt_template_2 = """

Read the {text} carefully, if there is data present in the conclusion section,
then extract it and arrange it in form of a dataframe. 

"""

prompt2 = PromptTemplate(template=prompt_template_2, input_variables=["text"])



# ### Fetch Number of Tokens

# In[82]:


import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num_tokens = num_tokens_from_string(abstract_text, model_name)
print(num_tokens)


# In[118]:


from langchain.chains.summarize import load_summarize_chain
import textwrap
from time import monotonic

gpt_35_turbo_max_tokens = 4097
verbose = True

if num_tokens < gpt_35_turbo_max_tokens:
  #chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, verbose=verbose)
  chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, verbose=verbose)

else:
  chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt,combine_prompt=prompt, verbose=verbose)

start_time = monotonic()
summary = chain.run(docs)


# In[156]:


if num_tokens < gpt_35_turbo_max_tokens:
  #chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, verbose=verbose)
  chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt2, verbose=verbose)

else:
  chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt2,combine_prompt=prompt, verbose=verbose)

summary2 = chain.run(docs)


# ## Summary

# In[157]:


print(summary)


# In[158]:


print(summary2)


# ### Data Cleaning

# In[159]:


# Convert the markdown table to a dataframe
import pandas as pd
from io import StringIO

# Using StringIO to simulate a file-like object
data = StringIO(summary2)

# Creating the DataFrame
df = pd.read_csv(data, sep="|", skipinitialspace=True)

# Trimming whitespace from all string elements in the DataFrame
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


# In[160]:


if df.iloc[:, 0].isnull().all():
    df = df.drop(df.columns[0], axis=1)


# In[161]:


if df.iloc[:, -1].isnull().all():
    df = df.drop(df.columns[-1], axis=1)


# In[162]:


def check_dash_or_float(cell):
    if isinstance(cell, float):
        return False  # floats are not considered as '---'
    return all(c == '-' for c in str(cell))  # Check if all characters are dashes in non-floats

# Check if all values in the first row contain only dashes
if all(check_dash_or_float(cell) for cell in df.iloc[0]):
    df = df.drop(df.index[0])


# In[163]:


df.head()


# ### Auto Data Visualization

# In[150]:


docs = [Document(page_content=row.to_json()) for index, row in df.iterrows()]
print(len(docs))


# In[164]:


prompt_template_3 = """

Read the {text} carefully, create a visualization to summarize this dataframe. 
Do not consider the columns with N/A as values.
Provide with only the code to create the visualization and nothing else. 

"""

prompt3 = PromptTemplate(template=prompt_template_3, input_variables=["text"])
chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt3, verbose=verbose)
summary3 = chain.run(docs)


# In[168]:


print(summary3)


# In[166]:


exec(summary3)

