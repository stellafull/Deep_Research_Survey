"""Utilities and Tools for the survey agent system.

This module provides various utility functions and classes that support the
operation of the survey agent, including paper retrieval, decup overlap checking,
and content summarization tools.
"""



from langchain.chat_models import init_chat_model




# load env variables
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file



### summarization models
summarization_model = init_chat_model(
    model='Pro/deepseek-ai/DeepSeek-R1',
    provider='openai',
    base_url=os.getenv('LLM_BASE_URL'),
    api_key=os.getenv('LLM_API_KEY'),
    temperature=0,
    max_retries=3,
)



### tools