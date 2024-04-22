import os
import openai
import streamlit as st

from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from .prompts import conversation_prompt_template

load_dotenv()

TEST_LOCAL = eval(os.getenv('TEST_LOCAL', 'False'))

if TEST_LOCAL:
    # Local Development
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
else:
    # Production Development
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

openai.api_key = OPENAI_API_KEY

title_desc = "Concise and meaningful title given to a generated text"
generated_text_desc = "Generated text provided by the model"

fastdoc_url = "https://fastdoc.io/"
base_url = "https://fastdoc-jira-integration.onrender.com/"

conversational_llm = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-4-1106-preview",
    max_tokens=5120
)

conversational_prompt = PromptTemplate(
    template=conversation_prompt_template,
    input_variables=["chat_history", "human_input", "context"]
)

SEPARATORS = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
