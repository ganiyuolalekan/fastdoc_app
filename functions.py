import os
import re
import json
import openai
import requests
import chromadb
import streamlit as st

from typing import List
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

from vector_db_funcs import HOST, PORT
from vector_db_funcs import add_data_to_vector_db, create_organization, get_vectorstore
from variables import base_url, conversational_llm, conversational_prompt, SEPARATORS

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

client = chromadb.HttpClient(host=HOST, port=PORT)


def json_to_dict(json_str):
    """Converts a JSON string to a Python dictionary."""

    return json.loads(json_str)


def dict_to_json(dict_data):
    """Converts a Python dictionary to a JSON string."""

    return json.dumps(dict_data)


def clean_string(input_string):
    cleaned_string = re.sub(r"http\S+|www\S+|https\S+", "", input_string)
    cleaned_string = re.sub(r"\[\~accountid:[a-fA-F0-9]+\]", "", cleaned_string)

    return cleaned_string


def clean_html_and_css(text):
    # Parse the text as HTML using BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')

    # Remove all HTML tags
    cleaned_text = soup.get_text()

    return cleaned_text


def get_relevant_doc_from_vector_db(goal, url="https://fastdoc.io/", org="fastdoc", _id=None, metadata=None):
    try:
        client.get_collection(name=org)
    except:
        loader = WebBaseLoader(url)
        data = loader.load()
        content = clean_html_and_css(data[0].page_content)

        create_organization(org)

        if _id is None:
            _id = '12345'
        if metadata is None:
            metadata = {}

        data = {
            'org': org,
            'ids': ['12345'],
            'contents': [content],
            'metadatas': [metadata]
        }

        add_data_to_vector_db(data)

    docsearch = get_vectorstore(org)
    relevant_docs = docsearch.max_marginal_relevance_search(goal, k=1)

    return relevant_docs[0].page_content.strip()


def get_issues(issue_key):
    url = base_url + f'issues/{issue_key}'

    response = requests.request("POST", url, data=dict_to_json({}))

    return response.text


def write_out_report(issue_key):
    issue = json_to_dict(get_issues(issue_key))

    issues = issue['issue']['issues']

    withdraw_pattern = re.compile(r'with\s?drawn*|with\s?drew', re.IGNORECASE)

    try:
        if issues[0]['fields']['parent']['key'] == issue_key:
            issue_key_type = "Parent Issue"
        else:
            issue_key_type = "Child Issue"
    except KeyError:
        issue_key_type = "Parent Issue"

    content = ""
    for _fields in issues:
        fields = _fields['fields']
        try:
            ticket_type = fields['fields']['status']['name']
        except KeyError:
            ticket_type = fields['status']['name']

        if not withdraw_pattern.match(ticket_type):
            content += f"Summary:\n\n{fields['summary']}\n\n"
            content += f"Description:\n\n{fields['description']}\n\n"

            comments = fields['comment']['comments']
            content += "Comments:\n\n"
            for comment in comments:
                content += f"{comment['body']}\n"

    return clean_string(content.strip()), issue_key_type


def text_to_doc(text, chunk_size=1600):
    """
    PRIVATE text_chunking  METHOD

    Converts document (text) to Document objects and split them
    Adds a source and metadata information to each document as well
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        separators=SEPARATORS,
        chunk_overlap=0
    )

    docs = [
        Document(page_content=split)
        for split in text_splitter.split_text(text)
    ]

    for i, doc in enumerate(docs):
        doc.metadata["page"] = str(i + 1)
        doc.metadata["source"] = str(i + 1)

    return docs


def convert_report(text):
    """Converts generated text into a document"""

    doc = Document(page_content=text)
    doc.metadata["page"] = 1
    doc.metadata["source"] = 1

    return [doc]


def embed_docs(docs: List[Document]) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS index"""

    embeddings = OpenAIEmbeddings()
    index = FAISS.from_documents(docs, embeddings)

    return index


def write_memory(conversation):
    """Writes from a memory and create a chain for the conversational model"""

    retrieved_messages = messages_from_dict(json_to_dict(conversation))
    retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
    retrieved_memory = ConversationBufferMemory(chat_memory=retrieved_chat_history, memory_key="chat_history",
                                                input_key="human_input")

    return load_qa_chain(
        llm=conversational_llm, chain_type="stuff", memory=retrieved_memory, prompt=conversational_prompt
    )


def read_memory(chain):
    """Reads a memory and stores the serialized file"""

    extracted_messages = chain.memory.chat_memory.messages

    return dict_to_json(messages_to_dict(extracted_messages))


def app_meta():
    """Adds app meta data to web applications"""

    # Set website details
    st.set_page_config(
        page_title="FastDoc | Document Generator",
        page_icon="images/fastdoc.png",
        layout='centered'
    )


def divider():
    """Sub-routine to create a divider for webpage contents"""

    st.markdown("""---""")
