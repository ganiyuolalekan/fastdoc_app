import os
import re
import json
import time
import openai
import requests
import chromadb
import functools
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

from .classes import GenerationModel
from .vector_db_funcs import HOST, PORT
from .prompts import generation_prompt_template
from .variables import OPENAI_API_KEY, SEPARATORS
from .vector_db_funcs import add_data_to_vector_db, create_organization, get_vectorstore
from .variables import base_url, conversational_llm, conversational_prompt, generated_text_desc

import boto3
from botocore.exceptions import NoCredentialsError, EndpointConnectionError, ClientError

load_dotenv()

openai.api_key = OPENAI_API_KEY

client = chromadb.HttpClient(host=HOST, port=PORT)

db = {}

IS_STREAMLIT_APP = True


# Wrapper function to time other functions

def time_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        if IS_STREAMLIT_APP:
            st.markdown(f"> {func.__name__} execution time: {execution_time:.4f} seconds")
        return result

    return wrapper


def exceptions_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if IS_STREAMLIT_APP:
                st.markdown(f"> The program failed, an error occurred in `{func.__name__}`: **{e}**. Please reach out to the developer")
            return None

    return wrapper


def json_to_dict(json_str):
    """Converts a JSON string to a Python dictionary."""

    return json.loads(json_str)


def dict_to_json(dict_data):
    """Converts a Python dictionary to a JSON string."""

    return json.dumps(dict_data)


@time_function
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


@time_function
def get_issues(issue_key):
    url = base_url + f'issues/{issue_key}'

    response = requests.request("POST", url, data=dict_to_json({}))

    return response.text


@exceptions_handler
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


@time_function
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


@time_function
def convert_report(text):
    """Converts generated text into a document"""

    doc = Document(page_content=text)
    doc.metadata["page"] = 1
    doc.metadata["source"] = 1

    return [doc]


@time_function
def embed_docs(docs: List[Document]) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS index"""

    embeddings = OpenAIEmbeddings()
    index = FAISS.from_documents(docs, embeddings)

    return index


@time_function
def write_memory(conversation):
    """Writes from a memory and create a chain for the conversational model"""

    retrieved_messages = messages_from_dict(json_to_dict(conversation))
    retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
    retrieved_memory = ConversationBufferMemory(chat_memory=retrieved_chat_history, memory_key="chat_history",
                                                input_key="human_input")

    return load_qa_chain(
        llm=conversational_llm, chain_type="stuff", memory=retrieved_memory, prompt=conversational_prompt
    )


@time_function
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


@time_function
def save(project_id, generated_report, generated_text_tracker, conv_chain, return_json_data=False):
    """
    Save model chain by returning it's json value and
    stores that into a database to be used later
    """

    result_json = dict_to_json({
        'conv_chain': conv_chain,
        'generated_report': generated_report,
        'generated_text_tracker': generated_text_tracker
    })

    db[project_id] = result_json

    if return_json_data:
        return result_json


@time_function
def load(project_id):
    """Loads a FastDoc object from a json object"""

    try:
        return json_to_dict(db[project_id])
    except TypeError:
        return dict_to_json({
            'status': 403,
            'log': "Failed to load project!!!"
        })


@time_function
def generate_text(project_id, text_content, tone, doc_type, url, org, goal=None, temperature='variable'):
    """Function to generate report"""

    temp = {
        'stable': 0.,
        'variable': 1.,
        'highly variable': 1.9
    }

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    conv_chain = load_qa_chain(llm=conversational_llm, chain_type="stuff", memory=memory, prompt=conversational_prompt)

    org_info = get_relevant_doc_from_vector_db(goal, url, org)

    generation_custom_functions = [
        {
            'name': 'text_generation',
            'description': generated_text_desc,
            'parameters': GenerationModel.schema()
        }
    ]

    if IS_STREAMLIT_APP:
        st.markdown("## Here's the EPIC pulled from JIRA")
        divider()
        st.write(text_content)
        divider()
        st.markdown("## Here's the Organization information extracted (strictly fastdoc)")
        divider()
        st.write(org_info)
        divider()

    response = openai.ChatCompletion.create(
        temperature=temp[temperature],
        model='gpt-3.5-turbo-16k',
        max_tokens=5120,
        messages=[{
            'role': 'user',
            'content': generation_prompt_template(
                doc_type, tone, text_content, org_info, goal
            )}],
        functions=generation_custom_functions,
        function_call={"name": "text_generation"}
    )

    result = eval(response['choices'][0]['message']['function_call']['arguments'])

    save(project_id, result['generated_text'], [result['generated_text']], read_memory(conv_chain))

    return result


@time_function
def regenerate_report(project_id, human_input):
    """Regenerates the results based on the users request"""

    json_data = load(project_id)

    try:
        conv_chain = write_memory(json_data['conv_chain'])
        generated_report = json_data['generated_report']
        generated_text_tracker = json_data['generated_text_tracker']

        result = conv_chain({
            "input_documents": convert_report(generated_report),
            "human_input": human_input
        }, return_only_outputs=True)

        generated_report = result['output_text'].strip()
        generated_text_tracker.append(generated_report)

        save(project_id, generated_report, generated_text_tracker, read_memory(conv_chain))

        return generated_report
    except KeyError:
        return json_data


@time_function
def write_to_s3(data, s3_file_name, bucket_name="fastdoc"):
    """
    Write data to a file in an AWS S3 bucket.

    :param bucket_name: str
        Name of the S3 bucket
    :param s3_file_name: str
        The name that the file should have in the S3 bucket
    :param data: str
        The data to be written to the file

    :return: str
        Success message if data was written, else an error message
    """

    # Create an S3 client from your session
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION")
    )

    try:
        # Encode the string data to bytes
        data_bytes = data.encode('utf-8')
        s3.put_object(Body=data_bytes, Bucket=bucket_name, Key=s3_file_name)
        return "Successfully wrote to S3"
    except NoCredentialsError:
        return "Credentials not available"
    except (EndpointConnectionError, ClientError) as e:
        return f"Error: {str(e)}"

