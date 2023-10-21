# Initializing project
import os
import re
import json
import openai
import sqlite3
import requests

from typing import List
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from vector_db_funcs import add_data_to_vector_db, create_organization, get_vectorstore

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.chains import LLMChain, load_chain
from langchain.document_loaders import WebBaseLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

generation_prompt_template = lambda doc_type, tone, goal="None": f"""Understand and study the context below, and use it to write/compose a {doc_type} write-up with a descriptive title as <title> and its content (the generated text) as <gen_text>. Ensure your generated text is only in the notable standardised format that matches the {doc_type} format of writing. Use the information about the organization to fine tune your generated text. Also, ensure it is detailed enough and does not include “accountid” information from the context below. Also, use a {tone} tone in your generated output. 
You're to write towards addressing this goal "{goal}", if the provided goal is None, then generate your text only in context to {doc_type} format, using the context below to gain scope/context on your write-up.
Never copy text from the context or use it to fill points in your generated text only when necessary. Also, <title> must never appear in your <gen_text> if <gen_text> must have a title give it something entirely different from <title>.""" + """You must always return your result in the format specified below alone. Finally, ensure your generated text never exceeds 3072 tokens and it must be quoted in triple single quotes. All quotes and square braces MUST be closed accurately in the order they appear in the format.

    Format: "[["title", '<title>'], ["generated_text", '''<gen_text>''']]"
    Context: {context}
    Organization Information: {org_info}"""

conversation_prompt_template = """You are a text modification/improvement bot. Given a text as input, your role is to re-write an improved version of the text template based on the human question and what you understand from your chat history. You're not to summarise the text but add intuitive parts to it or exclude irrelevant parts from it. Answer the human questions by modifying the text ONLY, maintaining the paragraphs and point from the input text.
You're not to add any comment of affrimation to you text, just answer the question by rewriting the text only.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

base_url = "https://fastdoc-jira-integration.onrender.com/"

conversational_llm = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo-16k",
    max_tokens=5120
)

conversational_prompt = PromptTemplate(
    template=conversation_prompt_template,
    input_variables=["chat_history", "human_input", "context"]
)

SEPARATORS = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]


class ClassDatabase:
    """A database for storing class objects."""

    def __init__(self, db_file):
        self.db_file = db_file
        self._connection = sqlite3.connect(db_file)

        try:
            self._create_table()
        except sqlite3.OperationalError:
            pass

    def add_class(self, class_object, _id):
        """Adds a class object to the database."""

        try:
            self._connection.execute(
                "INSERT INTO classes (id, class) VALUES (?, ?)", (_id, json.dumps(class_object)))
        except sqlite3.IntegrityError as e:
            self.update_class(_id, class_object)

    def get_class(self, _id):
        """Returns the class object with the given id."""

        cursor = self._connection.execute("SELECT class FROM classes WHERE id = ?", (_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def update_class(self, _id, class_object):
        """Updates the class object with the given id."""

        self._connection.execute(
            "UPDATE classes SET class = ? WHERE id = ?", (json.dumps(class_object), _id))

    def delete_class(self, _id):
        """Deletes the project with the given id."""

        self._connection.execute("DELETE FROM classes WHERE id = ?", (_id,))

    def _create_table(self):
        """Creates the classes table in the database."""

        self._connection.execute("""CREATE TABLE classes (
          id INTEGER PRIMARY KEY,
          class TEXT
        )""")

    def close(self):
        """Closes the database connection."""

        self._connection.close()


db = ClassDatabase("class_database.db")


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


def get_relevant_doc_from_vector_db(url, query, org="fastdoc", _id=None, metadata=None):
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
    relevant_docs = docsearch.max_marginal_relevance_search(query, k=1)

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

    db.add_class(result_json, project_id)

    if return_json_data:
        return result_json


def load(project_id):
    """Loads a FastDoc object from a json object"""

    try:
        return json_to_dict(db.get_class(project_id))
    except TypeError:
        return dict_to_json({
            'status': 403,
            'log': "Failed to load project!!!"
        })


def get_title_generated_text(response):
    result = {'title': None, 'generated_text': None}

    for i, res in enumerate(eval(response)):
        key, content = res

        if key == "generated_text":
            result[key] = "\n\n".join([point for point in content.split('\n\n') if point.strip() != result['title'].strip()]).strip()
        else:
            result[key] = content

    return result


def generate_text(project_id, text_content, tone, doc_type, url, query, goal=None, temperature='variable'):
    """Function to generate report"""

    temp = {
        'stable': 0.,
        'variable': .5,
        'highly variable': .9
    }

    generation_llm = ChatOpenAI(
        temperature=temp[temperature],
        model_name="gpt-3.5-turbo-16k",
        max_tokens=5120
    )

    generated_prompt = PromptTemplate(
        template=generation_prompt_template(
            doc_type, tone, goal
        ), input_variables=["context", "org_info"]
    )

    docs = text_to_doc(text_content)
    if goal is not None:
        index = embed_docs(docs).similarity_search(f"What best addresses this goal {goal}", k=4)
    else:
        index = embed_docs(docs).similarity_search("What are the most relevant documents here", k=4)

    inputs = [
        {
            "context": i.page_content,
            "org_info": get_relevant_doc_from_vector_db(url, query)
        }
        for i in index
    ]
    gen_chain = LLMChain(llm=generation_llm, prompt=generated_prompt)
    generated_report = gen_chain.apply(inputs)[0]['text'].strip()

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    conv_chain = load_qa_chain(llm=conversational_llm, chain_type="stuff", memory=memory, prompt=conversational_prompt)

    save(project_id, generated_report, [generated_report], read_memory(conv_chain))

    return get_title_generated_text(generated_report)


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


def init_project(json_input):
    """
    Initializes a new project and creates in id for it in our local database
    """

    try:
        keys = json_to_dict(json_input)

        content, issue_key_type = write_out_report(keys['scope'])

        try:
            temp = keys['temperature']
        except KeyError:
            temp = 'variable'

        result = generate_text(
            keys['project_id'],
            content,
            keys['tone'],
            keys['doc_type'],
            keys['url'],
            keys['query'],
            keys['goal'],
            temperature=temp
        )

        title = result['title']
        text = result['generated_text']

        return dict_to_json({
            'status': 200,
            'title': title,
            'generated_text': text,
            'issue_key_type': issue_key_type,
            'log': "Successfully generated report!!!"
        })
    except Exception as e:
        return dict_to_json({
            'status': 503,
            'log': f"Program failed with exception {e}"
        })


def return_project_value(json_input):
    """Responsible for continuous query for a particular database"""

    try:
        keys = json_to_dict(json_input)

        re_gen_report = regenerate_report(
            keys['project_id'],
            keys['user_query']
        )

        if type(re_gen_report) == str:
            return dict_to_json({
                're-generated_text': re_gen_report,
                'status': 200,
                'log': "Successfully re-generated report!!!"
            })
        else:
            return re_gen_report
    except Exception as e:
        return dict_to_json({
            'status': 503,
            'log': f"Program failed with exception {e}"
        })


def delete_project(json_input):
    """Responsible for delete a project from ML database"""

    try:
        keys = json_to_dict(json_input)
        db.delete_class(keys['project_id'])

        return dict_to_json({
            'status': 200,
            'log': "Successfully deleted projects!!!"
        })
    except Exception as e:
        return dict_to_json({
            'status': 503,
            'log': f"Program failed with exception {e}"
        })

