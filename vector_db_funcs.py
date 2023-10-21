import os
import openai
import chromadb

from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

HOST = '18.130.242.183'
PORT = 8000

client = chromadb.HttpClient(host=HOST, port=PORT)


class MissingOrganization(Exception):
    """
    Exception raised for errors for missing database.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Organization name not in database!!!"):
        self.message = message
        super().__init__(self.message)


def create_organization(org_name):
    try:
        client = chromadb.HttpClient(host=HOST, port=PORT)
        collection = client.create_collection(name=org_name)

        return {
            'status': 200,
            'log': "Successful!!!"
        }
    except Exception as e:
        return {
            'status': 403,
            'log': f"Failed with exception {e}"
        }


def text_to_docs(ids, metadatas, text):
    """Converts a string or list of strings to a list of Documents with metadata."""

    if isinstance(text, str):
        # Take a single string as one page
        text = [text]

    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []
    doc_id = []
    _metadatas = []

    for _id, meta, doc in zip(ids, metadatas, page_docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            separators=["\n\n", "\n", ".", "!", "?"],
            chunk_overlap=0,
        )

        chunks = text_splitter.split_text(doc.page_content)
        meta['id_source'] = _id
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"

            doc_chunks.append(doc.page_content)
            doc_id.append(_id + f"_{i}")
            _metadatas.append(meta)

    return doc_id, _metadatas, doc_chunks


def add_embedding(input_data):
    """Adds embedding to multiple documents from the database"""

    if (len(input_data['contents']) == len(input_data['ids'])) and (
            len(input_data['ids']) == len(input_data['metadatas'])):
        for i in range(len(input_data['contents'])):
            input_data['contents'][i] += f"\n||||\nMetadata Information: {str(input_data['metadatas'][i])}"

        input_data['ids'], input_data['metadatas'], input_data['contents'] = text_to_docs(
            input_data['ids'], input_data['metadatas'], input_data['contents']
        )
        input_data['embeddings'] = OpenAIEmbeddings().embed_documents(input_data['contents'])

        return input_data
    else:
        raise ValueError("Input mis-match!!!")


def add_data_to_vector_db(data_dict):
    """
    Receives data as input and adds that to the vector database

    Creates a vector database collection for each organization, and
    adds to existing organizations
    """

    try:
        data_dict = add_embedding(data_dict)

        owners = data_dict['org']
        collection = client.get_or_create_collection(name=owners)

        for i, (metadata, ids) in enumerate(zip(
                data_dict['metadatas'],
                data_dict['ids']
        )):
            metadata['source'] = ids

        collection.add(
            embeddings=data_dict['embeddings'],
            documents=data_dict['contents'],
            metadatas=data_dict['metadatas'],
            ids=[owners + i for i in map(str, data_dict['ids'])]
        )

        return {
            'status': 200,
            'log': "Successful!!!"
        }
    except Exception as e:
        return {
            'status': 403,
            'log': f"Failed with exception {e}"
        }


def get_vectorstore(org_name):
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        chunk_size=5000
    )

    try:
        client.get_collection(name=org_name)
    except:
        raise MissingOrganization

    return Chroma(client=client, collection_name=org_name, embedding_function=embedding_function)


def delete_organization(organization_name):
    """
    Deletion of organization from database
    """

    try:
        client.delete_collection(name=organization_name)

        return {
            'status': 200,
            'log': "Successful"
        }
    except Exception as e:
        return {
            'status': 403,
            'log': f"Failed with exception {e}"
        }


def modify_organization(prev_organization_name, new_organization_name):
    """
    Modification of organizations name
    """

    try:
        collection = client.get_collection(name=prev_organization_name)
        collection.modify(name=new_organization_name)
        client.delete_collection(name=prev_organization_name)

        return {
            'status': 200,
            'log': "Successful"
        }
    except Exception as e:
        return {
            'status': 403,
            'log': f"Failed with exception {e}"
        }


def get_emb_form_vector_db(owner, ids, return_only_emb=True):
    """Retrieve documents from vector database using their ids"""

    try:
        collection = client.get_or_create_collection(name=owner)

        if len(ids) > 1:
            result = collection.get(
                include=['embeddings', 'documents'],
                where={'$or': [{'id_source': _id} for _id in ids]}
            )
        else:
            result = collection.get(
                include=['embeddings', 'documents'],
                where={'id_source': ids[0]}
            )

        if return_only_emb:
            return result['embeddings']
        else:
            return result['documents'], result['embeddings']
    except Exception as e:
        return {
            'status': 403,
            'log': f"Failed with exception {e}"
        }
