import os
import openai
import streamlit as st

from prompts import generation_prompt_template
from classes import GenerationModel
from variables import conversational_llm, conversational_prompt, fastdoc_url, generated_text_desc
from functions import (
    convert_report, dict_to_json, get_relevant_doc_from_vector_db,
    json_to_dict, read_memory, write_out_report, write_memory
)

from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain


TEST_LOCAL = False

if TEST_LOCAL:
    # Local Development
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
else:
    # Production Development
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

openai.api_key = OPENAI_API_KEY

db = {}


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


def load(project_id):
    """Loads a FastDoc object from a json object"""

    try:
        return json_to_dict(db[project_id])
    except TypeError:
        return dict_to_json({
            'status': 403,
            'log': "Failed to load project!!!"
        })


def generate_text(project_id, text_content, tone, doc_type, url, goal=None, temperature='variable'):
    """Function to generate report"""

    temp = {
        'stable': 0.,
        'variable': 1.,
        'highly variable': 1.9
    }

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    conv_chain = load_qa_chain(llm=conversational_llm, chain_type="stuff", memory=memory, prompt=conversational_prompt)

    org_info = get_relevant_doc_from_vector_db(goal, url)

    generation_custom_functions = [
        {
            'name': 'text_generation',
            'description': generated_text_desc,
            'parameters': GenerationModel.schema()
        }
    ]

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

        try:
            url = keys['url']
        except KeyError:
            url = fastdoc_url

        goal = None if keys['goal'] == "" else keys['goal']

        result = generate_text(
            keys['project_id'],
            content,
            keys['tone'],
            keys['doc_type'],
            url,
            goal,
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

    return dict_to_json({
        'status': 200,
        'log': "Successfully deleted projects!!!"
    })
