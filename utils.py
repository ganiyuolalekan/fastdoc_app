import os
import json
import time
import openai
import functools
import streamlit as st

from classes import GenerationModel

from prompts import generated_text_desc, include_context, refactor_prompt

from langchain.docstore.document import Document

from langsmith import traceable


TEST_LOCAL = eval(os.getenv('TEST_LOCAL', 'False'))

if TEST_LOCAL:
    # Local Development
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
else:
    # Production Development
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


INPUT_TOKEN = 10240
OUTPUT_TOKEN = 16384


def json_to_dict(json_str):
    """Converts a JSON string to a Python dictionary."""

    return json.loads(json_str)


def dict_to_json(dict_data):
    """Converts a Python dictionary to a JSON string."""

    return json.dumps(dict_data)


def time_function(func):
    """Function to time process during execution"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        return result, execution_time

    return wrapper


def ordered_issue(issues):
    """
    This approach orders the ticket and uses 
    the ordered tickets in content generation
    """
    
    docs = []
    for issue in issues:
        _content, _key_type = issue['content'], issue['issue_type']
        doc = Document(
            page_content=_content,
            metadata={
                # 'issue_key': scope,
                'issue_type': _key_type
            }
        )
        
        docs.append(doc)
    
    parent_content = []
    child_content = []
    for doc in docs:
        if doc.metadata['issue_type'] == "Parent Issue":
            parent_content.append(doc.page_content)
        else:
            child_content.append(doc.page_content)
    
    return "\n\n".join(parent_content + child_content)


def refine_input(issues, prompt):
    """
    Generate a context based output to the model
    """
    
    prompt += include_context(ordered_issue(issues))
    
    response, generation_time = refine_context(prompt)
    
    return response, generation_time


@traceable(run_type="retriever")
def create_contents(issues, prompt=None, approach="Ordered Issue Approach"):
    """Using the scopes, we can generate the needed contents"""
    
    if approach == "Ordered Issue Approach":
        return ordered_issue(issues), 0
    elif approach == "Refine Input Approach":
        return refine_input(issues, prompt)


@traceable(run_type="parser")
def process_response(response):
    """Processes the asynchronous response"""
    
    # memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    # conv_chain = load_qa_chain(llm=conversational_llm, chain_type="stuff", memory=memory, prompt=conversational_prompt)
    
    result = eval(response.choices[0].message.function_call.arguments)
    
    return result


@time_function
@traceable(run_type="llm")
def generate_text(prompt, temperature='variable'):
    """Function to generate report"""

    temp = {
        'stable': 0.,
        'variable': .5,
        'highly variable': 1.0
    }

    generation_custom_functions = [
        {
            'name': 'text_generation',
            'description': generated_text_desc,
            'parameters': GenerationModel.schema()
        }
    ]
        
    response = openai.chat.completions.create(
        temperature=temp[temperature],
        model='gpt-4o-mini',
        max_tokens=OUTPUT_TOKEN,
        messages=[{
            'role': 'user',
            'content': prompt
        }],
        functions=generation_custom_functions,
        function_call={"name": "text_generation"}
    )

    return response


@time_function
def refine_context(prompt):
    """Function to generate report"""
        
    response = openai.chat.completions.create(
        temperature=0.1,
        model='gpt-4o-mini',
        max_tokens=OUTPUT_TOKEN,
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )

    return response.choices[0].message.content


@time_function
@traceable(run_type="llm")
def regenerate_report(generated_report, temperature, user_query):
    """Regenerates the results based on the users request"""
    
    temp = {
        'stable': 0.,
        'variable': .5,
        'highly variable': 1.
    }
        
    response = openai.chat.completions.create(
        temperature=temp[temperature],
        model='gpt-4o-mini',
        max_tokens=OUTPUT_TOKEN,
        messages=[
            {
                'role': 'user',
                'content': refactor_prompt(user_query)
            },
            {
                'role': "user",
                'content': generated_report
            }
        ],
        prediction={
            'type': "content",
            'content': generated_report
        },
    )

    return response.choices[0].message.content
