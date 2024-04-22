import asyncio

import streamlit as st

from modules.variables import fastdoc_url
from modules.functions import (
    dict_to_json, exceptions_handler, generate_text, 
    regenerate_report, write_to_s3, json_to_dict, 
    write_out_report, generate_text_section_based
)


# @exceptions_handler
def init_project(json_input):
    """
    Initializes a new project and creates in id for it in our local database
    """

    # try:
    keys = json_to_dict(json_input)
    
    scopes = [scope.strip() for scope in keys['scope'].split(',')]
    
    content = "\n\n".join([f"\n{'-'*50}\n".join([scope, write_out_report(scope)[0]]) for scope in scopes])

    try:
        temp = keys['temperature']
    except KeyError:
        temp = 'variable'

    try:
        url = keys['url']
    except KeyError:
        url = fastdoc_url

    try:
        org = keys['org']
    except KeyError:
        org = "fastdoc"

    goal = None if keys['goal'] == "" else keys['goal']

    result = generate_text(
        keys['project_id'],
        content,
        keys['tone'],
        keys['doc_type'],
        url,
        org,
        goal,
        temperature=temp,
        template=keys['template']
    )

    title = result['title']
    text = result['generated_text']

    # st.write(write_to_s3(dict_to_json(result), f"{org}/{keys['doc_type']}/{title}.json"))

    return dict_to_json({
        'status': 200,
        'title': title,
        'generated_text': text,
        'log': "Successfully generated report!!!"
    })
    # except Exception as e:
    #     return dict_to_json({
    #         'status': 503,
    #         'log': f"Program failed with exception {e}"
    #     })


@exceptions_handler
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
