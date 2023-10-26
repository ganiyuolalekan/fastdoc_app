TEST_LOCAL = False

if not TEST_LOCAL:
    __import__('pysqlite3')
    import sys

    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from functions import app_meta, divider
from utils import init_project, return_project_value, delete_project, json_to_dict, dict_to_json

app_meta()

with st.sidebar:
    st.write("FastDoc Document Generation")
    start_project = st.checkbox(
        label="Start Application",
        help="Starts The Demo Application"
    )
    divider()

if start_project:
    display_generated = display_comment = display_delete = False

    with st.sidebar:
        st.markdown("### 1. Input to generate text")
        scope = st.text_input(label="Enter Issue key - eg FD-5", value="FD-5")
        temperature = st.selectbox(
            label="Select a temperature for the model",
            options=['stable', 'variable', 'highly variable'], index=1
        )
        doc_type = st.selectbox(
            label="Enter preferred document type",
            options=[
                "Technical document", "Release Note",
                "Help Article", "FAQ", "Marketing Copy",
                "Sales Pitch", "User Guide", "Custom"
            ], index=0
        )
        tone = st.selectbox(
            label="What tone should your document have",
            options=[
                "Professional", "Friendly", "Direct",
                "Informal", "Authoritative", "Persuasive",
                "Empathetic", "Formal"
            ], index=0
        )
        goal = st.text_area(
            label="What's the goal of this document?",
            max_chars=100,
            value="Develop a non-technical documentation for the product release"
        )

        submit = st.button(label="Submit")
        divider()

    if submit:
        generate_text_res = init_project(dict_to_json({
            'goal': goal if goal != "" else None,
            'tone': tone,
            'scope': scope,
            'doc_type': doc_type,
            'temperature': temperature,
            'project_id': 12345,
        }))

        if generate_text_res is not None:
            display_generated = True

    if display_generated:
        divider()
        st.markdown("## Here's the generated text".upper())
        divider()
        title = json_to_dict(generate_text_res)['title']
        text = json_to_dict(generate_text_res)['generated_text']
        st.write(f"TITLE: \n\n{title}")
        divider()
        st.write(f"GENERATED TEXT: \n\n{text}")
        divider()

    with st.sidebar:
        st.markdown("### 2. Comment to improve text".upper())
        comment = st.text_input(label="What would you like to change?", value="Could you make the tone more like a newsletter")
        submit_2 = st.button(label="Submit", key=11)
        divider()

    if submit_2:
        st.write("Generating write up...")
        suggest_text_res = return_project_value(dict_to_json({
            'project_id': 12345,
            'user_query': comment
        }))

        if suggest_text_res is not None:
            display_comment = True

    if display_comment:
        divider()
        st.write(json_to_dict(suggest_text_res)['re-generated_text'])
        divider()

    with st.sidebar:
        st.markdown("### 3. Delete from database".upper())
        delete = st.button(label="Delete", key=22)

    if delete:
        input_delete = delete_project(dict_to_json({
            'project_id': 12345
        }))

        if input_delete is not None:
            display_delete = True

    if display_delete:
        st.write(delete_project(input_delete))
else:
    with open('README.md', 'r') as f:
        demo_report = f.read()
    st.markdown(demo_report)
