import os
from dotenv import load_dotenv

load_dotenv()

TEST_LOCAL = eval(os.getenv('TEST_LOCAL', 'False'))

if not TEST_LOCAL:
    __import__('pysqlite3')
    import sys

    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

from templates import DOCUMENT_TEMPLATES
from functions import app_meta, divider, write_out_report, create_input, generate_content, re_generate_content, remove_markdown
from prompts import generation_prompt_template, document_refine_prompt


app_meta()


with st.sidebar:
    st.write("FastDoc Document Generation")
    start_project = st.checkbox(
        label="Start Application",
        help="Starts The Demo Application"
    )
    divider()


def generate_response():
    with st.sidebar:
        st.markdown("### 1. Input to generate text")
        scope = st.text_input(
            label="Enter Issue key - eg FD-5, FD-12", value="SPX-21, SPX-20, SPX-19, SPX-18, SPX-17, SPX-16, SPX-13, SPX-12, SPX-11, SPX-10, SPX-7, SPX-6",
            help="You can enter multiple issue keys, separate them by commas (,)"
        )
        doc_type = st.selectbox(
            label="Enter preferred document type",
            options=[
                "Technical document", "Release Note",
                "Help Article", "FAQ", "Marketing Copy",
                "Sales Pitch", "User Guide"
            ], index=0
        )
        document_length = st.selectbox(
            label="How long should the document be?",
            options=[
                "Medium", "Short", "Long"
            ], index=0
        )
        goal = st.text_area(
            label="What's the goal of this document?",
            max_chars=100,
            value=""
        )
        
        use_template = st.checkbox(
            label="Do you want to use the template here",
            value=False
        )

        submit = st.button(label="generate")
        divider()
    
    return scope, doc_type, document_length, goal, use_template, submit


if start_project:
    scope, doc_type, document_length, goal, use_template, submit = generate_response()
    
    keys = [s.strip() for s in scope.split(',')]
    issues = write_out_report(keys)
    
    with st.expander("Input Context to the Model"):
        st.write(issues)
    
    with st.expander("Custom Prompt"):
        prompt = st.text_area(
            label="You can update the prompt here to test",
            value=generation_prompt_template(
                doc_type, 
                document_length, 
                scope, 
                goal, 
                is_temp=use_template
            ),
            height=400
        )
    
    template = list(DOCUMENT_TEMPLATES[
        doc_type
    ].values())[0] if use_template else None
    
    if template is not None:
        with st.expander("Define Template"):
            st.text_area(
                label="Define a template for the model",
                value=template,
                height=400
            )
    
    if submit:
        keys = [s.strip() for s in scope.split(',')]
        issues = write_out_report(keys)
        response = generate_content(create_input(
            keys, 
            issues, 
            goal, 
            document_length, 
            doc_type,
            template
        ), prompt)
        last_generated_text = response['generated_text']
        st.write("Generated in ", round(response['generation_time'], 2), "secs")
        divider()
        st.write(last_generated_text)
