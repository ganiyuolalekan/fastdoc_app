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
from functions import app_meta, divider, create_issues, create_input, generate_content
from prompts import generation_prompt_template, document_refine_prompt


app_meta()


with st.sidebar:
    st.write("FastDoc Document Generation")
    start_project = st.checkbox(
        label="Start Application",
        help="Starts The Demo Application"
    )
    divider()


if start_project:
    with st.sidebar:
        st.markdown("### 1. Input to generate text")
        approach = st.selectbox(
            label="Enter the approach type you'd like to use",
            options=["Ordered Issue Approach", "Refine Input Approach"], index=0
        )
        scope = st.text_input(
            label="Enter Issue key - eg FD-5, FD-12", value="FD-5",
            help="You can enter multiple issue keys, separate them by commas (,)"
        )
        temperature = st.selectbox(
            label="Select a temperature for the model",
            options=['stable', 'variable', 'highly variable'], index=1
        )
        doc_type = st.selectbox(
            label="Enter preferred document type",
            options=[
                "Technical document", "Release Note",
                "Help Article", "FAQ", "Marketing Copy",
                "Sales Pitch", "User Guide"
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
            value="Transfer knowledge from subject matter experts to other team members or stakeholders."
        )
        
        use_template = st.checkbox(
            label="Do you want to use the template here",
            value=False
        )

        submit = st.button(label="generate")
        divider()
    
    template = list(DOCUMENT_TEMPLATES[doc_type].values())[0] if use_template else None
    
    with st.expander("Custom Prompt"):
            prompt = st.text_area(
                label="You can update the prompt here to test",
                value=generation_prompt_template(doc_type, tone, scope, goal, is_temp=True),
                height=400
            )
    
    if template is not None:
        with st.expander("Define Template"):
            st.text_area(
                label="Define a template for the model",
                value=template,
                height=400
            )
    
    if approach == "Ordered Issue Approach":
        if submit:
            keys = [s.strip() for s in scope.split(',')]
            issues = create_issues(keys)
            response = generate_content(create_input(keys, issues, goal, tone, doc_type, temperature, template), prompt)
            st.write("Title: ", response['title'])
            st.write("Generated in ", round(response['generation_time'], 2), "secs")
            divider()
            st.write(response['generated_text'])
    elif approach == "Refine Input Approach":
        with st.expander("Refine Prompt"):
            refine_prompt = st.text_area(
                label="You can update the prompt here to test",
                value=document_refine_prompt("", goal, doc_type, is_temp=True),
                height=400
            )
        if submit:
            keys = [s.strip() for s in scope.split(',')]
            issues = create_issues(keys)
            response = generate_content(create_input(keys, issues, goal, tone, doc_type, temperature, template), prompt, refine_prompt, approach)
            st.write("Title: ", response['title'])
            st.write("Generated in ", round(response['generation_time'], 2), "secs")
            divider()
            st.write(response['generated_text'])
    elif approach == "Approach 2":
        pass
    else:
        pass
