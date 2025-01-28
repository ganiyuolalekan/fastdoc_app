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
from functions import app_meta, divider, create_issues, create_input, generate_content, re_generate_content
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

        submit = st.checkbox(label="generate", value=False)
        divider()
    
    return approach, scope, temperature, doc_type, tone, goal, use_template, submit


def re_generate_response(count):
    with st.sidebar:
        st.markdown(f"### Input to re-generate text [Count: {count}]")
        user_query = st.text_area(
            label=f"What adjustment will you like to make for you number {count} re-generation?",
            max_chars=200,
            value=""
        )
        submit = st.checkbox(label=f"re-generate-{count}", value=False)
    
    return user_query, submit


if start_project:
    approach, scope, temperature, doc_type, tone, goal, use_template, submit = generate_response()
    
    with st.expander("Custom Prompt"):
        prompt = st.text_area(
            label="You can update the prompt here to test",
            value=generation_prompt_template(
                doc_type, 
                tone, 
                scope, 
                goal, 
                is_temp=True
            ),
            height=400
        )
    
    # if 'generation_report_state' not in st.session_state:
    #     st.session_state['generation_report_state'] = {}
    # if 'last_generated_text' not in st.session_state:
    #     st.session_state['last_generated_text'] = ""
    
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
    
    generation_tab, re_generation_tab_1, re_generation_tab_2, re_generation_tab_3 = st.tabs([
        "Initial Generation", 
        "First Regeneration", 
        "Second Regeneration", 
        "Thrid Regeneration"
    ])
    
    if approach == "Ordered Issue Approach":
        if submit:
            keys = [s.strip() for s in scope.split(',')]
            issues = create_issues(keys)
            response = generate_content(create_input(
                keys, 
                issues, 
                goal, 
                tone, 
                doc_type, 
                temperature, 
                template
            ), prompt)
            # st.session_state['generation_report_state']["Initial Generation"] = response
            with generation_tab:
                # st.session_state['last_generated_text'] = response['generated_text']
                last_generated_text = response['generated_text']
                st.write("Generated in ", round(response['generation_time'], 2), "secs")
                divider()
                # st.write(st.session_state['last_generated_text'])
                st.write(last_generated_text)
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
            # st.session_state['generation_report_state']["Initial Generation"] = response
            with generation_tab:
                # st.session_state['last_generated_text'] = response['generated_text']
                last_generated_text = response['generated_text']
                st.write("Generated in ", round(response['generation_time'], 2), "secs")
                divider()
                # st.write(st.session_state['generation_report_state']["Initial Generation"]['generated_text'])
                # st.write(st.session_state['generation_report_state'])
                st.write(last_generated_text)
    
    if submit:
        for re_generation_count, tab in enumerate([
            re_generation_tab_1, 
            re_generation_tab_2, 
            re_generation_tab_3
        ], start=1):
            user_query, re_generate_submit = re_generate_response(count=re_generation_count)
            if re_generate_submit:
                re_response = re_generate_content(
                    # st.session_state['last_generated_text'], 
                    last_generated_text,
                    temperature, 
                    user_query
                )
                # st.session_state['generation_report_state'][f"Regeneration {re_generation_count}"] = re_response
                # st.session_state['last_generated_text'] = re_response['generated_text'] 
                last_generated_text = re_response['generated_text'] 
                with tab:
                    st.write("Generated in ", round(response['generation_time'], 2), "secs")
                    divider()
                    # st.write(st.session_state['generation_report_state'][f"Regeneration {re_generation_count}"]['generated_text'])
                    # st.write(st.session_state['generation_report_state'])
