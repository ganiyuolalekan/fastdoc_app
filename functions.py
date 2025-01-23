import os
import re
import requests
import streamlit as st

from langchain_text_splitters import CharacterTextSplitter

from prompts import include_context

from utils import INPUT_TOKEN
from utils import dict_to_json, json_to_dict, create_contents, generate_text, process_response, regenerate_report

from dotenv import load_dotenv

load_dotenv()

TEST_LOCAL = eval(os.getenv('TEST_LOCAL', 'False'))

if TEST_LOCAL:
    # Local Development
    JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
else:
    # Production Development
    JIRA_BASE_URL = st.secrets["JIRA_BASE_URL"]


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


def clean_string(input_string):
    """Cleans a string from the extracted website text"""

    cleaned_string = re.sub(r"http\S+|www\S+|https\S+", "", input_string)
    cleaned_string = re.sub(r"\[\~accountid:[a-fA-F0-9]+\]", "", cleaned_string)

    return cleaned_string

def get_issues(issue_key):
    url = JIRA_BASE_URL + f'issues/{issue_key}'

    response = requests.request("POST", url, data=dict_to_json({}))

    return response.text


def write_out_report(issue):
    """Extracts an issues context from Jira as a report, using the issue key"""

    issue = json_to_dict(issue)['issue']['issues']
    issues = [issue] if type(issue) == dict else issue
    issue_key = issues[0]['key']
    withdraw_pattern = re.compile(r'with\s?drawn*|with\s?drew', re.IGNORECASE)

    try:
        if issues[0]['fields']['parent']['key'] == issue_key:
            issue_key_type = "Parent Issue"
        else:
            issue_key_type = "Child Issue"
    except KeyError:
        issue_key_type = "Parent Issue"

    content = f"Jira issue: (key: {issue_key})\n"
    for _fields in issues:
        fields = _fields['fields']
        
        try:
            ticket_type = fields['fields']['status']['name']
        except KeyError:
            ticket_type = fields['status']['name']

        if not withdraw_pattern.match(ticket_type):
            if fields['summary'] is not None:
                content += f"\n{issue_key} Issue Summary:\n\n{fields['summary']}\n\n"
            if fields['description'] is not None:
                content += f"\n{issue_key} Issue Description:\n\n{fields['description']}\n\n"

            comments = fields['comment']['comments']
            if len(comments):
                content += f"\n{issue_key} Comments:\n\n"
                for comment in comments:
                    content += f"{comment['body']}\n"

    return clean_string(content.strip()), issue_key_type


def remove_links(markdown_text):
    """Removes various types of links from markdown and text"""
    
    # Patterns to match different types of links
    plain_link_pattern = r'(https?|ftp|mailto|javascript):\/\/[\w\-\.]+(\.[a-z]{2,3})?([^\s]*)'
    markdown_link_pattern = r'\[([^\]]+)\]\((https?|ftp|mailto|javascript):\/\/[^\s]+\)'
    image_link_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    html_link_pattern = r'<a\s+href="[^"]+">([^<]+)<\/a>'
    html_image_pattern = r'<img\s+src="[^"]+"\s*\/?>'
    shortened_url_pattern = r'(https?|ftp|mailto):\/\/(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|buff\.ly|adf\.ly|ow\.ly)\/\S+'
    nested_markdown_link_pattern = r'\[!\[([^\]]*)\]\([^\)]+\)\]\((https?|ftp|mailto|javascript):\/\/[^\s]+\)'
    markdown_link_with_title_pattern = r'\[([^\]]+)\]\((https?|ftp|mailto|javascript):\/\/[^\s]+ "([^"]*)"\)'

    # Apply regex patterns to remove the links
    no_html_images = re.sub(html_image_pattern, '', markdown_text)
    no_html_links = re.sub(html_link_pattern, '', no_html_images)
    no_image_links = re.sub(image_link_pattern, '', no_html_links)
    no_nested_markdown_links = re.sub(nested_markdown_link_pattern, '', no_image_links)
    no_markdown_links_with_title = re.sub(markdown_link_with_title_pattern, '', no_nested_markdown_links)
    no_markdown_links = re.sub(markdown_link_pattern, '', no_markdown_links_with_title)
    no_shortened_links = re.sub(shortened_url_pattern, '', no_markdown_links)
    no_plain_links = re.sub(plain_link_pattern, '', no_shortened_links)
    
    return no_plain_links.strip()


def create_input(issue_keys, issues, goal, tone, doc_type, temperature, template=None):
    
    return {
        "goal": goal,
        "tone": tone,
        "scope": issue_keys,
        "issue": issues,
        "doc_type": doc_type,
        "temperature": temperature,
        "template": template,
        "url": "https://testai.atlassian.net/",
        "org": "https://testai.atlassian.net/",
    }


def create_issues(keys):
    content = []
    for key in keys:
        _content, _issue_type = write_out_report(get_issues(key))
        content.append({
            "content": _content,
            "issue_type": _issue_type
        })
    
    return content


def generate_content(doc_input, prompt=None, refine_prompt=None, approach="Ordered Issue Approach"):
    """Initializes a new project and creates a database instance"""
    
    content, _time = create_contents(doc_input["issue"], refine_prompt, approach)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=INPUT_TOKEN, chunk_overlap=0
    )
    content = text_splitter.split_text(content)[0]
    
    template = doc_input.get('template', None)
    temperature = doc_input.get('temperature')
    
    prompt += include_context(content, template)
    
    response, generation_time = generate_text(
        prompt,
        temperature=temperature
    )
    
    generation_time += _time
    
    result = process_response(response)

    title = result["title"]
    text = remove_links(result["generated_text"])

    return {
        "status": 200,
        "title": title,
        "generated_text": text,
        "generation_time": generation_time,
        "log": "Successfully generated report!!!",
    }


def re_generate_content(generated_response, temperature, user_query):
    re_gen_report, generation_time = regenerate_report(generated_response, temperature, user_query)

    return {
        "generated_text": remove_links(re_gen_report),
        "generation_time": generation_time,
        "status": 200,
        "log": "Successfully re-generated report!!!",
    }
