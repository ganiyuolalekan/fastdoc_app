import os
import re
import requests
import streamlit as st

from requests.auth import HTTPBasicAuth
from langchain_text_splitters import CharacterTextSplitter

from prompts import include_context

from utils import INPUT_TOKEN
from utils import dict_to_json, generate_text, process_response, regenerate_report

from dotenv import load_dotenv

load_dotenv()

TEST_LOCAL = eval(os.getenv('TEST_LOCAL', 'False'))

if TEST_LOCAL:
    # Local Development
    JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
    USER_NAME = os.getenv("USER_NAME")
    JIRA_API_KEY = os.getenv("JIRA_API_KEY")
else:
    # Production Development
    JIRA_BASE_URL = st.secrets["JIRA_BASE_URL"]
    USER_NAME = st.secrets["USER_NAME"]
    JIRA_API_KEY = st.secrets["JIRA_API_KEY"]


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


def extract_plain_text(node):
    """
    Recursively extract plain text from an Atlassian document node.
    """
    text = ""
    if isinstance(node, dict):
        # If there's direct text, use it.
        if 'text' in node:
            text += node['text']
        # Otherwise, process any children.
        if 'content' in node:
            for child in node['content']:
                text += extract_plain_text(child)
    elif isinstance(node, list):
        for item in node:
            text += extract_plain_text(item)
    return text


def get_issue_details(issue_key, jira_base_url=JIRA_BASE_URL, username=USER_NAME, api_token=JIRA_API_KEY):
    """
    Retrieve specific details of an issue from Jira.
    
    The returned dictionary contains:
      1. issue_type: The type of issue (Epic, Bug, etc.)
      2. description: Plain text description extracted from the rich-text structure.
      3. summary: The issue summary.
      4. comments: A text block of comments formatted as "<Name>: <comment>" for each comment.
      5. severity: The issue severity (if provided; adjust custom field as needed).
      6. status: The current status (e.g., In Progress, Done, etc.).
      7. priority: The priority level (defaults to "Low" if not provided).
    
    Parameters:
      - jira_base_url (str): Base URL of your Jira instance (e.g., 'https://your-domain.atlassian.net').
      - issue_key (str): The issue key (e.g., 'EPIC-123' or 'TASK-456').
      - username (str): Your Jira username or email address.
      - api_token (str): Your Jira API token.
    
    Returns:
      - dict: A dictionary containing the issue details.
    """
    # Build the API URL
    url = f"{jira_base_url}/rest/api/3/issue/{issue_key}"
    auth = HTTPBasicAuth(username, api_token)
    headers = {"Accept": "application/json"}
    
    response = requests.get(url, auth=auth, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve issue details: {response.status_code}")
        return None
    
    issue_data = response.json()
    fields = issue_data.get('fields', {})

    # Get issue type
    issue_type = fields.get('issuetype', {}).get('name', 'Unknown')
    
    # Extract description as plain text (if available)
    raw_description = fields.get('description', 'No description available')
    if isinstance(raw_description, (dict, list)):
        description = extract_plain_text(raw_description)
    else:
        description = raw_description
    
    # Get summary
    summary = fields.get('summary', 'No summary available')
    
    # Process comments: extract plain text and remove links/images.
    comment_entries = fields.get('comment', {}).get('comments', [])
    comments_texts = []
    for comment in comment_entries:
        author = comment.get('author', {}).get('displayName', 'Unknown User')
        raw_body = comment.get('body', '')
        if isinstance(raw_body, (dict, list)):
            body_text = extract_plain_text(raw_body)
        else:
            body_text = str(raw_body)
        # Remove any URLs, markdown links or images if still present
        body_text = re.sub(r'http\S+', '', body_text)
        body_text = re.sub(r'!\[.*?\]\(.*?\)', '', body_text)
        body_text = re.sub(r'\[.*?\]\(.*?\)', '', body_text)
        comments_texts.append(f"{author}: {body_text.strip()}")
    # Combine all comments into a single text block
    comments = "\n".join(comments_texts)
    
    # Get severity from a custom field (update 'customfield_12345' to your field's key)
    severity = fields.get('customfield_12345', 'Not provided')
    
    # Get status as the label (e.g., In Progress, Done)
    status = fields.get('status', {}).get('name', 'Unknown')
    
    # Get priority (default to "Low" if not provided)
    priority = fields.get('priority', {}).get('name', 'Low')
    
    return {
        "issue_type": issue_type,
        "description": description,
        "summary": summary,
        "comments": comments,
        "severity": severity,
        "status": status,
        "priority": priority
    }


def generate_issue_report(issues):
    """
    Given an array of issue dictionaries, sort them by:
      1. Issue type: Epics first, then Child, then Story, etc.
      2. Priority: High > Medium > Low.
      3. Status: Done > In Review > In Progress > To Do.
    
    Then return a formatted string output for the issues.
    
    Each issue dictionary is expected to have the following keys:
      - issue_key (optional): The unique identifier.
      - issue_type: e.g., "Epic", "Child", "Story", etc.
      - summary: A brief summary.
      - description: The description text.
      - comments: Either a list of strings or a newline-delimited string.
      - status: The status of the issue.
      - priority: The priority level.
    """
    # Define the sort order for issue types, priority, and status.
    type_order = {
        "Epic": 0,
        "Child": 1,
        "Story": 2,
        "Bug": 3,
        "Task": 4,
        "Sub-task": 5
    }
    priority_order = {
        "High": 0,
        "Medium": 1,
        "Low": 2
    }
    status_order = {
        "Done": 0,
        "In Review": 1,
        "In Progress": 2,
        "To Do": 3
    }
    
    def sort_key(issue):
        # Use default high numbers if the field isn't found.
        return (
            type_order.get(issue.get("issue_type", ""), 99),
            priority_order.get(issue.get("priority", "Low"), 99),
            status_order.get(issue.get("status", ""), 99)
        )
    
    # Sort the issues according to the defined sort key.
    sorted_issues = sorted(issues, key=sort_key)
    
    report_lines = []
    for issue in sorted_issues:
        # Use issue_key if provided; otherwise, fallback to summary.
        key = issue.get("issue_key", issue.get("summary", "N/A"))
        report_lines.append(f"Issue: {key} ({issue.get('issue_type','N/A')})\n")
        report_lines.append(f"  Summary: {issue.get('summary','')}\n")
        report_lines.append(f"  Description: {issue.get('description','')}\n")
        report_lines.append(f"  Status: {issue.get('status','')}\n")
        report_lines.append(f"  Priority: {issue.get('priority','')}\n")
        report_lines.append("  Comments:\n")
        
        # Handle comments: if it's a string, split by newline; if it's a list, iterate directly.
        comments = issue.get("comments", "")
        if isinstance(comments, str):
            comment_list = comments.split("\n")
        else:
            comment_list = comments
        
        for comment in comment_list:
            if comment.strip():
                report_lines.append(f"    - {comment.strip()}")
        report_lines.append("\n")
    
    return "\n\n".join(report_lines)


def write_out_report(issues):
    """Extracts an issues context from Jira as a report, using the issue key"""

    return clean_string(generate_issue_report([
        get_issue_details(issue)
        for issue in issues
    ]))


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


def create_input(issue_keys, issues, goal, document_length, doc_type, template=None):
    
    return {
        "goal": goal,
        "document_length": document_length,
        "scope": issue_keys,
        "issue": issues,
        "doc_type": doc_type,
        "template": template,
        "url": "https://testai.atlassian.net/",
        "org": "https://testai.atlassian.net/",
    }


def remove_markdown(text):
    """
    Removes markdown formatting from a given text while preserving newlines.
    """

    # Remove headers (e.g., # Header, ## Header, ### Header)
    text = re.sub(r'(?m)^\s*#{1,6}\s+', '', text)

    # Remove bold and italic (**text**, *text*, __text__, _text_)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)  # Bold
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)  # Italic

    # Remove inline code (`code`)
    text = re.sub(r'`(.+?)`', r'\1', text)

    # Remove code blocks (```code```)
    text = re.sub(r'```[\s\S]+?```', '', text)

    # Remove links but keep the text [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove images ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Remove blockquotes ("> text")
    text = re.sub(r'(?m)^\s*>+\s?', '', text)

    # Remove list markers but preserve newlines
    text = re.sub(r'(?m)^\s*[-*+]\s+', '', text)
    text = re.sub(r'(?m)^\s*\d+\.\s+', '', text)

    # Remove horizontal rules (---, ***)
    text = re.sub(r'(?m)^\s*(?:-|\*){3,}\s*$', '', text)

    # Remove strikethroughs (~~text~~)
    text = re.sub(r'~~(.*?)~~', r'\1', text)

    # Remove tables (| column | column |) but keep content
    text = re.sub(r'(?m)^\|(.+?)\|$', r'\1', text)

    # Normalize extra spaces but preserve single newlines
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    return text


def generate_content(doc_input, prompt=None, refine_prompt=None, approach="Ordered Issue Approach"):
    """Initializes a new project and creates a database instance"""
    
    # content, _time = create_contents(doc_input["issue"], refine_prompt, approach)
    content = doc_input["issue"]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=INPUT_TOKEN, chunk_overlap=0
    )
    content = text_splitter.split_text(content)[0]
    
    template = doc_input.get('template', None)
    
    prompt += include_context(content, template)
    
    response, generation_time = generate_text(prompt)
    
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


def re_generate_content(generated_response, user_query):
    re_gen_report, generation_time = regenerate_report(generated_response, user_query)

    return {
        "generated_text": remove_links(re_gen_report),
        "generation_time": generation_time,
        "status": 200,
        "log": "Successfully re-generated report!!!",
    }
