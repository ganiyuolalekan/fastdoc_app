from .templates import TECHNICAL_DOCUMENT

focused_prompts = {
    "Technical document": """Generate a technical document using the CONTEXT and ORGANIZATION INFORMATION provided. Ensure the document includes a clear introduction, a detailed body with headings and subheadings, and a conclusion. Incorporate technical details, jargon, and diagrams where relevant. The document should be concise yet comprehensive, offering in-depth explanations of technical aspects. Your write-up should reflect the organization's objectives and adhere to industry standards for technical documentation.""",
    "Release Note": """Create a release note based on the CONTEXT and ORGANIZATION INFORMATION. The release note should start with a summary of the new version, followed by a structured list of new features, improvements, and bug fixes. Keep the language clear and concise. Tailor the note to reflect the organization's brand voice and ensure it aligns with the standard format for release notes in the industry.""",
    "Help Article": """Compose a help article using the CONTEXT and ORGANIZATION INFORMATION provided. The article should address a specific issue or topic, pxroviding clear and easy-to-follow step-by-step guidance. Use a conversational tone suitable for a broad audience. Include a problem statement, solution steps, and FAQs if applicable. The article should align with the organization's style and the standard format for help articles.""",
    "FAQ": """Develop an FAQ section based on the CONTEXT and ORGANIZATION INFORMATION. Structure the document as a series of common questions followed by concise, clear answers. The tone should be conversational and accessible. Ensure the FAQ is relevant to the target audience and reflects the organization's voice and goals.""",
    "Marketing Copy": """Write marketing copy using the CONTEXT and ORGANIZATION INFORMATION. The copy should be engaging, persuasive, and align with the brand's voice. Highlight key benefits, features, and what sets the product or service apart. Include a strong call to action. Ensure the copy is tailored to the target audience and meets industry standards for effective marketing content.""",
    "Sales Pitch": """Craft a sales pitch with the provided CONTEXT and ORGANIZATION INFORMATION. Start with a compelling introduction, then detail the benefits of the product or service, and how it differs from competitors. Conclude with a persuasive call to action. The pitch should be concise, focused on value proposition, and tailored to the target audience, reflecting the organization's sales strategy and industry norms.""",
    "User Guide": """Generate a user guide based on the CONTEXT and ORGANIZATION INFORMATION. The guide should include an introductory overview, detailed step-by-step instructions, and troubleshooting tips. Incorporate visuals or diagrams where necessary for clarity. Ensure the guide is comprehensive, easy to understand, and aligns with the organization's standards and the typical format of user guides in the industry.""",
    "Custom": """You’re an AI system perfect for text generation, as such perfectly understand and study the CONTEXT below making reference to the ORGANIZATION INFORMATION below to provide more context about the organization requesting the text generation. The CONTEXT provided below should act as the MAJOR source of your write up, organization information is intended to be used to help you understand what the organization is about so you generate you text towards their intended goal."""
}

generation_prompt_template = lambda doc_type, tone, context, org_info, template=TECHNICAL_DOCUMENT, goal=None: f"""{focused_prompts[doc_type]}
Make use of this information to write/compose a {doc_type} write-up with a descriptive title and its content (the generated text). Ensure the text you generated is only in the format described below for the {doc_type}. Use the information about the organization to improve/fine-tune your generated text. Also, ensure it is detailed enough and does not include “accountid” information from the context below. Use a {tone} tone in your generated output. You're to write towards addressing this goal "{goal}", if the provided goal is None, then generate your text only in context to {doc_type} format, using the context below to gain scope/context on your write-up.
 Always ensure your generated text follows a markdown syntax. Never copy text from the context or use it to fill points in your generated text only when necessary. Finally, ensure your generated text never exceeds 3072 tokens.
 
 Note: you're free to skip sections of the defined templates that you do not have answers/context to, while maintaining the order properly labeled, and in the conclusion of the document, you can offer an explanation of the the missing sections.

TEMPLATE: {template}
CONTEXT: {context}
ORGANIZATION INFORMATION: {org_info}"""

template_generation_prompt = lambda document, example_template=TECHNICAL_DOCUMENT: f"""Given the document below, you're expected to understand and extract the template used in this document. A template is simply the headings and sub-headings - with their purpose - that can form the skeletal structure of any newly formed document. Take a look at the example template below.

NOTE: For your template headings and sub-headings, please ensure you ignore the topics of the headings and sub-headings from the document, only get the general idea of the heading, for example, "Overview of System Authentication" should be interpreted as "Overview" in your template, "Benefits of MFA Implementation" should be interpreted as "Benefits" in your template and so on.

DOCUMENT: {document}
EXAMPLE TEMPLATE: {example_template}"""

conversation_prompt_template = """You are a text modification/improvement bot. Given a text as input, your role is to re-write an improved version of the text template based on the human question and what you understand from your chat history. You're not to summarise the text but add intuitive parts to it or exclude irrelevant parts from it. Answer the human questions by modifying the text ONLY, maintaining the paragraphs and point from the input text.
You're not to add any comment of affrimation to you text, just answer the question by rewriting the text only. Always ensure your generated text follows a markdown syntax.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""
