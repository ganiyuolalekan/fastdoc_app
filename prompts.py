generated_text_desc = "Generated text provided by the model"

TITLE_DESC = "Concise and meaningful title given to a generated text"
GENERATED_TEXT_DESC = "Generated text on the content topic/title"

focused_prompts = {
    "Technical document": """Generate a technical document using the CONTEXT provided. Ensure the document includes a clear introduction, a detailed body with headings and subheadings, and a conclusion. Incorporate technical details, jargon, and diagrams where relevant. The document should be concise yet comprehensive, offering in-depth explanations of technical aspects. Your write-up should reflect the organization's objectives and adhere to industry standards for technical documentation.""",
    "Release Note": """Create a release note based on the CONTEXT. The release note should start with a summary of the new version, followed by a structured list of new features, improvements, and bug fixes. Keep the language clear and concise. Tailor the note to reflect the organization's brand voice and ensure it aligns with the standard format for release notes in the industry.""",
    "Help Article": """Compose a help article using the CONTEXT provided. The article should address a specific issue or topic, providing clear and easy-to-follow step-by-step guidance. Use a conversational tone suitable for a broad audience. Include a problem statement, solution steps, and FAQs if applicable. The article should align with the organization's style and the standard format for help articles.""",
    "FAQ": """Develop an FAQ section based on the CONTEXT. Structure the document as a series of common questions followed by concise, clear answers. The tone should be conversational and accessible. Ensure the FAQ is relevant to the target audience and reflects the organization's voice and goals.""",
    "Marketing Copy": """Write marketing copy using the CONTEXT. The copy should be engaging, persuasive, and align with the brand's voice. Highlight key benefits, features, and what sets the product or service apart. Include a strong call to action. Ensure the copy is tailored to the target audience and meets industry standards for effective marketing content.""",
    "Sales Pitch": """Craft a sales pitch with the provided CONTEXT. Start with a compelling introduction, then detail the benefits of the product or service, and how it differs from competitors. Conclude with a persuasive call to action. The pitch should be concise, focused on value proposition, and tailored to the target audience, reflecting the organization's sales strategy and industry norms.""",
    "User Guide": """Generate a user guide based on the CONTEXT. The guide should include an introductory overview, detailed step-by-step instructions, and troubleshooting tips. Incorporate visuals or diagrams where necessary for clarity. Ensure the guide is comprehensive, easy to understand, and aligns with the organization's standards and the typical format of user guides in the industry.""",
    "Custom": """You’re an AI system perfect for text generation, as such perfectly understand and study the CONTEXT below making reference to the ORGANIZATION INFORMATION below to provide more information/idea/context about the organization requesting the text generation. The CONTEXT provided below should act as the MAJOR source of your write up, organization information is intended to be used to help you understand what the organization is about so you generate you text towards their intended goal."""
}

include_context = lambda context, template, is_temp=False: f"""\n\nCONTEXT: ```{context}```{'' if template is None else "TEMPLATE: " + '```' + template + '```'}""" if not is_temp else ""

generation_prompt_template = lambda doc_type, tone, context, goal, template=None, is_temp=False: f"""{focused_prompts[doc_type]}

Use this information to write/compose a {doc_type} write-up with a descriptive title and its content (the generated text). Also, ensure it is detailed enough and does not include “accountid” information from the CONTEXT below. Use a {tone} tone in your generated output. You're to write towards addressing this goal "{goal}".

Always ensure your generated text follows a markdown syntax. Do not copy the text from the context as is, instead only make reference to it while your generated text are in isr own words.{include_context(context, template, is_temp)}"""
