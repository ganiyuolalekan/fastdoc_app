generated_text_desc = "Generated text provided by the model"

TITLE_DESC = "Concise and meaningful title given to a generated text"
GENERATED_TEXT_DESC = "Generated text on the content topic/title"

doc_length_prompt = {
    'Short': "Keep the document concise capturing the major important details necessary for the generated content. Focus on critical child issues or summarize minor tasks.",
    'Medium': "You should focus on explaining the key parts of the documents and the less relevants part of the generated report should be keep brief/concise. Provide balanced detail. ",
    'Long': "Wrte a detailed and elaborate report about CONTEXT in-line with the guidelines. Ensure you elaborate each points, but also ensure your generated content are information fed to you from the provided CONTEXT."
}

prompt_template = {
    "Technical document": '''\n\n1. Introduction: Briefly state the document's purpose, scope, and target audience.
2. Core Content: Present the main information logically, using structured sections (e.g., concepts, processes, instructions).
3. Technical Details: Provide necessary specifications, data, code snippets, or diagrams to support the content.
4. Best Practices & Considerations: Highlight important guidelines, limitations, or optimizations.
5. Troubleshooting & FAQs (Optional): Address common issues, solutions, and clarifications.
6. References & Appendices (Optional): Include citations, external resources, or supplementary materials as needed.''',
    "Release Note": '''\n\n1. Introduction: Provide a brief summary of the release, including its significance. Do not include a version number unless explicitly provided.
2. What’s New: Highlight new features and enhancements, emphasizing their benefits and use cases.
3. Bug Fixes: List resolved issues, prioritizing high-severity bugs. If applicable, structure them under parent issues, followed by child issues in order of priority.
4. Known Issues (Optional): Include unresolved issues, workarounds, or expected fixes if relevant.''',
    "Help Article": '''\n\n1. Begin with a Problem Statement: Clearly define the issue or topic being addressed.
2. Provide Step-by-Step Instructions: Use simple, easy-to-follow language, breaking down complex processes into actionable steps.
3. Include Tips or FAQs: Add relevant additional information that may assist users in resolving related issues.
4. End with a Call to Action or Next Steps: Direct users toward further support or additional resources.
The tone should be conversational, accessible to a broad audience, and align with the organization’s guidelines for help articles.''',
    "FAQ": '''\n\n1. Structure: Present a list of frequently asked questions with concise, clear, and informative answers.
2. Tone: Use a conversational and approachable tone while maintaining accuracy and professionalism.
3. Relevance: Ensure questions address common concerns or pain points related to the CONTEXT, offering practical, actionable answers.
4. Extras: If applicable, group related questions into categories to improve usability, and include links to additional resources for deeper understanding.''',
    "Marketing Copy": '''\n\n1. Start with a Strong Hook: Grab the reader’s attention immediately.
2. Highlight Unique Selling Points: Showcase what sets the product, service, or idea apart from competitors.
3. Emphasize Benefits: Focus on how it addresses the audience's needs or solves a problem.
4. End with a Call to Action (CTA): Prompt the reader to take a specific action, such as purchasing, signing up, or learning more.
The tone should be persuasive and align with the brand’s voice. Use engaging language and tailor the copy to resonate with the target audience.''',
    "Sales Pitch": '''\n\n1. Introduction: Start with a compelling statement or question to engage the audience.
2. Body: Present the key benefits of the product or service, addressing the audience's specific pain points or needs. Differentiate it from competitors by emphasizing unique advantages.
3. Closing: End with a strong, action-oriented call to action that encourages the audience to take the next step.
The language should be concise, persuasive, and tailored to the intended audience, reflecting the organization’s sales strategy.''',
    "User Guide": '''\n\n1. Start with an Introduction: Provide an overview of the product, feature, or process being explained, along with its purpose and relevance.
2. Include Step-by-Step Instructions: Break down the process into logical, clear steps. Use numbered lists or bullet points for clarity, and include visuals or diagrams where applicable.
3. Add Troubleshooting Tips: Offer solutions to common problems or challenges users might face.
4. End with a Summary or Next Steps: Recap key points and direct users to further resources if needed.
The guide should be user-friendly, comprehensive, and adhere to industry best practices for technical writing.'''
}

key_guidelines_template = {
    "Technical document": """Key Guidelines:

- Clarity & Precision: Use clear, unambiguous language and define technical terms as needed.
- Logical Structure: Organize content with headings, bullet points, and a clear hierarchy.
- Accuracy & Conciseness: Provide correct, up-to-date information while keeping it brief and to the point.
- Audience Awareness: Tailor depth, tone, and complexity to the target readers.
- Use of Visuals: Incorporate diagrams, code snippets, or tables to simplify complex concepts.
- Consistency & Formatting: Maintain uniform terminology, structure, and adherence to style guidelines.
- Review & Versioning: Proofread, verify accuracy, and include version history for updates.""",
    "Release Note": """Key Guidelines:
    
- Maintain a concise, professional tone.
- Ensure accuracy and logical structuring of information.
- For short documents, focus on high-severity bugs. Medium/long documents may include additional details.
- Do not invent version numbers; only include them if explicitly provided.""",
    "Help Article": """Key Guidelines:
    
- Clear Purpose: Define the problem or question the article addresses upfront.
- Concise & Actionable: Use simple, direct language with step-by-step instructions.
- Logical Structure: Organize content with headings, bullet points, and numbered lists.
- Common Issues & Solutions: Anticipate user problems and provide troubleshooting tips.""",
    "FAQ": """Key Guidelines:

- Clear & Direct Questions: Phrase questions as users would ask them.
- Concise Answers: Provide straightforward, to-the-point responses.
- Logical Organization: Group related questions under relevant categories.
- Use Simple Language: Avoid jargon; keep explanations easy to understand.""",
    "Marketing Copy": """Key Guidelines:
    
- Know Your Audience: Tailor the message to their needs, interests, and pain points.
- Clear & Compelling Hook: Grab attention with a strong opening statement or headline.
- Concise & Persuasive: Use simple, impactful language with a focus on benefits.
- Strong Call-to-Action (CTA): Encourage immediate action with clear instructions.
- Emotional & Value-Driven: Appeal to emotions while highlighting unique selling points.
- Consistent Brand Voice: Align tone and style with the brand’s personality.""",
    "Sales Pitch": """Key Guidelines:
    
- Know Your Audience: Understand their needs, pain points, and goals.
- Strong Opening Hook: Capture attention with a compelling insight or question.
- Clear Value Proposition: Highlight the key benefits and unique advantages.
- Keep It Concise: Focus on essential points without unnecessary details.
- Engage & Personalize: Adapt your pitch to the specific client or situation.
- Compelling Call-to-Action (CTA): Guide the prospect toward the next step.""",
    "User Guide": """Key Guidelines:
    
- Clear Purpose: Define the guide’s objective and target audience.
- Step-by-Step Structure: Present instructions in a logical, sequential order.
- Simple & Precise Language: Use clear, concise wording to ensure easy understanding.
- Consistent Formatting: Use headings, bullet points, and numbering for readability."""
}

focused_prompts = {
    "Technical document": lambda use_template: f"""Generate a structured technical document using the CONTEXT provided. {prompt_template['Technical document'] if use_template else ''}
    
{key_guidelines_template['Technical document']}""",

    "Release Note": lambda use_template: f"""Generate a structured release note using the CONTEXT provided. {prompt_template['Release Note'] if use_template else ''}

{key_guidelines_template['Release Note']}""",

    "Help Article": lambda use_template: f"""Compose a user-focused help article using the CONTEXT provided. {prompt_template['Help Article'] if use_template else ''}
    
{key_guidelines_template['Help Article']}""",

    "FAQ": lambda use_template: f"""Develop a comprehensive FAQ document using the CONTEXT provided. {prompt_template['FAQ'] if use_template else ''}
    
{key_guidelines_template['FAQ']}""",

    "Marketing Copy": lambda use_template: f"""Generate a compelling marketing copy based on the CONTEXT. {prompt_template['Marketing Copy'] if use_template else ''}
    
{key_guidelines_template['Marketing Copy']}""",

    "Sales Pitch": lambda use_template: f"""Create a persuasive sales pitch using the CONTEXT. {prompt_template['Sales Pitch'] if use_template else ''}
    
{key_guidelines_template['Sales Pitch']}""",

    "User Guide": lambda use_template: f"""Compose a detailed user guide using the CONTEXT provided. {prompt_template['User Guide'] if use_template else ''}
    
{key_guidelines_template['User Guide']}""",

    "Custom": lambda use_template: f"""Create a custom write-up using the CONTEXT provided. Analyze the information deeply to understand the organization’s goals, tone, and audience. Use the CONTEXT as the primary source of information, rephrasing it where necessary to produce content that is original, coherent, and tailored to the intended purpose. Ensure the output aligns with the specific needs and objectives outlined by the organization."""
}


include_context = lambda context, template=None, is_temp=False: f"""\nCONTEXT: ```{context}```{'' if template is None else "TEMPLATE: " + '```' + template + '```'}""" if not is_temp else ""

generation_prompt_template = lambda doc_type, document_length, context, goal, template=None, is_temp=False: f"""{focused_prompts[doc_type](is_temp)}

GOAL: ```{goal}```

With this goal generate a high-quality {doc_type}. Use the CONTEXT below as the foundation of your write-up, ensuring it reflects the intended purpose, audience, and style of the document. Your task is to understand the GOAL explicitly, whether it is specific, general, or a mix of both, and ensure that the generated content directly aligns with achieving that goal. 

GOAL Analysis: Carefully interpret the goal to determine the key priorities and expectations of the user. If the goal is specific, address the explicit requirements. If it is general or descriptive, infer and align with the implied intent, offering a comprehensive, context-aware solution.

Note: {doc_length_prompt[document_length]}

Your content should:
- Avoid copying the CONTEXT verbatim; instead, synthesize and rephrase it into original, coherent, and well-structured language.
- Focus on improving reasoning, ensuring logical flow, clarity, and conciseness.
- Highlight the most critical details, prioritizing accuracy and relevance.
- Reference the CONTEXT and any templates provided (if applicable) to maintain consistency and alignment.
- Avoid duplicating contents, and keep your content unique all through.
- Avoid applying comments from the context directly in your generated text.
- Do not include a version in the generated document except it is part of the context passed.
- Ensure the headings and seb-headings of headings are properly formatted within the markdown.
- When generating the {doc_type}, only use the context as your source of information, do not attempt to fomulate ideas outside that context.
{include_context(context, template, is_temp)}"""

document_refine_prompt = lambda context, goal, doc_type, template=None, is_temp=False: f"""Given the context below, your goal is to refine the context to properly highlight the important components/points mentioned that aligns to the goal "{goal}". Note that this refined context should be detailed enough to be used in the generation of a "{doc_type}" document. Thus, provide every necessary details in generating that document. {include_context(context, template, is_temp)}"""

refactor_prompt = lambda human_input: f"""You are a text modification/improvement bot. Given a text as input, your role is to re-write an improved version of the text template based on the human suggestion and what you understand from your chat history. You're not to summarise the text but add intuitive parts to it or exclude irrelevant parts from it. Answer the human suggestion by modifying the text ONLY, maintaining the paragraphs and point from the input text.
You're not to add any comment of affrimation to you text, just answer the question by rewriting the text only. Always ensure your generated text follows a markdown syntax. Do not include links of any kind in your generated report.

User Query: ```{human_input}```"""

section_prompt = lambda context, tone, goal, title, generation_type: f"""You're writing a specific portion (section) of a document, the section is titled {title}. Thus, compose a 80 - 120 word write-up using a {tone} tone on the title "{title}". You're generating this section as part of an organization {generation_type}, and you're to write this section towards this goal {goal}. Your write-up should work entirely from the context provided below. Ensure you do not include hyperlinks in your modified content.

Note: Please go straight to the point and write out the content, do not begin with the title or topic, and you can buttress on you points with paragraphs.

Context
-------
{context}"""
