generated_text_desc = "Generated text provided by the model"

TITLE_DESC = "Concise and meaningful title given to a generated text"
GENERATED_TEXT_DESC = "Generated text on the content topic/title"

focused_prompts = {
    "Technical document": """You are tasked with generating a technical document based on the CONTEXT provided. Ensure the document is structured and start by asserting the quality of the provided CONTEXT, and create a good technical document that aligns with the context. Your tecnical documents shoud be informative and accurate.""",

    "Release Note": """Generate a structured release note using the CONTEXT provided. The release note should:
1. Begin with an **Introduction**: Provide a summary of the release, including the version number and its significance.
2. Highlight **New Features**: List and describe newly added features, focusing on their benefits and use cases.
3. Detail **Improvements**: Summarize enhancements to existing features, emphasizing how they improve functionality or user experience.
4. Include **Bug Fixes**: Specify resolved issues, providing clear but concise descriptions.
5. Add **Known Issues (Optional)**: If applicable, note any remaining issues, including workarounds or expected fixes.
6. End with **Call to Action**: Encourage users to update, explore new features, or provide feedback.

The tone should be professional, clear, and aligned with the organization’s voice. Maintain a logical structure and ensure all information is accurate and concise.""",

    "Help Article": """Compose a user-focused help article using the CONTEXT provided. The article should:
1. Begin with a **Problem Statement**: Clearly define the issue or topic being addressed.
2. Provide **Step-by-Step Instructions**: Use simple, easy-to-follow language, breaking down complex processes into actionable steps.
3. Include **Tips or FAQs**: Add relevant additional information that may assist users in resolving related issues.
4. End with a **Call to Action or Next Steps**: Direct users toward further support or additional resources.
The tone should be conversational, accessible to a broad audience, and align with the organization’s guidelines for help articles.""",

    "FAQ": """Develop a comprehensive FAQ document using the CONTEXT provided. The document should:
- **Structure**: Present a list of frequently asked questions with concise, clear, and informative answers.
- **Tone**: Use a conversational and approachable tone while maintaining accuracy and professionalism.
- **Relevance**: Ensure questions address common concerns or pain points related to the CONTEXT, offering practical, actionable answers.
- **Extras**: If applicable, group related questions into categories to improve usability, and include links to additional resources for deeper understanding.""",

    "Marketing Copy": """Generate a compelling marketing copy based on the CONTEXT. The copy should:
1. Start with a **Strong Hook**: Grab the reader’s attention immediately.
2. Highlight **Unique Selling Points**: Showcase what sets the product, service, or idea apart from competitors.
3. Emphasize **Benefits**: Focus on how it addresses the audience's needs or solves a problem.
4. End with a **Call to Action (CTA)**: Prompt the reader to take a specific action, such as purchasing, signing up, or learning more.
The tone should be persuasive and align with the brand’s voice. Use engaging language and tailor the copy to resonate with the target audience.""",

    "Sales Pitch": """Create a persuasive sales pitch using the CONTEXT. The pitch should:
- **Introduction**: Start with a compelling statement or question to engage the audience.
- **Body**: Present the key benefits of the product or service, addressing the audience's specific pain points or needs. Differentiate it from competitors by emphasizing unique advantages.
- **Closing**: End with a strong, action-oriented call to action that encourages the audience to take the next step.
The language should be concise, persuasive, and tailored to the intended audience, reflecting the organization’s sales strategy.""",

    "User Guide": """Compose a detailed user guide using the CONTEXT provided. The guide should:
1. Start with an **Introduction**: Provide an overview of the product, feature, or process being explained, along with its purpose and relevance.
2. Include **Step-by-Step Instructions**: Break down the process into logical, clear steps. Use numbered lists or bullet points for clarity, and include visuals or diagrams where applicable.
3. Add **Troubleshooting Tips**: Offer solutions to common problems or challenges users might face.
4. End with a **Summary or Next Steps**: Recap key points and direct users to further resources if needed.
The guide should be user-friendly, comprehensive, and adhere to industry best practices for technical writing.""",

    "Custom": """Create a custom write-up using the CONTEXT provided. Analyze the information deeply to understand the organization’s goals, tone, and audience. Use the CONTEXT as the primary source of information, rephrasing it where necessary to produce content that is original, coherent, and tailored to the intended purpose. Ensure the output aligns with the specific needs and objectives outlined by the organization."""
}


include_context = lambda context, template=None, is_temp=False: f"""\n\nCONTEXT: ```{context}```{'' if template is None else "TEMPLATE: " + '```' + template + '```'}""" if not is_temp else ""

generation_prompt_template = lambda doc_type, tone, context, goal, template=None, is_temp=False: f"""{focused_prompts[doc_type]}

GOAL: ```{goal}```

With this goal generate a high-quality {doc_type}. Use the CONTEXT below as the foundation of your write-up, ensuring it reflects the intended purpose, audience, and style of the document. Your task is to understand the GOAL explicitly, whether it is specific, general, or a mix of both, and ensure that the generated content directly aligns with achieving that goal. 

**GOAL Analysis:** Carefully interpret the goal to determine the key priorities and expectations of the user. If the goal is specific, address the explicit requirements. If it is general or descriptive, infer and align with the implied intent, offering a comprehensive, context-aware solution.

Use a {tone} tone in your output and write the content in markdown syntax. Your content should:
- Avoid copying the CONTEXT verbatim; instead, synthesize and rephrase it into original, coherent, and well-structured language.
- Focus on improving reasoning, ensuring logical flow, clarity, and conciseness.
- Highlight the most critical details, prioritizing accuracy and relevance.
- Reference the CONTEXT and any templates provided (if applicable) to maintain consistency and alignment.
- Avoid duplicating contents, and keep your content unique all through.

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
