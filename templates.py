"""
Templates that can be used on the app
"""

technical_document = {
    "Software Development Technical Document": """## Introduction
[Purpose and expected outcome of the document]
## Objectives
[Goals the document aims to achieve]
## System Architecture
[Overview of the system architecture]
## Implementation Details
[Code structure, design patterns, and components]
## Testing
[Testing strategies, cases, and results]
## Appendices
[Include any additional information, glossaries, or reference materials.]"""
}

release_notes = {
    "Release Notes": """## Introduction
[Brief overview highlighting the key focus of the release.]
## What's New
[Detailed description of new features, improvements, or updates in this release.]
## Bug Fixes
[List of bugs addressed in this release with a brief description of each.]
## Known Issues
[Outline any known issues that are yet to be resolved.]
"""
}

help_articles = {
    "Help Articles": """## Introduction
[Brief introduction summarizing the purpose of the article and what it will cover.]
## Detailed Explanation
[Comprehensive explanation of the topic, process, or feature using clear and concise language.]
## Step-by-Step Guide
[Include a step-by-step guide with bullet points or numbered steps to make the process easy to follow.]
## Visual Aids
[Incorporate screenshots, diagrams, or videos where necessary to enhance understanding and provide visual guidance.]"""
}

faqs = {
    "FAQs": """## Introduction
[Provide a brief introduction that sets the context for the FAQs.]
## Frequently Asked Questions
    Q: [Question 1]
    A: [Answer 1]
    Q: [Question 2]
    A: [Answer 2]
    Q: [Question 3]
    A: [Answer 3]
(Continue with additional questions and answers)
"""
}

marketing_copy = {
    "Marketing Copy": """## Headline
[Create an engaging and attention-grabbing headline that clearly conveys the main benefit or unique value proposition of your product or service.]
## Introduction
[Briefly introduce the problem or need your product/service addresses.]
## Features and Benefits
[List the key features of your product/service and explain the benefits they provide to the customer.]
## Call to Action (CTA)
[Clearly state what you want the reader to do next (e.g., sign up, purchase, learn more) and make it easy for them to take that action.]"""
}

sales_pitch = {
    "Sales Pitch": """## Introduction
Start with a brief introduction of yourself and your company, aiming to connect with the prospect.
## The Problem
[Highlight a common problem faced by the prospect's industry or role.]
## Customer Success Stories
[Share stories of how your product/service has helped similar customers overcome similar challenges.]
## Call to Action
Invite the prospect to take a specific action, such as arranging a follow-up meeting or starting a free trial."""
}

user_guide = {
    "User Guide": """## Introduction
[Brief overview of the product and the user guide's purpose.]
## Quick Start Guide
[Essential initial setup steps to get the user up and running quickly.]
## Using the Product
[Detailed, step-by-step instructions on how to use the product effectively.]
## Additional Resources
[Links to further information, tutorials, or customer support for more detailed help]"""
}

DOCUMENT_TEMPLATES = {
    "Technical document": technical_document,
    "Release Note": release_notes,
    "Help Article": help_articles,
    "FAQ": faqs,
    "Marketing Copy": marketing_copy,
    "Sales Pitch": sales_pitch,
    "User Guide": user_guide,
    "Custom": None
}
