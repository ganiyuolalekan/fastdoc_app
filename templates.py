"""
Enhanced Templates with Optional and Compulsory Headings
"""

technical_document = {
    "Software Development Technical Document": """## Introduction
[Purpose and expected outcome of the document]
(Optional)
## Objectives
[Goals the document aims to achieve]
(Optional)
## System Architecture
[Overview of the system architecture, including diagrams if applicable]
(Optional)
## Implementation Details
[Code structure, design patterns, components, and algorithms used]
(Optional)
## Testing
[Testing strategies, test cases, tools used, and results]
(Optional)
## Best Practices
[Recommended practices or standards relevant to the project]
(Optional)
## Appendices
[Include additional resources such as glossaries, references, or related materials]
(Optional)"""
}

release_notes = {
    "Release Notes": """## Introduction
[Brief overview highlighting the key focus of the release and its significance.]
(Compulsory)
## What's New
[Detailed description of new features, improvements, or updates in this release.]
(Compulsory)
## Bug Fixes
[List of bugs addressed in this release with a brief description of each.]
(Compulsory)
## Known Issues
[Outline any known issues that are yet to be resolved.]
(Optional)
## Upgrade Instructions
[Provide a concise guide on how users can upgrade to the latest release.]
(Optional)"""
}

help_articles = {
    "Help Articles": """## Introduction
[Brief introduction summarizing the purpose of the article and what it will cover.]
(Compulsory)
## Detailed Explanation
[Comprehensive explanation of the topic, process, or feature using clear and concise language.]
(Compulsory)
## Step-by-Step Guide
[Include a step-by-step guide with numbered or bulleted steps to simplify the process.]
(Compulsory)
## Visual Aids
[Incorporate screenshots, diagrams, or videos to enhance understanding.]
(Optional)
## Troubleshooting Tips
[Provide advice for resolving common issues related to the topic.]
(Optional)"""
}

faqs = {
    "FAQs": """## Introduction
[Brief introduction providing context for the FAQ section.]
(Optional)
## Frequently Asked Questions
    Q: [Question 1]
    A: [Answer 1]
(Compulsory)
    Q: [Question 2]
    A: [Answer 2]
(Compulsory)
    Q: [Question 3]
    A: [Answer 3]
(Continue with additional questions and answers as needed)
## Additional Resources
[Links or references to related content, guides, or support.]
(Optional)"""
}

marketing_copy = {
    "Marketing Copy": """## Headline
[Engaging and attention-grabbing headline clearly conveying the main benefit or unique value proposition.]
(Compulsory)
## Introduction
[Briefly introduce the problem or need your product/service addresses.]
(Compulsory)
## Features and Benefits
[List the key features of your product/service and explain the benefits they provide to the customer.]
(Compulsory)
## Testimonials
[Include quotes or reviews from satisfied customers.]
(Optional)
## Call to Action (CTA)
[Clearly state the next step for the reader (e.g., sign up, purchase, learn more).]
(Compulsory)"""
}

sales_pitch = {
    "Sales Pitch": """## Introduction
[Briefly introduce yourself and your company while connecting with the prospect.]
(Compulsory)
## The Problem
[Highlight a common problem faced by the prospect's industry or role.]
(Compulsory)
## Solution Overview
[Describe how your product or service solves the problem.]
(Compulsory)
## Customer Success Stories
[Share stories of how your product/service helped similar customers overcome challenges.]
(Optional)
## Closing Statement
[Wrap up the pitch with a compelling statement to reinforce value.]
(Compulsory)
## Call to Action
[Encourage the prospect to take the next step (e.g., schedule a demo, start a trial).]
(Compulsory)"""
}

user_guide = {
    "User Guide": """## Introduction
[Overview of the product and the user guide's purpose.]
(Compulsory)
## Quick Start Guide
[Essential initial setup steps to get users up and running quickly.]
(Compulsory)
## Using the Product
[Detailed, step-by-step instructions on how to use the product effectively.]
(Compulsory)
## Tips and Tricks
[Additional suggestions or shortcuts for optimal use.]
(Optional)
## Troubleshooting
[Common issues users might face and how to resolve them.]
(Optional)
## Additional Resources
[Links to tutorials, customer support, or other related materials.]
(Optional)"""
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
