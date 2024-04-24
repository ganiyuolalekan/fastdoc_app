"""
Templates that can be used on the app
"""

EXAMPLE_TEMPLATE = """# [Document Title]

## Introduction
[A brief introduction of the concept...]

## Objectives
[The objective of the write-up...]

...
OTHER OUTLINES FROM THE SOURCE DOCUMENT WITH DEFINATE HEADING TITLES
[INSTRUCTIONS TO FOLLOW FOR OTHER OUTLINES]
...

## Conclusion
[Conclusion of the topic...]"""

TECHNICAL_DOCUMENT = """# [Document Title]

## Introduction
[Briefly describe the purpose and expected outcome of the document.]

## Audience
[Identify the primary readers of the document, e.g., developers, stakeholders.]

## Table of Contents
[Automatically generated.]

## Executive Summary
[Provide a concise summary of the document's key points and findings.]

## Objectives
[Clearly state the goals the document aims to achieve.]

## Scope
[Define the boundaries and limitations of the document's content.]

## Terminology
[Explain any technical terms, acronyms, or jargon used.]

## Process Overview
[Detail the development process, project plans, and standards relevant to the document.]

## Findings/Results
[Present main findings or results, supported by data or analysis.]

## Product Information
[Include user guides, technical specifications, and other relevant product-based information.]

## Recommendations
[Offer actionable recommendations based on the findings.]

## Conclusion
[Summarize the document, reinforcing the key outcomes and implications.]

## Maintenance and Updates
[Outline the protocol for regular document updates and maintenance.]

## Testing and Feedback
[Describe the process for testing the document for usability and incorporating user feedback.]

## Appendices
[Include any additional data, charts, or reference material.]

## References
[List any sources or references used in the document.]"""

RELEASE_NOTE = """# [Release Title]

## Introduction
[A brief overview highlighting the key focus of the release.]

## What's New
[Detailed description of new features, improvements, or updates in this release.]

## Enhancements
[Briefly describe any enhancements made to existing features.]

## Bug Fixes
[List of bugs addressed in this release with a brief description of each.]

## Known Issues
[Outline any known issues that are yet to be resolved.]

## Impact and Actions Required
[Explain any changes users need to make or be aware of due to this release.]

## Acknowledgements
[Credit team members or contributors who played a significant role in this release.]

## Additional Resources
[Links to relevant guides, FAQs, or support resources.]

## Conclusion
[A closing statement summarizing the release's importance or future outlook.]

## [Version Number]
[The version number associated with the release.]"""


HELP_ARTICLE = """# [Article Title]

## Introduction
[Begin with a brief introduction that summarizes the purpose of the article and what it will cover]

## Detailed Explanation
[Provide a comprehensive explanation of the topic, process, or feature. Use clear and concise language to ensure understanding]

###Step-by-Step Guide
[If applicable, include a step-by-step guide with bullet points or numbered steps to make the process easy to follow]

## Visual Aids
[Incorporate screenshots, diagrams, or videos where necessary to enhance understanding and provide visual guidance]

## FAQs
[Include a section for frequently asked questions related to the article's topic]

## Troubleshooting
[Offer troubleshooting tips or common issues and their solutions]

## Additional Resources
[Provide links to related articles, external resources, or further reading]

## Conclusion
[Summarize the key points of the article and offer next steps or calls to action]

## Feedback and Contact Information - Optional
[Encourage readers to provide feedback on the article and provide contact information for further assistance]"""

FAQ = """## Introduction
[Provide a brief introduction that sets the context for the FAQs]

## Frequently Asked Questions

- **Q:** [Question 1]
  - **A:** [Answer 1]

- **Q:** [Question 2]
  - **A:** [Answer 2]

- **Q:** [Question 3]
  - **A:** [Answer 3]

- (Continue with additional questions and answers)

## Additional Resources
[Include links to more detailed resources, guides, or related topics]

## Contact Information
[Provide information for further inquiries or where to get more help]

## Conclusion
[A closing remark that encourages users to explore more or get in touch for additional support]"""

MARKETING_COPY = """## Headline
[Create an engaging and attention-grabbing headline that clearly conveys the main benefit or unique value proposition of your product or service.]

## Introduction
[Briefly introduce the problem or need your product/service addresses.]

## Features and Benefits
[List the key features of your product/service and explain the benefits they provide to the customer.]

## Testimonials or Social Proof
[Include quotes or endorsements from satisfied customers to build trust and credibility.]

## Call to Action (CTA)
[Clearly state what you want the reader to do next (e.g., sign up, purchase, learn more) and make it easy for them to take that action.]

## Contact Information
[Provide ways for the customer to get in touch with you, such as a phone number, email address, or link to a contact form.]

## Conclusion
[Wrap up with a compelling closing statement that reinforces the value proposition and encourages action.]"""

SALES_PITCH = """## Introduction
[Introduce yourself and your company briefly, focusing on establishing a connection.]

## The Problem
[Start with a statement or question about the problem you solve, tailored to the prospect's industry and needs.]

## Value Proposition
[Clearly articulate the value your product/service offers, addressing how it solves the identified problem.]

## Product/Service Details
[Explain how your product/service works and its unique features.]

## Proof Points
[Provide evidence such as customer testimonials, case studies, or data that validates the effectiveness of your solution.]

## Customer Success Stories
[Share stories of how your product/service has helped similar customers.]

## Call to Action
[Encourage the prospect to take the next step, whether it's a meeting, a demo, or a trial offer.]

## Contact Information
[Provide your contact details for further communication.]"""

USER_GUIDE = """# [Product Name: User Guide]

## Introduction
[Provide an overview of the product and the purpose of the guide.]

## Identifying Users
[Define the target audience of the guide.]

## Table of Contents
[An organized list of topics covered in the guide.]

## Getting Started
- [Initial setup instructions or basic operations.]

## Detailed Instructions
[Step-by-step instructions on using the product, with sequential steps.]

## Visuals and Diagrams
[Include images, diagrams, or videos for clarification.]

## Problem-Solving and Troubleshooting
[Address common problems and their solutions.]

## FAQs
[Answers to frequently asked questions about the product.]

## Additional Resources
[Links to further information, tutorials, or customer support.]

## Feedback Section
[Encourage users to provide feedback on the guide for improvements.]

## Conclusion
[Summarize the key points and thank the user for using the product.]

## Contact Information
[Provide contact details for further support.]"""

DOCUMENT_TEMPLATES = {
    "Technical document": TECHNICAL_DOCUMENT,
    "Release Note": RELEASE_NOTE,
    "Help Article": HELP_ARTICLE,
    "FAQ": FAQ,
    "Marketing Copy": MARKETING_COPY,
    "Sales Pitch": SALES_PITCH,
    "User Guide": USER_GUIDE,
    "Custom": None
}
