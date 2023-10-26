from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from prompts import conversation_prompt_template


title_desc = "Concise and meaningful title given to a generated text"
generated_text_desc = "Generated text provided by the model"

fastdoc_url = "https://fastdoc.io/"
base_url = "https://fastdoc-jira-integration.onrender.com/"

conversational_llm = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo-16k",
    max_tokens=5120
)

conversational_prompt = PromptTemplate(
    template=conversation_prompt_template,
    input_variables=["chat_history", "human_input", "context"]
)

SEPARATORS = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
