import openai

from pydantic import BaseModel, Field

from .variables import OPENAI_API_KEY
from .variables import title_desc, generated_text_desc

openai.api_key = OPENAI_API_KEY


class GenerationModel(BaseModel):
    """Speaker names in the given transcript"""

    title: str = Field(..., description=title_desc)
    generated_text: str = Field(..., description=generated_text_desc)
