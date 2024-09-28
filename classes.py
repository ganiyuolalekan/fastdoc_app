from pydantic import BaseModel, Field

from prompts import GENERATED_TEXT_DESC, TITLE_DESC


class GenerationModel(BaseModel):
    """Speaker names in the given transcript"""

    title: str = Field(..., description=TITLE_DESC)
    generated_text: str = Field(..., description=GENERATED_TEXT_DESC)
