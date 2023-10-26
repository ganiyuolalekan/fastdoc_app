import json
import sqlite3

from pydantic import BaseModel, Field
from variables import title_desc, generated_text_desc


class GenerationModel(BaseModel):
    """Speaker names in the given transcript"""

    title: str = Field(..., description=title_desc)
    generated_text: str = Field(..., description=generated_text_desc)
