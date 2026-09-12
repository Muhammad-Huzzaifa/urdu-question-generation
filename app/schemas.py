from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class GenerationResponse(BaseModel):
    source: str
    greedy: str
    beam: str