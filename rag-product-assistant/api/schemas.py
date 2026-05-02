from pydantic import BaseModel, Field
from typing import Optional


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The product question to answer")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier for tracing")


class SourceDocument(BaseModel):
    content: str
    product_name: Optional[str] = None
    source: Optional[str] = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocument]
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    vector_db_backend: str
    documents_indexed: int
