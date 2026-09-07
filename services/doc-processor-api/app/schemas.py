from __future__ import annotations

from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"


class TableContent(BaseModel):
    headers: list[str]
    rows: list[list[str]]


class Element(BaseModel):
    # Field name kept as chunk_id for compatibility with retrieval-api schema.
    # Semantically: element_id == chunk_id in Phase 1.
    chunk_id: str
    document_id: str
    page: list[int]  # 1-based
    section: str | None = None
    content_type: ContentType
    content: Union[str, TableContent]
    bbox: list[float] | None = None  # normalized [x0,y0,x1,y1]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessResponse(BaseModel):
    document_id: str
    num_pages: int
    elements: list[Element]
