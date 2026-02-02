"""State and schema definitions for the survey agent."""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Optional, Sequence

from langchain_core.messages import BaseMessage, MessageLikeRepresentation
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


def override_reducer(current_value, new_value):
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


class AgentInputState(MessagesState):
    """State representing the input messages to the survey agent."""


class AgentState(MessagesState):
    """Main state for the survey agent pipeline."""

    research_brief: Optional["ResearchQuestion"]
    supervisor_messages: Annotated[list[BaseMessage], add_messages] = []
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    search_batches: Annotated[list[dict], override_reducer] = []
    supervisor_state: Annotated[dict, operator.or_] = Field(default_factory=dict)
    rag_stats: Annotated[dict, operator.or_] = Field(default_factory=dict)
    final_report: Optional[str] = None


class SupervisorState(TypedDict, total=False):
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: dict
    keywords: list[str]
    prisma_stats: dict
    search_run_ids: list[str]
    deduped_papers: list[dict]
    pdf_queue: list[dict]
    raw_notes: Annotated[list[str], operator.add]
    notes: Annotated[list[str], operator.add]


class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""

    research_topic: str = Field(
        description="A focused research topic described in detail.",
    )


class ResearchComplete(BaseModel):
    """Call this tool to indicate the research phase is complete."""


class ResearcherState(TypedDict):
    """State for individual researchers conducting RAG-based research."""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer]


class ResearcherOutputState(BaseModel):
    """Output state from a researcher subgraph."""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []


class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )


class ResearchQuestion(BaseModel):
    """Schema for structured query plan generation."""

    scope_notes: "ScopeNotes"
    filters: "Filters"
    boolean_query_generic: str
    queries: "Queries"
    must_include: list[str] = Field(default_factory=list)
    must_exclude: list[str] = Field(default_factory=list)


class SearchBatch(BaseModel):
    """Search result batch with its query context."""

    source: str
    user_query: Optional[str] = None
    generated_query: Optional[str] = None
    filters: dict = Field(default_factory=dict)
    papers: list[dict] = Field(default_factory=list)


class PrismaStats(BaseModel):
    retrieved_total: int = 0
    retrieved_by_source: dict[str, int] = Field(default_factory=dict)
    deduplicated_total: int = 0
    duplicates_removed: int = 0


class ScopeNotes(BaseModel):
    user_constraints: list[str] = Field(default_factory=list)
    open_considerations: list[str] = Field(default_factory=list)


class YearRange(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: Optional[int] = Field(default=None, alias="from")
    to: Optional[int] = None


class Filters(BaseModel):
    year_range: YearRange = Field(default_factory=YearRange)
    language: Optional[str] = None
    doc_types: Optional[list[str]] = None


class ArxivQuery(BaseModel):
    search_query: str
    notes: list[str] = Field(default_factory=list)


class AcmQuery(BaseModel):
    query: str
    notes: list[str] = Field(default_factory=list)


class Queries(BaseModel):
    arxiv: ArxivQuery
    acm: AcmQuery


class Paper(BaseModel):
    """Schema for collected research papers."""

    source: str
    source_id: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    pdf_url: str | None = None


class BaseSearchRequest(BaseModel):
    """Base schema for search requests."""

    query: str = Field(description="The search query string.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=200,
        description="The maximum number of search results to return.",
    )
    date_from: Optional[datetime] = Field(
        default=None,
        description="The start date for filtering search results.",
    )
    date_to: Optional[datetime] = Field(
        default=None,
        description="The end date for filtering search results.",
    )


class RagIngestionStats(BaseModel):
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    indexed: int = 0
    errors: list[str] = Field(default_factory=list)


ResearchQuestion.model_rebuild()


__all__ = [
    "AgentInputState",
    "AgentState",
    "SupervisorState",
    "ClarifyWithUser",
    "ResearchQuestion",
    "SearchBatch",
    "PrismaStats",
    "ScopeNotes",
    "YearRange",
    "Filters",
    "ArxivQuery",
    "AcmQuery",
    "Queries",
    "Paper",
    "BaseSearchRequest",
    "RagIngestionStats",
    "ConductResearch",
    "ResearchComplete",
    "ResearcherState",
    "ResearcherOutputState",
    "MessageLikeRepresentation",
]
