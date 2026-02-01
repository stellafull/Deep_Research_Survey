import operator
from datetime import datetime
from typing import Optional, Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ConfigDict

# ========= State Definitions

class AgentInputState(MessagesState):
    """State representing the input messages to the survey agent."""
    pass

class AgentState(MessagesState):
    """main state for the survey agent, holding conversation messages."""

    # structured query plan generated from the conversation
    research_brief: Optional["ResearchQuestion"]
    # Messages exchanged with the supervisor agent for coordination
    supervisor_messages: Annotated[list[BaseMessage], add_messages] = []
    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[list[str], operator.add] = []
    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add] = []
    # Collected paper metadata from search tools
    papers: Annotated[list[dict], operator.add] = []
    # Final formatted research report
    final_report: Optional[str] = None


# ========= structured Output

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
    """Schema for collected research papers"""
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
    query: str = Field(
        description="The search query string.",
    )
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


ResearchQuestion.model_rebuild()
