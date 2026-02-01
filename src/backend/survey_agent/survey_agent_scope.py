"""User Clarification and Research Brief Generation.

This module implements the scoping phase of the research workflow, where we:
1. Assess if the user's request needs clarification
2. Generate a detailed research brief from the conversation

The workflow uses structured output to make deterministic decisions about
whether sufficient context exists to proceed with research.
"""

import json
from datetime import datetime
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from survey_agent.prompts import clarify_with_user_instructions, transform_messages_into_boolean_query_prompt
from survey_agent.state_scope import AgentInputState, AgentState, ClarifyWithUser, ResearchQuestion
from survey_agent.utils import get_today_str, search_arxiv_papers, search_acm_papers

# ========== load env variables
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

# initialize model
model = init_chat_model(    
    model='Pro/deepseek-ai/DeepSeek-R1',
    model_provider='openai',
    base_url=os.getenv('LLM_BASE_URL'),
    api_key=os.getenv('LLM_API_KEY'),
    temperature=0,
    )

# ========== Workflow Nodes

def clarify_with_user(state: AgentState)-> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """
    # setup structured output
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # invoke
    response = structured_output_model.invoke([
        HumanMessage(
            content=clarify_with_user_instructions.format(
                messages=get_buffer_string(messages=state['messages']),
                date=get_today_str()
            )
        )
    ])


    # Always generate a research brief to build source queries.
    # If clarification is needed, add the question but continue.
    if response.need_clarification:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.question)]},
        )

    return Command(
        goto="write_research_brief",
        update={"messages": [AIMessage(content=response.verification)]},
    )
    

def write_research_brief(state: AgentState):
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # Generate research brief from conversation history
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_boolean_query_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str(),
            sources=json.dumps(["arxiv", "acm"])
        ))
    ])

    # Update state with generated research brief and pass it to the supervisor
    return {
        "research_brief": response.model_dump(by_alias=True),
        "supervisor_messages": [HumanMessage(content=response.boolean_query_generic)]
    }


def _year_range_to_dates(year_range: dict | None) -> tuple[datetime | None, datetime | None]:
    if not year_range:
        return None, None
    year_from = year_range.get("from")
    year_to = year_range.get("to")
    start = datetime(year_from, 1, 1) if year_from else None
    end = datetime(year_to, 12, 31, 23, 59) if year_to else None
    return start, end


def run_arxiv_search(state: AgentState):
    brief = state.get("research_brief") or {}
    arxiv_query = brief.get("queries", {}).get("arxiv", {}).get("search_query")
    if not arxiv_query:
        return {"papers": []}

    date_from, date_to = _year_range_to_dates(
        brief.get("filters", {}).get("year_range")
    )
    results = search_arxiv_papers.invoke(
        {
            "query": arxiv_query,
            "max_results": 200,
            "date_from": date_from,
            "date_to": date_to,
        }
    )
    return {"papers": results}


def run_acm_search(state: AgentState):
    brief = state.get("research_brief") or {}
    acm_query = brief.get("queries", {}).get("acm", {}).get("query")
    if not acm_query:
        return {"papers": []}

    date_from, date_to = _year_range_to_dates(
        brief.get("filters", {}).get("year_range")
    )
    results = search_acm_papers.invoke(
        {
            "query": acm_query,
            "max_results": 200,
            "date_from": date_from,
            "date_to": date_to,
        }
    )
    return {"papers": results}


# ===== GRAPH CONSTRUCTION =====

# Build the scoping workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("search_arxiv_papers", run_arxiv_search)
deep_researcher_builder.add_node("search_acm_papers", run_acm_search)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "search_arxiv_papers")
deep_researcher_builder.add_edge("write_research_brief", "search_acm_papers")
deep_researcher_builder.add_edge("search_arxiv_papers", END)
deep_researcher_builder.add_edge("search_acm_papers", END)

# Compile the workflow
scope_research = deep_researcher_builder.compile()
