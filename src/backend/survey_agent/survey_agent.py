"""Survey agent graphs (scope + full pipeline)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from survey_agent.configuration import Configuration
from survey_agent.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_boolean_query_prompt,
)
from survey_agent.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    PrismaStats,
    ResearchComplete,
    ResearchQuestion,
    ResearcherOutputState,
    ResearcherState,
    SearchBatch,
    SupervisorState,
)
from survey_agent.storage import (
    canonical_id_for_paper,
    ingest_pdf_queue,
    persist_search_results,
)
from survey_agent.utils import (
    build_arxiv_query,
    get_today_str,
    rag_search,
    search_acm_papers,
    search_arxiv_papers,
    think_tool,
)


def _build_llm(cfg: Configuration, *, structured_output=None):
    model = init_chat_model(
        model=cfg.llm_model,
        model_provider=cfg.llm_provider,
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
    )
    if structured_output is not None:
        model = model.with_structured_output(structured_output)
    return model


def _with_retry(runnable, cfg: Configuration):
    return runnable.with_retry(stop_after_attempt=cfg.max_structured_output_retries)


def _bind_tools(model, tools):
    if hasattr(model, "bind_tools"):
        return model.bind_tools(tools)
    underlying = getattr(model, "runnable", None) or getattr(model, "_runnable", None)
    if underlying and hasattr(underlying, "bind_tools"):
        return underlying.bind_tools(tools)
    raise AttributeError("bind_tools not available on model")


# ===== Scope nodes

def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_research_brief", "__end__"]]:
    cfg = Configuration.from_runnable_config(config)
    if not cfg.allow_clarification:
        return Command(goto="write_research_brief")

    structured_output_model = _with_retry(
        _build_llm(cfg, structured_output=ClarifyWithUser), cfg
    )
    response = structured_output_model.invoke(
        [
            HumanMessage(
                content=clarify_with_user_instructions.format(
                    messages=get_buffer_string(messages=state["messages"]),
                    date=get_today_str(),
                )
            )
        ]
    )

    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]},
        )

    return Command(
        goto="write_research_brief",
        update={"messages": [AIMessage(content=response.verification)]},
    )


def write_research_brief(state: AgentState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    structured_output_model = _with_retry(
        _build_llm(cfg, structured_output=ResearchQuestion), cfg
    )
    response = structured_output_model.invoke(
        [
            HumanMessage(
                content=transform_messages_into_boolean_query_prompt.format(
                    messages=get_buffer_string(state.get("messages", [])),
                    date=get_today_str(),
                    sources=json.dumps(["arxiv", "acm"]),
                )
            )
        ]
    )

    brief = response.model_dump(by_alias=True)
    keywords = response.must_include or []
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=cfg.max_concurrent_research_units,
        max_researcher_iterations=cfg.max_researcher_iterations,
    )
    return {
        "research_brief": brief,
        "supervisor_messages": [
            SystemMessage(content=supervisor_system_prompt),
            HumanMessage(content=response.boolean_query_generic),
        ],
        "supervisor_state": {
            "keywords": keywords,
            "research_brief": brief,
        },
    }


def _year_range_to_dates(year_range: dict | None) -> tuple[datetime | None, datetime | None]:
    if not year_range:
        return None, None
    year_from = year_range.get("from")
    year_to = year_range.get("to")
    start = datetime(year_from, 1, 1) if year_from else None
    end = datetime(year_to, 12, 31, 23, 59) if year_to else None
    return start, end


def run_arxiv_search(state: AgentState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    brief = state.get("research_brief") or {}
    arxiv_query = brief.get("queries", {}).get("arxiv", {}).get("search_query")
    if not arxiv_query:
        return {"search_batches": []}

    date_from, date_to = _year_range_to_dates(brief.get("filters", {}).get("year_range"))
    full_query = build_arxiv_query(arxiv_query, date_from, date_to)
    results = search_arxiv_papers.invoke(
        {
            "query": arxiv_query,
            "max_results": cfg.arxiv_max_results,
            "date_from": date_from,
            "date_to": date_to,
        }
    )
    batch = SearchBatch(
        source="arxiv",
        user_query=arxiv_query,
        generated_query=full_query,
        filters={
            "year_range": brief.get("filters", {}).get("year_range"),
            "date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None,
            "max_results": cfg.arxiv_max_results,
        },
        papers=results,
    )
    return {"search_batches": [batch.model_dump()]}


def run_acm_search(state: AgentState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    brief = state.get("research_brief") or {}
    acm_query = brief.get("queries", {}).get("acm", {}).get("query")
    if not acm_query:
        return {"search_batches": []}

    date_from, date_to = _year_range_to_dates(brief.get("filters", {}).get("year_range"))
    results = search_acm_papers.invoke(
        {
            "query": acm_query,
            "max_results": cfg.acm_max_results,
            "date_from": date_from,
            "date_to": date_to,
        }
    )
    batch = SearchBatch(
        source="acm",
        user_query=acm_query,
        generated_query=acm_query,
        filters={
            "year_range": brief.get("filters", {}).get("year_range"),
            "date_from": date_from.date().isoformat() if date_from else None,
            "date_to": date_to.date().isoformat() if date_to else None,
            "max_results": cfg.acm_max_results,
        },
        papers=results,
    )
    return {"search_batches": [batch.model_dump()]}


def finalize_search_results(state: AgentState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    batches = state.get("search_batches", [])
    retrieved_by_source: dict[str, int] = {}
    all_papers: list[dict] = []
    search_run_ids: list[str] = []
    doc_ids_by_canonical: dict[str, str] = {}

    for batch in batches:
        source = batch.get("source") or "unknown"
        batch_papers = batch.get("papers") or []
        retrieved_by_source[source] = retrieved_by_source.get(source, 0) + len(batch_papers)
        all_papers.extend(batch_papers)

        persist_result = persist_search_results(
            cfg.db_path,
            source=source,
            user_query=batch.get("user_query"),
            generated_query=batch.get("generated_query"),
            filters=batch.get("filters"),
            papers=batch_papers,
        )
        if persist_result:
            search_run_ids.append(persist_result.run_id)
            doc_ids_by_canonical.update(persist_result.doc_ids_by_canonical)

    seen: set[str] = set()
    deduped: list[dict] = []
    for paper in all_papers:
        canonical_id = canonical_id_for_paper(paper)
        if canonical_id in seen:
            continue
        seen.add(canonical_id)
        enriched = dict(paper)
        enriched["canonical_id"] = canonical_id
        doc_id = doc_ids_by_canonical.get(canonical_id)
        if doc_id:
            enriched["doc_id"] = doc_id
        deduped.append(enriched)

    prisma_stats = PrismaStats(
        retrieved_total=len(all_papers),
        retrieved_by_source=retrieved_by_source,
        deduplicated_total=len(deduped),
        duplicates_removed=len(all_papers) - len(deduped),
    ).model_dump()

    pdf_queue = []
    for paper in deduped:
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            continue
        pdf_queue.append(
            {
                "doc_id": paper.get("doc_id"),
                "canonical_id": paper.get("canonical_id"),
                "pdf_url": pdf_url,
                "source": paper.get("source"),
                "source_id": paper.get("source_id"),
                "title": paper.get("title"),
                "url": paper.get("url"),
            }
        )

    return {
        "supervisor_state": {
            "prisma_stats": prisma_stats,
            "search_run_ids": search_run_ids,
            "deduped_papers": deduped,
            "pdf_queue": pdf_queue,
        }
    }


# ===== RAG ingestion node

def ingest_pdfs(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    pdf_queue = state.get("supervisor_state", {}).get("pdf_queue", [])
    if not pdf_queue:
        return Command(goto="research_supervisor", update={"rag_stats": {"indexed": 0}})

    rag_stats = ingest_pdf_queue(pdf_queue, config=config)
    return Command(goto="research_supervisor", update={"rag_stats": rag_stats})


# ===== Research supervisor + researcher subgraph

def _get_notes_from_tool_calls(messages):
    return [
        tool_msg.content
        for tool_msg in filter_messages(messages, include_types="tool")
        if getattr(tool_msg, "name", "") not in {"think_tool"}
    ]


def _default_subtopics(research_brief) -> list[str]:
    brief_text = ""
    if isinstance(research_brief, dict):
        brief_text = research_brief.get("boolean_query_generic") or json.dumps(
            research_brief, ensure_ascii=False
        )
    else:
        brief_text = str(research_brief or "")
    if not brief_text:
        brief_text = "the research brief"
    return [
        f"{brief_text} — definitions, background, and core concepts.",
        f"{brief_text} — key methods/models and representative approaches.",
        f"{brief_text} — datasets, evaluation metrics, and benchmarks.",
    ]


def _research_tools():
    return [rag_search, think_tool, ResearchComplete]


async def researcher(state: ResearcherState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    tools = _research_tools()
    researcher_prompt = research_system_prompt.format(date=get_today_str())

    research_model = _with_retry(_bind_tools(_build_llm(cfg), tools), cfg)
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        },
    )


async def researcher_tools(state: ResearcherState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    if not most_recent_message.tool_calls:
        # Force at least one RAG retrieval on the first turn.
        if state.get("tool_call_iterations", 0) == 0:
            query = state.get("research_topic") or get_buffer_string(researcher_messages)
            observation = await rag_search.ainvoke({"query": query}, config)
            tool_outputs = [
                ToolMessage(
                    content=observation,
                    name="rag_search",
                    tool_call_id="forced_rag_search",
                )
            ]
            return Command(goto="researcher", update={"researcher_messages": tool_outputs})
        return Command(goto="compress_research")

    tool_outputs: list[ToolMessage] = []
    research_complete_called = False

    for tool_call in most_recent_message.tool_calls:
        name = tool_call.get("name")
        if name == "think_tool":
            reflection = tool_call["args"].get("reflection")
            tool_outputs.append(
                ToolMessage(
                    content=f"Reflection recorded: {reflection}",
                    name="think_tool",
                    tool_call_id=tool_call["id"],
                )
            )
        elif name == "rag_search":
            observation = await rag_search.ainvoke(tool_call["args"], config)
            tool_outputs.append(
                ToolMessage(
                    content=observation,
                    name="rag_search",
                    tool_call_id=tool_call["id"],
                )
            )
        elif name == "ResearchComplete":
            research_complete_called = True
        else:
            tool_outputs.append(
                ToolMessage(
                    content=f"Unknown tool call: {name}",
                    name=name or "unknown",
                    tool_call_id=tool_call.get("id"),
                )
            )

    exceeded_iterations = state.get("tool_call_iterations", 0) >= cfg.max_react_tool_calls
    if research_complete_called and state.get("tool_call_iterations", 0) < 2:
        # Ensure at least two retrieval rounds for richer evidence when possible.
        query = state.get("research_topic") or get_buffer_string(researcher_messages)
        observation = await rag_search.ainvoke({"query": query}, config)
        tool_outputs.append(
            ToolMessage(
                content=observation,
                name="rag_search",
                tool_call_id="forced_rag_search_second",
            )
        )
        return Command(goto="researcher", update={"researcher_messages": tool_outputs})

    if exceeded_iterations or research_complete_called:
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs},
        )

    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs},
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    compression_prompt = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=compression_prompt)] + researcher_messages

    response = await _with_retry(_build_llm(cfg), cfg).ainvoke(messages)

    raw_notes_content = "\n".join(
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    )

    return {
        "compressed_research": str(response.content),
        "raw_notes": [raw_notes_content],
    }


researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState,
    config_schema=Configuration,
)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()


async def supervisor(state: SupervisorState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    research_model = _with_retry(
        _bind_tools(_build_llm(cfg), [ConductResearch, ResearchComplete, think_tool]),
        cfg,
    )

    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)

    most_recent_message = supervisor_messages[-1]
    exceeded_allowed_iterations = research_iterations > cfg.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls
    )

    if no_tool_calls and research_iterations == 0:
        # Fallback: force a single research run if the model didn't call tools.
        topics = _default_subtopics(state.get("research_brief"))
        topics = topics[: cfg.max_concurrent_research_units]
        research_tasks = [
            researcher_subgraph.ainvoke(
                {
                    "researcher_messages": [HumanMessage(content=topic)],
                    "research_topic": topic,
                },
                config,
            )
            for topic in topics
        ]
        tool_results = await asyncio.gather(*research_tasks)
        tool_msgs = []
        for idx, (observation, topic) in enumerate(zip(tool_results, topics), start=1):
            tool_msgs.append(
                ToolMessage(
                    content=observation.get(
                        "compressed_research",
                        "Error synthesizing research report: Maximum retries exceeded",
                    ),
                    name="ConductResearch",
                    tool_call_id=f"fallback_conduct_research_{idx}",
                )
            )
        raw_notes_concat = "\n".join(
            "\n".join(observation.get("raw_notes", [])) for observation in tool_results
        )
        update = {"supervisor_messages": tool_msgs}
        if raw_notes_concat:
            update["raw_notes"] = [raw_notes_concat]
        return Command(goto="supervisor", update=update)

    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": _get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
            },
        )

    all_tool_messages: list[ToolMessage] = []
    update_payload: dict = {"supervisor_messages": []}

    think_tool_calls = [
        tool_call
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(
            ToolMessage(
                content=f"Reflection recorded: {reflection_content}",
                name="think_tool",
                tool_call_id=tool_call["id"],
            )
        )

    conduct_research_calls = [
        tool_call
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        allowed_calls = conduct_research_calls[: cfg.max_concurrent_research_units]
        overflow_calls = conduct_research_calls[cfg.max_concurrent_research_units :]

        # If the model only asked for one research task on the first iteration,
        # auto-add extra subtopics to encourage map-reduce.
        if research_iterations == 0 and len(allowed_calls) < 3:
            topics = _default_subtopics(state.get("research_brief"))
            existing_topics = {call["args"]["research_topic"] for call in allowed_calls}
            extra_topics = [t for t in topics if t not in existing_topics]
            for topic in extra_topics:
                if len(allowed_calls) >= cfg.max_concurrent_research_units:
                    break
                allowed_calls.append(
                    {
                        "name": "ConductResearch",
                        "args": {"research_topic": topic},
                        "id": f"auto_conduct_research_{len(allowed_calls)+1}",
                    }
                )

        research_tasks = [
            researcher_subgraph.ainvoke(
                {
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"],
                },
                config,
            )
            for tool_call in allowed_calls
        ]
        tool_results = await asyncio.gather(*research_tasks)

        for observation, tool_call in zip(tool_results, allowed_calls):
            all_tool_messages.append(
                ToolMessage(
                    content=observation.get(
                        "compressed_research",
                        "Error synthesizing research report: Maximum retries exceeded",
                    ),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        for overflow_call in overflow_calls:
            all_tool_messages.append(
                ToolMessage(
                    content=(
                        "Error: exceeded maximum concurrent research units. "
                        f"Please use {cfg.max_concurrent_research_units} or fewer."
                    ),
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"],
                )
            )

        raw_notes_concat = "\n".join(
            "\n".join(observation.get("raw_notes", [])) for observation in tool_results
        )
        if raw_notes_concat:
            update_payload["raw_notes"] = [raw_notes_concat]

    update_payload["supervisor_messages"] = all_tool_messages
    return Command(goto="supervisor", update=update_payload)


supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()


# ===== Final report node

def final_report_generation(state: AgentState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    notes = state.get("notes", [])
    raw_notes = state.get("raw_notes", [])
    findings = "\n".join([*notes, *raw_notes])
    papers = state.get("supervisor_state", {}).get("deduped_papers", [])
    paper_metadata = json.dumps(papers, indent=2)

    prompt = final_report_generation_prompt.format(
        research_brief=json.dumps(state.get("research_brief") or {}, indent=2),
        messages=get_buffer_string(state.get("messages", [])),
        findings=findings,
        paper_metadata=paper_metadata,
    )

    response = _with_retry(_build_llm(cfg), cfg).invoke([HumanMessage(content=prompt)])
    return {
        "final_report": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def global_rag_context(state: AgentState, config: RunnableConfig) -> Command[Literal["final_report_generation"]]:
    query = ""
    brief = state.get("research_brief") or {}
    if isinstance(brief, dict):
        query = brief.get("boolean_query_generic") or " ".join(brief.get("must_include", []) or [])
    if not query:
        query = get_buffer_string(state.get("messages", []))
    observation = rag_search.invoke({"query": query}, config)
    return Command(
        goto="final_report_generation",
        update={"raw_notes": [f"GLOBAL_RAG_CONTEXT\\n{observation}"]},
    )


# ===== Graphs
survey_builder = StateGraph(AgentState, input_schema=AgentInputState)

survey_builder.add_node("clarify_with_user", clarify_with_user)
survey_builder.add_node("write_research_brief", write_research_brief)
survey_builder.add_node("search_arxiv_papers", run_arxiv_search)
survey_builder.add_node("search_acm_papers", run_acm_search)
survey_builder.add_node("finalize_search_results", finalize_search_results)
survey_builder.add_node("ingest_pdfs", ingest_pdfs)
survey_builder.add_node("research_supervisor", supervisor_subgraph)
survey_builder.add_node("global_rag_context", global_rag_context)
survey_builder.add_node("final_report_generation", final_report_generation)

survey_builder.add_edge(START, "clarify_with_user")
survey_builder.add_edge("write_research_brief", "search_arxiv_papers")
survey_builder.add_edge("write_research_brief", "search_acm_papers")
survey_builder.add_edge("search_arxiv_papers", "finalize_search_results")
survey_builder.add_edge("search_acm_papers", "finalize_search_results")
survey_builder.add_edge("finalize_search_results", "ingest_pdfs")
survey_builder.add_edge("ingest_pdfs", "research_supervisor")
survey_builder.add_edge("research_supervisor", "global_rag_context")
survey_builder.add_edge("global_rag_context", "final_report_generation")
survey_builder.add_edge("final_report_generation", END)

survey_research = survey_builder.compile()


__all__ = [
    "survey_research",
    "clarify_with_user",
    "write_research_brief",
    "run_arxiv_search",
    "run_acm_search",
    "finalize_search_results",
    "ingest_pdfs",
    "final_report_generation",
]
