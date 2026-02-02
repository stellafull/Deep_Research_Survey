This project aims to build a survey agent to help researchers collect and write surveys in specific domains. The agent collects papers from arXiv and ACM (via Crossref), deduplicates them, and supports RAG over the retrieved PDFs.

This still a very early project.

Graphs (LangGraph):
- `scope_research`: clarify scope → build queries → fetch arXiv/ACM → persist + dedupe
- `survey_research`: scope flow + PDF ingestion + RAG-grounded report
