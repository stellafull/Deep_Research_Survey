"""Prompt templates for the deep research survey system."""

clarify_with_user_instructions="""
These are the messages that have been exchanged so far from the user asking for the survey:
<Messages>
{messages}
</Messages>

Today's date is {date}.

Assess whether you need to ask a clarifying reseach scope, or if the user has already provided enough information for you to start research.
IMPORTANT: If you can see in the messages history that you have already asked a clarifying question, you almost always do not need to ask another one. Only ask another question if ABSOLUTELY NECESSARY.

If there are acronyms, abbreviations, or unknown terms, ask the user to clarify.
If you need to ask a question, follow these guidelines:
- Be concise while gathering all necessary information
- Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner.
- Use bullet points or numbered lists if appropriate for clarity. Make sure that this uses markdown formatting and will be rendered correctly if the string output is passed to a markdown renderer.
- Don't ask for unnecessary information, or information that the user has already provided. If you can see that the user has already provided the information, do not ask for it again.

Respond in valid JSON format with these exact keys:
"need_clarification": boolean,
"question": "<question to ask the user to clarify the survey scope>",
"verification": "<verification message that we will start research>"

If you need to ask a clarifying question, return:
"need_clarification": true,
"question": "<your clarifying question>",
"verification": ""

If you do not need to ask a clarifying question, return:
"need_clarification": false,
"question": "",
"verification": "<acknowledgement message that you will now start research based on the provided information>"

For the verification message when no clarification is needed:
- Acknowledge that you have sufficient information to proceed
- Briefly summarize the key aspects of what you understand from their request
- Confirm that you will now begin the research process
- Keep the message concise, professional and academic
"""

transform_messages_into_boolean_query_prompt = """
You will be given messages exchanged so far between me (the user) and you (the assistant).
Your job is to produce Boolean search queries for academic paper retrieval tools.

<Messages>
{messages}
</Messages>

Today's date is {date}.

Target sources (you MUST generate a query for each):
{sources}

=== Output requirements (STRICT) ===
Return ONLY a single JSON object (no markdown, no extra text), following this schema:

{{
  "scope_notes": {{
    "user_constraints": string[],
    "open_considerations": string[]
  }},
  "filters": {{
    "year_range": {{"from": number|null, "to": number|null}},
    "language": string|null,
    "doc_types": string[]|null
  }},
  "boolean_query_generic": string,
  "queries": {{
    "arxiv": {{"search_query": string, "notes": string[]}},
    "acm":   {{"query": string, "notes": string[]}}
  }},
  "must_include": string[],
  "must_exclude": string[]
}}

=== Construction guidelines ===
0) Interpret relative time windows:
- If I say "last N years" / "近N年" / "recent N years", convert it into an explicit year_range using {date}.
- Put the computed range into filters.year_range.
- Apply date filtering in source queries ONLY when that source clearly supports it.

1) Build concept groups first, then assemble:
- Create 2-4 core concept groups based on the user's request (e.g., serendipity, recommender systems, generative models).
- Each group must include synonyms, abbreviations, and common variants.
- Assemble the final query as: (A) AND (B) AND (C...), with OR inside each group.

2) Avoid unwarranted assumptions:
- Only put explicit negatives into must_exclude if I stated them.
- Unspecified dimensions (venues, categories, tasks, evaluation metrics) must go to open_considerations, not forced into the query.

3) Wildcards / truncation:
- Use wildcard/truncation ONLY when the target source supports it.
- Prefer explicit variants for arXiv (e.g., serendipity OR serendipitous; recommendation OR recommender OR recsys).
- For ACM, you MAY use safe high-recall stems like serendip* and recommend* to improve coverage.
- If using wildcard for ACM, do not put wildcards inside quoted phrases.

4) Source-specific rules:
- arXiv API supports field prefixes (ti, abs, au, co, jr, cat, rn, all), Boolean operators AND/OR/ANDNOT,
  parentheses grouping, and quoted phrases.
  It also documents submittedDate filtering with the format: submittedDate:[YYYYMMDDTTTT TO YYYYMMDDTTTT] (minutes, GMT).
  When calling the arXiv API, remind downstream to URL-encode spaces (+), parentheses, and quotes.
- ACM is retrieved via Crossref in this system. The ACM query MUST be Crossref-friendly:
  - Use a simple keyword query (space-separated terms).
  - Do NOT include boolean operators, wildcards, field prefixes, parentheses, or quotes.
  - Keep it short (roughly 4-8 core terms/phrases) to avoid over-filtering.
  If date filtering is needed, put it into filters.year_range (tool-side filter).

5) Auditable output:
- Put the best general query in boolean_query_generic.
- Put broader/narrower alternates (if any) into source notes.
- must_include must contain at least one representative term from EACH core concept group.

=== FEW-SHOT EXAMPLE ===
Input Messages (example):
<User>
I want papers about serendipity in recommender systems, focusing on generative models (LLMs, GANs, VAEs).
Time window: last 3 years.
Sources: arxiv and acm.
</User>

Output JSON (example):
{{
  "scope_notes": {{
    "user_constraints": [
      "Topic: serendipity in recommender systems",
      "Focus: generative models (LLM, GAN, VAE)",
      "Time window: last 3 years",
      "Sources: arxiv, acm"
    ],
    "open_considerations": [
      "Venues/categories are unspecified (consider all unless later constrained)",
      "Whether to include adjacent notions (novelty, diversity, unexpectedness) is unspecified"
    ]
  }},
  "filters": {{
    "year_range": {{"from": 2023, "to": 2026}},
    "language": null,
    "doc_types": null
  }},
  "boolean_query_generic": "((serendipity OR serendipitous OR \\"serendipitous recommendation\\" OR \\"serendipity-aware\\") AND (recommender OR recommendation OR \\"recommender system\\" OR \\"recommendation system\\" OR recsys) AND (\\"generative model\\" OR \\"generative models\\" OR \\"generative AI\\" OR \\"large language model\\" OR \\"language model\\" OR LLM OR \\"foundation model\\" OR GAN OR \\"generative adversarial network\\" OR VAE OR \\"variational autoencoder\\" OR diffusion OR \\"diffusion model\\"))",
  "queries": {{
    "arxiv": {{
      "search_query": "((ti:serendipity OR ti:serendipitous OR abs:serendipity OR abs:serendipitous) AND (ti:recommender OR abs:recommender OR ti:\\"recommender system\\" OR abs:\\"recommender system\\" OR ti:recsys OR abs:recsys) AND (ti:\\"large language model\\" OR abs:\\"large language model\\" OR ti:LLM OR abs:LLM OR ti:GAN OR abs:GAN OR ti:\\"generative adversarial network\\" OR abs:\\"generative adversarial network\\" OR ti:VAE OR abs:VAE OR ti:\\"variational autoencoder\\" OR abs:\\"variational autoencoder\\" OR ti:\\"generative model\\" OR abs:\\"generative model\\" OR ti:diffusion OR abs:diffusion)) AND submittedDate:[202302010000 TO 202601312359]",
      "notes": [
        "arXiv supports field prefixes ti/abs and boolean AND/OR/ANDNOT; keep parentheses for grouping.",
        "URL-encode spaces as '+', '(' as %28, ')' as %29, and quotes as %22 when calling the API."
      ]
    }},
    "acm": {{
      "query": "serendipity serendipitous recommender recommendation recsys generative model large language model LLM GAN VAE diffusion",
      "notes": [
        "Crossref-friendly keyword query (no boolean/wildcards/field prefixes).",
        "Apply year filter via Crossref date filters using filters.year_range."
      ]
    }}
  }},
  "must_include": [
    "serendipity",
    "recommender system",
    "large language model",
    "GAN",
    "VAE"
  ],
  "must_exclude": []
}}

Remember: output ONLY the JSON object.
"""
