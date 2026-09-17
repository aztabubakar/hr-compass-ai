# Proposed Architecture

```text
Browser / Chat UI
       |
       v
Web Application (/chat, /health)
       |
       v
Agent Orchestrator + MCP Client
   |          |           |
   |          |           +--> LLM Provider
   |          |
   |          +--> MCP Server
   |                 |-- search_policy_documents --> RAG index
   |                 |-- get_policy_section ------> policy corpus
   |                 |-- lookup_employee_profile -> mock data
   |                 |-- check_pto_balance -------> mock data
   |                 |-- lookup_benefits_status --> mock data
   |                 |-- check_policy_compliance -> policy + structured context
   |                 |-- draft_hr_email ----------> mock/draft operation
   |                 +-- create_mock_hr_ticket ---> mock operation + confirmation
   |
   +--> Operational trace returned with final answer
```

## RAG
Policy documents are ingested, cleaned, chunked, embedded, and persisted locally. Chunks retain document ID/title, section heading, source path, and snippet. Retrieval returns top-k evidence. Final policy claims must be grounded in retrieved evidence and cite document/section identifiers.

## Agent behavior
The orchestrator classifies the request into policy-only, structured-data/tool-required, multi-step workflow, ambiguous, or out-of-scope. It discovers available MCP tools and invokes them as needed. It synthesizes the final answer from tool outputs and retrieved evidence.

## Operational trace
Expose events such as:
- intent/workflow selected,
- MCP tool selected,
- sanitized tool arguments,
- tool success/failure and concise result summary,
- policy sources retrieved,
- confirmation/escalation decision,
- final response basis.

Never expose hidden chain-of-thought.

## Demo workflows
### PTO
`lookup_employee_profile` -> `check_pto_balance` -> `search_policy_documents`/`get_policy_section` -> `check_policy_compliance` -> optional `draft_hr_email`.

### Remote work
`lookup_employee_profile` -> `search_policy_documents` across remote-work/security/location policies -> `check_policy_compliance` -> escalation/next steps when approval is required.
