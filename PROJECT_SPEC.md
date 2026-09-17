# HR Compass AI — Quantic AI Engineering Project Specification

## Goal
Build, deploy, and evaluate an agentic AI system for a hypothetical company's HR policy and operations tasks. The system must combine policy RAG, an agent orchestrator, MCP-exposed tools, synthetic structured HR data, grounded citations, a web chat UI, CI/CD, deployment, and evaluation.

## Core requirements
- Policy corpus: 5–20 short Markdown/HTML/PDF/TXT files totaling roughly 30–120 pages.
- Parse/clean at least two source formats where feasible.
- Chunking strategy must be justified; embed chunks and store in a lightweight vector database.
- Preserve citation metadata: document title/ID, section, and source snippet.
- RAG: top-k retrieval, cited grounded answers, guardrails, and at least one multi-document question.
- Agent orchestrator decides whether RAG is sufficient, selects tools, calls MCP tools, and synthesizes responses.
- At least two multi-step HR workflows.
- Operational trace must show selected tools, arguments, outputs, retrieved sources, final-answer basis, and escalation decisions without exposing hidden chain-of-thought.
- Handle missing IDs, unavailable tools, insufficient policy evidence, and ambiguous requests gracefully.
- Irreversible actions must be mock or require explicit confirmation.
- At least five MCP tools. At least one accesses RAG; at least one accesses mock structured data or performs a mock operation.
- Agent must actually invoke tools through MCP.
- Web chat UI with /chat equivalent and /health equivalent.
- Deploy to a free-tier/zero-cost host such as Render or Railway.
- CI/CD on push/PR: dependency install, build/start/import check, tests, app-start test, and MCP discovery/tool-call test. Deployment only after tests pass.
- Evaluation set: 20–30 questions/tasks with gold answers/rubrics, including simple Q&A, multi-document, tool-required, ambiguous, and out-of-scope cases.
- Metrics: groundedness, citation accuracy, tool selection accuracy, workflow completion, clarification/escalation accuracy, action-safety pass rate, and p50/p95 latency. Include at least one ablation/comparison.
- Documentation: architecture/design choices, MCP architecture/transport/tool schemas, RAG choices, safety, deployment, evaluation, and two demo workflows.
- Final repository must include README.md, design-and-evaluation.md, ai-tooling.md, deployed.md, evaluation/, mock_data/, and mcp/.
- Recorded demo: 7–10 minutes, two end-to-end agentic tasks, showing/explaining MCP tool names, arguments, outputs, citations, final response/action, plus architecture, deployment, CI/CD, and evaluation.

## Recommended demo workflows
1. PTO request guidance: employee asks about taking three days next week. Look up employee, check PTO balance, retrieve PTO/approval policy, check compliance, and optionally draft a manager message.
2. Remote work eligibility: employee asks to work from another state/country for six weeks. Look up employee, retrieve remote-work/security/location/tax/approval policies, check compliance, and return cited next steps or escalation.

## Required implementation quality target
Target the top rubric band: functional deployed agentic HR system, correct grounded citations, real MCP integration, clear traces, graceful errors, strong RAG, two end-to-end workflows, CI/CD, evaluation, architecture documentation, and a reproducible demo.
