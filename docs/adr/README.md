# Architectural Decision Records (ADRs)

This directory contains Architectural Decision Records (ADRs) for the OpenAgent project. ADRs document important architectural decisions and their context, consequences, and rationale.

## What are ADRs?

Architectural Decision Records are documents that capture important architectural decisions made during the project's development, along with their context and consequences. They help teams:

- Understand why certain decisions were made
- Maintain consistency in architectural choices
- Onboard new team members quickly
- Review and potentially reverse decisions when needed

## ADR Format

We use a lightweight ADR format with the following structure:

```markdown
# ADR-XXXX: [Decision Title]

## Status
[Proposed | Accepted | Deprecated | Superseded by ADR-YYYY]

## Context
Description of the forces and constraints that led to this decision.

## Decision
The architectural decision that was made.

## Consequences
What becomes easier or more difficult as a result of this decision.

## Alternatives Considered
What other options were evaluated and why they were rejected.
```

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](ADR-001-agent-tool-pattern.md) | Agent-Tool Pattern Architecture | Accepted | 2024-01 |
| [ADR-002](ADR-002-multi-llm-support.md) | Multi-LLM Provider Support | Accepted | 2024-01 |
| [ADR-003](ADR-003-policy-driven-security.md) | Policy-Driven Security Architecture | Accepted | 2024-01 |
| [ADR-004](ADR-004-async-first-design.md) | Async-First Design Pattern | Accepted | 2024-01 |
| [ADR-005](ADR-005-plugin-architecture.md) | Plugin System Architecture | Accepted | 2024-01 |

## Creating New ADRs

When making significant architectural decisions:

1. Create a new ADR file: `ADR-XXXX-short-title.md`
2. Use the next sequential number
3. Follow the standard format
4. Update this index
5. Submit as part of your pull request
6. Get architectural review before merging

## Tools

- Use [adr-tools](https://github.com/npryce/adr-tools) for ADR management (optional)
- Keep ADRs in version control with the rest of the codebase
- Link to relevant code, issues, or documentation
