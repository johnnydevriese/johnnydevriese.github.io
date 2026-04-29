---
layout: post
title: "Building Enterprise Agents That Actually Work"
date: 2026-04-28
categories: artificial-intelligence engineering agents
slug: enterprise-agents-playbook
---

# Building Enterprise Agents That Actually Work

Over the past year our team built an agentic system for enterprise financial workflows — invoice extraction, lease parsing, payment matching, analytics. We learned most of our lessons the hard way. This post distills the architectural patterns that survived contact with production.

---

## The God Agent Anti-Pattern

We started where most teams start: one agent, forty tools, a system prompt that grew longer every sprint.

It worked in demos. It even worked for simple requests. Then we hit real traffic, and everything degraded together. The agent would call `submit_lease` when the user asked about an invoice. It hallucinated plausible-looking tool arguments that silently corrupted data. Instructions buried deep in the prompt were ignored entirely.

The pattern was clear: **the more tools we added, the worse every tool performed.** Attention is a finite resource, and we were spending it on context the agent didn't need for any given request.

![The God Agent anti-pattern](/blog_assets/god-agent-anti-pattern.png)

---

## The Fix: Router-Worker Architecture

Instead of one monolith, we decomposed into a **lightweight router** and **specialist workers**.

The router reads the request, classifies intent, and hands it to the right specialist. Each worker sees only its own tools, its own short prompt, and its own domain context.

| Concern | Monolithic Agent | Router-Worker |
|:--|:--|:--|
| **Tool selection** | Models confuse arguments when overloaded | Workers see only relevant tools |
| **Attention** | Critical instructions buried mid-prompt | Short, high-signal prompts |
| **Debugging** | "The AI failed" | "The Invoice Agent failed validation on `tax_amount`" |
| **Maintenance** | Changing one tool means re-testing everything | Update one worker without touching others |

![Router-Worker Architecture](/blog_assets/router-worker-architecture.png)

The router runs on a small, fast model — classification doesn't need a frontier model. The worker registry is a plain dictionary. No framework magic. A new engineer can read the orchestration function and understand the entire flow.

---

## Type-Safe Boundaries

Most agent frameworks pass data as loosely-typed strings. We used **PydanticAI** so that every agent output is validated against a typed schema before it touches a database.

If the LLM drifts — negative amounts, missing fields, malformed dates — Pydantic catches it and retries with the validation error as context. The LLM learns from its own mistake within the same conversation turn.

The system prompt is four lines. The agent sees two tools. Compare that to a God Agent with fifty tools and a prompt measured in pages.

---

## Reversibility: The Undo Problem

Most agent tutorials end at "the agent completed the task." Nobody talks about what happens when the agent completed the _wrong_ task.

We classify every tool into three tiers:

- **Read actions** — no side effects, always safe.
- **Soft writes** — creates a draft/staged record. Reversible.
- **Hard writes** — submits, sends, or deletes. Requires human confirmation.

Every soft write is tagged with a session ID. A user can hit one button and revert everything the agent did in the current conversation. The undo isn't per-action; it's per-session.

![Action Tiers](/blog_assets/action-tiers.png)

**Reversibility is what turns an agent from a liability into a collaborator.** Users who know they can revert are more willing to let the agent try things.

---

## The Data Flywheel

We don't just log errors — we harvest them. Every failure is potential training data.

The loop follows the classic **MAPE** cycle:

1. **Monitor** — every turn logs a `prompt_hash` and `model_version`.
2. **Analyze** — a scheduled job pulls traces where the model's output failed schema validation.
3. **Plan** — failures are formatted into a `failures.jsonl` dataset, organized by failure mode.
4. **Execute** — the dataset drives **LoRA fine-tuning** or **GRPO reinforcement learning**, teaching the model the exact structures it keeps getting wrong.

![The MAPE Data Flywheel](/blog_assets/mape-flywheel.png)

Over time, this flywheel narrows the gap between what we ask for and what we get — without manual prompt engineering.

---

## The Request Lifecycle

Putting it all together, a single request passes through seven stages:

1. **PII scanning** — sensitive data is masked before any LLM sees it.
2. **Token budget gate** — the session is checked against a hard cap to prevent runaway costs.
3. **Router agent** — intent is classified and delegated.
4. **Specialist worker** — the domain agent runs with its focused prompt and scoped tools.
5. **Schema validation** — the LLM's output is checked against its typed contract.
6. **Retry on failure** — if validation fails, the worker retries with the error as context.
7. **Validated response** — only type-checked, PII-safe output reaches the user.

![Request Lifecycle](/blog_assets/request-lifecycle.png)

Failures are caught at the schema level and retried automatically. The user never sees a malformed response. When retries are exhausted, the failure is logged with enough context to feed the data flywheel.

---

## What I'd Tell a Team Starting Today

1. **Decompose early.** The God Agent is a trap. Start with a router and two workers. Add workers as domains emerge.

2. **Type your outputs.** If the LLM's output isn't validated against a schema, you're hoping, not engineering.

3. **Design for undo.** Classify every tool by risk tier. Make soft writes reversible by default.

4. **Build the flywheel from day one.** Log every schema failure. Format them as training data. Use them to improve the model. The compound returns are enormous.

5. **Degrade gracefully.** When LLM providers go down (and they will), your system should drop to progressively simpler capabilities — not show a blank error page. Design every degradation tier explicitly, before the incident.

None of this is novel. It's just good software engineering applied to AI systems. The hard part isn't knowing these patterns — it's having the discipline to implement them before production traffic forces your hand.
