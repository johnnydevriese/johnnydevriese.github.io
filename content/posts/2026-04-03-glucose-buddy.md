---
layout: post
title: "Glucose Buddy: Turning an AI Agent into a Real Product"
date: 2026-04-03
categories: artificial-intelligence engineering product
slug: glucose-buddy
---

# Glucose Buddy: Turning an AI Agent into a Real Product

Last year I wrote about building an AI agent for glucose tracking. That first version was useful as an experiment in agent design, typed tools, and natural-language interaction. It proved the concept. It did not yet feel like a product.

This new iteration, **Glucose Buddy**, is the version where I took the idea seriously as software.

The goal was no longer just "can an LLM help log glucose readings?" The goal became:

- can this feel trustworthy?
- can the user interface feel considered?
- can the architecture advertise good engineering judgment?
- can someone clone it, run it, and believe it could become a real app?

That reframing changed almost everything.

## From Agent Demo to Product System

The original version leaned harder on the idea of an agent. The new version uses AI more selectively.

That was an intentional engineering decision.

In a health-adjacent workflow, I do not want the critical path to depend on model luck. So the core logging flow is now **deterministic first**:

- parse glucose values from natural language
- identify fasting vs. post-meal context
- validate the extracted reading
- ask for confirmation before saving

The LLM still has a role, but it is a supporting one. It helps with classification, conversational polish, and flexible interaction, while the structured workflow stays predictable.

That balance is exactly the kind of tradeoff I care about when building AI products: use models where they create leverage, and use conventional software where reliability actually matters.

## The Engineering Work

I used the project as an excuse to do a full-stack modernization pass.

On the backend, I refactored a prototype into a clearer package structure with separate responsibilities for parsing, orchestration, persistence, analytics, schemas, and settings. I moved the project onto **Python 3.10+, `uv`, and `taskipy`**, added tests, and made the stack easier to run locally and in containers.

On the frontend, I upgraded the app to **Svelte 5 and SvelteKit 2**, redesigned the interface, improved the visual hierarchy, and pushed it toward something that looks more like a product showcase than an internal tool.

I also productionized the runtime:

- multi-stage Docker images
- Docker Compose for local startup
- a compiled SvelteKit Node server instead of a dev server in a container
- CI to validate backend, frontend, and Docker build paths

That kind of work is not glamorous, but it is exactly what turns a prototype into software other people can trust.

## Why I Think This Version Is Better

The newer Glucose Buddy project demonstrates a more complete engineering story than the first post:

- better product taste
- better architecture
- better tooling
- better delivery discipline
- better judgment about where AI belongs

I still like agentic systems. I think they are genuinely powerful. But I also think many AI demos stop one layer too early. They show that the model can do something interesting, but they do not show that the surrounding system is coherent, reliable, and pleasant to use.

This project is my attempt to close that gap.

## Repo

The source is here:

<https://github.com/johnnydevriese/glucose_doc>

If you want to compare it to the earlier thinking, the older post is still useful as a record of the original agent-centric framing. This newer repo is the version where that idea matured into something closer to a real product.
