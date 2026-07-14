---
title: Building an AI Agent for Glucose Tracking
author: Johnny Devriese
date: March 3, 2025
tags: [GenAI, Machine Learning, Agents, pydantic-ai]
---

# Building an AI Agent for Glucose Tracking

## The Promise of Agentic AI in Healthcare

AI agents are useful when a model needs to do more than produce text. Instead of only answering a prompt, an agent can choose tools, call APIs, inspect results, and decide what to do next from a natural language request.

This matters in domains like healthcare, where getting from intent to action often means clicking through rigid UIs, memorizing API endpoints, or writing custom scripts. A user should be able to ask, "What was my average glucose level last week?" and get an answer without writing a database query or opening a dashboard.

## Why Agents? Why Now?

The agentic pattern is powerful precisely because it bridges the gap between human intention and computational execution. Consider the alternative: a user needs to know *exactly* which endpoint to call, what parameters to pass, and how to interpret the response. With an agent, the LLM becomes a reasoning layer that:

- **Interprets natural language queries** with all their ambiguity and context
- **Selects the appropriate tool** from a registry of available functions
- **Orchestrates multi-step workflows** when a single tool isn't sufficient
- **Handles errors gracefully** and can retry or pivot when something goes wrong

For glucose tracking specifically, this means a user can ask complex questions like "Show me my glucose spikes after meals this month" and the agent will understand it needs to query time-series data, filter by time windows, identify spike patterns, and perhaps correlate with meal logging data.

## The Architecture: Building with Pydantic AI

I built this agent using [pydantic-ai](https://ai.pydantic.dev/), a framework that brings type safety and structured validation to agent development. Here's what the architecture looks like:

### Tool Registration

Each capability is exposed as a typed tool that the LLM can invoke:

```python
@agent.tool
async def get_glucose_readings(
    start_date: datetime,
    end_date: datetime,
    user_id: str
) -> List[GlucoseReading]:
    """Retrieve glucose readings within a date range."""
    # Query database, validate data, return structured results
    pass
```

The useful part of pydantic-ai is that these tool signatures become part of the agent's context. The LLM sees the available tools, their parameters, and their docstrings, which gives it enough structure to choose tools and chain them together.

### State Management: The Real Challenge

One of the most interesting challenges I encountered was **state persistence across LLM calls**. Here's why this matters:

In a typical REST API, each request is stateless. But agent conversations are inherently stateful—the agent needs to remember:
- What the user asked three turns ago
- What tools were already invoked
- Intermediate results from previous steps
- Context about the user's data and preferences

My initial approach used **WebSockets** to maintain persistent connections, allowing real-time streaming of agent thoughts and results. However, pydantic-ai also supports **serializing conversation history** to JSON, which can be persisted to a database. This enables:

- Session resumption across disconnections
- Multi-device continuity
- Audit trails for healthcare compliance
- Fine-tuning data collection from real conversations

### The Pattern in Practice

Here's a simplified flow of how a user interaction works:

1. **User Query**: "What's my average glucose over the last 7 days?"
2. **Agent Reasoning**: The LLM parses the query and determines it needs the `get_glucose_readings` tool with `start_date = today - 7 days` and `end_date = today`
3. **Tool Execution**: The system invokes the tool, queries the database, validates results
4. **Result Processing**: The agent receives structured data and computes the average
5. **Response Generation**: The LLM formats the answer naturally: "Your average glucose over the last 7 days was 112 mg/dL, which is within normal range."

What makes this powerful is that the agent can *chain* tools. If the user follows up with "How does that compare to last month?", the agent maintains context and knows to query a different time window without needing the user to re-specify everything.

## Looking Forward: The Evolution of Agent Workflows

We're still in the early days of agent design, and several exciting directions are emerging:

**Reinforcement Learning for Improved Tool Selection**: Instead of relying solely on few-shot prompting, we can fine-tune agents with RL to learn optimal tool sequences from successful completions. Imagine an agent that learns to anticipate which tools a user typically needs based on query patterns.

**Multi-Agent Collaboration**: Complex healthcare tasks might benefit from specialized agents that collaborate—one for data retrieval, another for statistical analysis, another for generating visualizations. Orchestrating these agents efficiently is an open research problem.

**Streaming and Observability**: As agents invoke multiple tools, providing real-time feedback about what's happening "behind the scenes" improves trust and debuggability. Streaming partial results and exposing the agent's reasoning process will be crucial for adoption.

**Hybrid Systems**: Not every task needs an agent. Sometimes a direct API call is faster and more predictable. The future likely involves hybrid architectures where agents handle ambiguous, exploratory tasks while structured APIs handle routine operations.

## Try It Yourself

The full implementation is open source and available on GitHub:

**Repository**: [github.com/johnnydevriese/glucose-agent](https://github.com/johnnydevriese/glucose-agent)

The repo includes:
- Complete agent implementation with pydantic-ai
- WebSocket server for real-time interactions
- Example tools for glucose data management
- Docker setup for easy deployment

The agent pattern fits domains where natural language needs to drive structured data workflows. The hard part is balancing flexibility with reliability: users should be able to express intent naturally, while the system still executes actions safely and predictably.

What domains do you think would benefit most from agentic AI? I'd love to hear your thoughts.