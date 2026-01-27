---
layout: post
title: "The Shannon Problem: Why We Need a Mathematics of Neural Networks"
date: 2026-01-26
categories: artificial-intelligence research interpretability
slug: the-shannon-problem
---

# The Shannon Problem: Why We Need a Mathematics of Neural Networks

In 1937, Claude Shannon was a master's student at MIT working on a theoretical problem that seemed narrow and academic: could you describe electrical circuits using algebra?

Before Shannon, people built circuits by trial and error. They worked. Engineers knew rules of thumb. But there was no systematic mathematical framework to describe what a circuit actually computed. You couldn't look at a circuit and write down its logic. You couldn't design one from first principles.

Shannon showed you could represent circuits with Boolean algebra. Switches could be AND, OR, NOT gates. You could write down the mathematics of what any circuit computed. Suddenly you could reason about circuits formally, optimize them systematically, prove properties about them.

This wasn't just an incremental improvement. It was the foundation for digital computing. Everything we've built since—processors, memory, the device you're reading this on—depends on having the right mathematical language to describe circuits.

We don't have that for neural networks. And it's becoming a problem.

## We're Building Without Understanding

Right now, we train neural networks and they work. We can measure their behavior on benchmarks. We can make them better through careful training. We have rules of thumb about architecture and optimization.

But we don't have the equivalent of a circuit diagram. We can't look at the weights of a trained model and write down what it computes. We can't design one from first principles to reliably perform a specific task beyond "train it and see what happens."

I've felt this gap directly. I once spent weeks debugging why a RAG system would occasionally return completely irrelevant results. The model looked confident. The retrieval seemed fine. Something was broken.

Eventually I traced it to how the model weighted certain tokens in the retrieved context. But I found this through extensive testing and intuition—not by understanding what the model was actually computing. I was debugging a black box.

This works fine when you're building toys. It starts to matter when you're deploying systems that people depend on. And it becomes critical as models get more capable.

## What Interpretability is Actually Trying to Do

Mechanistic interpretability is searching for the Boolean algebra of neural networks.

Not just "what does this model output" but "what computations does it perform internally." Not just behavioral measurements but mechanistic understanding. The goal is to reverse-engineer trained models into something we can actually reason about.

This is harder than it sounds. Neural networks don't come with clean abstraction layers. A single neuron can participate in representing multiple concepts simultaneously (superposition). The computational units we thought were fundamental—individual neurons, attention heads—turn out to be uninterpretable when you look closely.

Recent work from Anthropic's interpretability team is making progress. They've found ways to extract interpretable "features" from models—units of computation that actually correspond to coherent concepts. They've started building "circuits" that describe how these features connect to perform specific tasks. They've done this at scale, finding millions of features in production models.

This is early-stage work. But it's moving in the right direction: finding the primitives, the abstractions, the mathematical language to describe what neural networks compute.

## Why This Matters

You might think: "These models work pretty well already. Why do we need to understand them mechanistically?"

A few reasons.

**First: trust.** You can't fully trust what you don't understand. Right now we test models extensively, build guardrails, hope they behave. But we're fundamentally guessing. As these systems get more capable and we deploy them in higher-stakes contexts, "we tested it a lot" stops being adequate.

**Second: reliability.** Without understanding mechanisms, debugging is trial and error. You can't systematically fix problems. You can't predict failure modes. You're patching symptoms without understanding root causes.

**Third: capability.** Shannon's work didn't just let us build the same circuits more confidently—it let us build circuits we couldn't have built before. Once we understood the mathematics, we could design complex systems from first principles. The same will be true for neural networks. Understanding mechanisms will unlock capabilities that pure scaling can't reach.

**Fourth: safety.** As models become more capable, the stakes get higher. We need to know not just that a model behaves correctly in testing, but that we understand *why* it behaves that way and can predict how it generalizes. Behavioral testing scales poorly with capability. Mechanistic understanding might actually scale.

## Where We Are

We're somewhere between "trial and error circuits" and "Boolean algebra." We have some tools. We can identify features. We can trace some computational paths. We're starting to build simple circuits.

But we don't have the full mathematical framework. We can't yet look at a trained model and systematically describe all the computations it performs. We can't design models from first principles to provably compute specific things.

This is normal for foundational science. Shannon didn't solve everything in 1937. But he gave us the right abstractions to build on. That's what interpretability research is searching for now.

The work is hard. Neural networks are messier than circuits. They're optimized by gradient descent, not designed by humans. They operate in high-dimensional spaces that don't map cleanly to human concepts. Finding the right level of abstraction—the equivalent of "gates" for neural networks—requires both theoretical insight and extensive empirical work.

But it's also some of the most important work happening in AI right now.

## The Foundation For Everything Else

Most AI work focuses on making models more capable. That's important. But capabilities without understanding is building on sand.

We need the equivalent of what Shannon gave us for circuits: a mathematical language to describe what neural networks actually compute. Not just better behavioral testing. Not just more extensive evaluation. Actual mechanistic understanding.

This is foundational work. Everything else in AI safety—alignment, robustness, truthfulness—gets easier if we understand what we're building. And everything gets harder, maybe impossible, if we don't.

Right now we're in the pre-Shannon era of AI. We can build these systems. We can make them work. But we're still figuring out how to describe what they actually do.

Someone will solve this. The question is whether we solve it before or after we've deployed systems powerful enough that not understanding them becomes a serious problem.

I'm betting on before. But only if people work on it.
