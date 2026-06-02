---
layout: post
title: "The Bitter Lesson Is Not a Law"
date: 2026-04-30
categories: artificial-intelligence philosophy research
slug: bitter-lesson-is-not-a-law
---

# The Selection Effect

_On benchmarks, compression, and what counts as a principle_

---

Rich Sutton's _Bitter Lesson_ is routinely cited as the organizing truth of modern AI — seventy years of history distilled into a single directive, presented as something close to a law. A law is only as strong as the evidence behind it. The evidence here has a signature worth examining.

> _General methods that leverage computation are ultimately the most effective, and by a large margin._ — Richard Sutton

## I. The Record

Chess. Go. ImageNet. Protein structure. Next-token prediction on the open internet. These are the canonical victories of scale, and they share four properties: a well-defined objective, a cheap and mechanical way to check answers, effectively unlimited practice, and a loss function that remains informative across many orders of magnitude.

The Bitter Lesson is usually read as _compute beats cleverness_. But every item on that list is a domain where compute _could_ be applied — because the domain was shaped to admit it. The field did not select these problems because they were the important ones. It selected them because they were the ones its tools could touch.

The pattern repeats in miniature. As training-data scaling neared its limits, the field pivoted to inference-time compute — letting models think longer before answering. The benchmarks that rose to meet the new wave, competition mathematics, coding, agentic evaluations, share the same signature: verifiable answers, mechanical graders, clean rewards. When a frontier saturates, the field does not conclude the Lesson has a boundary. It builds a new frontier shaped to keep the Lesson true.

## II. The Missing Half

In domains without a cheap oracle — scientific discovery, mathematical taste, long-horizon judgment, anything whose correctness cannot be verified by a script — compute has not demonstrably won. It has not demonstrably lost. It has largely not been tried, because the benchmark infrastructure to try it does not exist.

The deepest case is the one the Lesson cannot reach on its own terms. For genuinely novel discovery, no cheap oracle is possible, because the oracle would need to be as capable as the thing being graded.

> _The scaling recipe requires a teacher. The most interesting questions don't come with one._

## III. What the Lesson Actually Says

Physics offers the contrast. It is a field that chose its problems before it had the tools to touch them — planetary motion, the nature of heat, the structure of the atom — and built the mathematics to reach them over centuries. The benchmark came first; the instrument was shaped to fit. Modern AI has inverted that order. It has built an extraordinary instrument and selected problems to match its shape.

It is worth being precise, then, about what kind of regularity the Bitter Lesson actually is. Newton's first law describes the world whether anyone is measuring or not. Moore's Law described a manufacturing target the industry organized itself to hit — geometric, economic, contingent on the choice to keep shrinking transistors rather than do something else. It held for fifty years and stopped holding when the choices changed. The Bitter Lesson sits closer to Moore than to Newton. It is conditional on a cheap oracle, a differentiable loss, and a field willing to keep building benchmarks that admit those conditions. It is not a discovery about cognition. It is a description of how silicon performs against problems shaped to suit it.

The strongest case against this reading is compression. Models that once required warehouses now run on laptops at comparable quality, which suggests something genuinely structural was learned — a mere lookup table should not compress so cleanly. Information theory has long held that compression is a form of understanding, and the concession is worth making in full: these models have found _something_ about the regularities of their training distribution, and that something is not nothing.

But there are two kinds of compression, and physics has known the difference for four hundred years. Ptolemy's epicycles compressed planetary motion to a handful of nested circles and predicted the night sky well enough to survive for a millennium. They were wrong. _F = ma_ compresses too, but it compresses to a form that can be inspected, extended to regimes it was never fit on, and composed with other laws to yield new ones. The first kind of compression fits a curve. The second kind hands you a principle.

> _A principle can be inspected, extended, and composed. A fit cannot. They are not on a spectrum; they are different objects._

A neural network distilled from a warehouse to a laptop has done something real, but what it has done is Ptolemaic. The weights still cannot be read, extended, or composed. A tight fit to a chosen benchmark, compressed, is still a tight fit to a chosen benchmark.

Stated honestly, the Lesson reads: _given a problem with a cheap oracle, a differentiable loss, and unlimited practice, compute scales better than human insight._ That is a real and useful claim. It is also a much narrower one than the version in circulation.

The move from the narrow claim to the broad one is the sleight of hand. It treats the domains where evidence exists as if they were representative of cognition itself. They are representative of the problems the field chose to benchmark. The Bitter Lesson is not a discovery about intelligence. It is a self-portrait of a research culture that picked its fights to match its weapons.

---

_April 2026 · Philosophical Engineering Series_