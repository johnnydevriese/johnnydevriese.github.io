---
layout: post
title: "Prompting vs. Tuning: The Banking77 Intent Classification Bake-off"
date: 2026-04-20
categories: artificial-intelligence machine-learning engineering
slug: banking77-bakeoff
---

# Prompting vs. Tuning: The Banking77 Intent Classification Bake-off

So, you've got 77 possible banking intents and 10,000 customer queries. How do you build a classifier that doesn't just "kind of work," but actually nails the nuances between "my card was declined" and "my card is blocked"—two very different problems that sound suspiciously similar?

We took three of the most powerful modern techniques—**LoRA Fine-tuning**, **MIPROv2**, and **GEPA**—and pitted them against each other on the **Banking77** dataset. What emerged wasn't just a leaderboard, but a clearer picture of *when* each approach makes sense and *why* they behave so differently.

Here's the breakdown: how they work under the hood, what the physics of optimization tells us about their behavior, and why "trial and error" prompting is officially dead.

---

> **TL;DR**
> - **Got 5K+ labeled examples?** → Fine-tune with LoRA (91.4% accuracy, $0.02/1K queries)
> - **Only 50-500 examples?** → Use GEPA's reflective mutation (90.0% accuracy)
> - **Strong baseline already?** → Generic prompt optimizers like MIPROv2 won't help much
> 
> The secret sauce: GEPA *learns from its mistakes* by having an LLM analyze failure cases and write disambiguation rules. LoRA is still king for production, but GEPA closes the gap with 180× less data.

---

## The Problem: 77 Classes, Infinite Confusion

Banking77 is a deceptively hard benchmark. On the surface, it's "just" intent classification—match a customer query to one of 77 categories. But the devil is in the semantic overlap:

- `card_arrival` vs. `card_delivery_estimate` (when will it arrive vs. when *should* it arrive)
- `card_payment_wrong_exchange_rate` vs. `wrong_exchange_rate_for_cash_withdrawal`
- `declined_card_payment` vs. `card_not_working` vs. `card_swallowed`

These aren't just similar—they occupy nearly the same region in embedding space. Any classifier needs to learn incredibly fine-grained decision boundaries. Think of it like trying to distinguish between isotopes: the gross structure is identical, but the subtle differences matter enormously.

---

## The Tech Stack: Deeper Than a Prompt

### The "Software Engineering" of Prompts

We've moved past the era of long, hand-written prompts that break the moment you switch models or the wind changes direction. Using **DSPy**, we treat prompting like software engineering rather than creative writing.

Instead of crafting prose, we define **Signatures**—formal specifications like `query -> intent`. This abstraction separates the *what* (classify this query) from the *how* (the specific tokens and structure the model needs). DSPy then "compiles" this logic into whatever format works best for your target LLM.

Why does this matter? Because prompts are brittle. A prompt optimized for GPT-4 might fall apart on Claude. A few-shot example that works beautifully today might conflict with tomorrow's system prompt. By treating the prompt as a compiled artifact rather than hand-tuned scripture, we can systematically optimize and port across models.

---

### LoRA: Perturbation Theory for Neural Networks

**Low-Rank Adaptation (LoRA)** is the physicist's dream of an optimization technique. If you remember perturbation theory from quantum mechanics—where you solve a hard Hamiltonian by starting from a solved one and adding small corrections—LoRA does exactly that for neural networks.

Here's the setup: you have a pre-trained model (Qwen 2.5, 500M parameters) that already "knows" language. You want it to specialize in banking intents without destroying everything it learned. The naive approach—fine-tune all 500M parameters—is both computationally brutal and risks **catastrophic forgetting**, where the model overwrites general knowledge to memorize your specific task.

LoRA's insight: the *update* to the weight matrices during fine-tuning is typically low-rank. Instead of modifying the original weights $W$, we freeze them and learn two small matrices $A$ and $B$ such that:

$$W_{new} = W_{frozen} + BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with $r \ll d, k$. We're learning a small perturbation in a low-dimensional subspace of the full parameter space.

**Why it works**: Most of the "knowledge" in a language model is distributed across all those frozen parameters. The task-specific signal—"this combination of words means `card_arrival`"—can be captured by a much smaller set of adjustments. It's like adding a small potential to a solved system: you get the specificity you need without recomputing the entire wavefunction.

**The practical upside**: Training LoRA on a laptop. Inference costs identical to the base model. Zero API fees once deployed.

![LoRA: Perturbation Theory for Neural Networks](/blog_assets/plot_06_lora_diagram.png)

---

### MIPROv2: Gradient Descent on the Prompt Landscape

**Multi-Input Prompt Optimization (MIPROv2)** is DSPy's industrial-strength optimizer, and it's fundamentally a search algorithm over a rugged landscape.

Imagine every possible prompt—every combination of instructions, few-shot examples, and formatting choices—as a point in a high-dimensional space. Each point has an associated "fitness" (accuracy on your validation set). MIPROv2's job is to find the peak.

The problem: this landscape is non-differentiable, discontinuous, and full of local maxima. You can't just compute gradients. So MIPROv2 uses a **surrogate model**—essentially, a fast "critic" that learns to predict which prompts will perform well without actually running them. This is Bayesian optimization: build a model of the objective function, use it to decide where to sample next, update the model with new evidence.

**The catch for Banking77**: Bayesian optimization excels at finding good *average* solutions. It's less effective when the performance variance is driven by specific, rare failure modes. If your prompt gets 95% of queries right but systematically confuses two specific intents, MIPROv2 might not "see" that failure clearly in its surrogate model. It optimizes for the bulk of the distribution, not the tails.

This is exactly what happened. MIPROv2 found prompts that performed near the baseline—because the baseline was already capturing the "easy" 74% of cases. The remaining 26% required surgical precision that MIPROv2's statistical approach couldn't target.

---

### GEPA: Evolutionary Dynamics with Reflective Mutation

**Generative Prompt Evolution Algorithm (GEPA)** takes a different tack entirely. If MIPROv2 is gradient-free optimization, GEPA is more like directed evolution with an intelligent mutation operator.

Here's the loop:

1. **Evaluate**: Run the current prompt on a batch of examples.
2. **Identify failures**: Collect the specific cases where it got things wrong.
3. **Reflect**: Ask a separate "Reflection LM" to analyze *why* these failures occurred. What pattern did the model miss? What ambiguity tripped it up?
4. **Mutate**: Generate new prompt variants that explicitly address the diagnosed failure modes.
5. **Select**: Keep the best performers and repeat.

This is fitness-proportionate selection with **intelligent offspring generation**. Instead of random mutations (which would be hopelessly inefficient in prompt space), GEPA uses the reflection step to propose targeted fixes.

The physics analogy here is closer to **simulated annealing with domain knowledge**. You're not just randomly perturbing the system—you're using a model of the energy landscape to make informed jumps toward lower-energy configurations.

**What GEPA discovered on Banking77**: Through its reflective process, GEPA identified that the base model was systematically confusing `card_arrival` (tracking a specific shipment) with `card_delivery_estimate` (asking about standard delivery times). It then *wrote its own disambiguation rule* into the prompt:

> "If the user asks about a specific card they're waiting for, classify as `card_arrival`. If they ask generally about how long cards take to arrive, classify as `card_delivery_estimate`."

This is the kind of rule a human annotator would write after reviewing confusion matrices. GEPA automated that process.

Here's how GEPA's accuracy evolved over generations, with key rule discoveries annotated:

![GEPA Evolution Timeline](/blog_assets/plot_05_gepa_evolution.png)

And the impact on confusion errors—the pairs that tripped up the baseline were surgically addressed:

![Confusion Pairs: Baseline vs GEPA](/blog_assets/plot_04_confusion_pairs.png)

---

## The Showdown

We tested everything on the non-negotiable **Banking77 Test Set** (3,080 samples). No peeking, no leakage, no excuses.

Here's the full comparison at a glance:

![Method Comparison Table](/blog_assets/plot_08_comparison_table.png)

And the accuracy breakdown:

![Banking77 Classification Accuracy](/blog_assets/plot_01_main_results.png)

### Reading the Results

**The baseline was already strong.** 74.5% zero-shot is nothing to sneeze at—it means the base model already "gets" most banking intents without any task-specific optimization. This is both good news (you start from a high floor) and bad news (there's less room for generic optimization to help).

**MIPROv2 hit the baseline ceiling.** When your starting point is already extracting most of the "easy" performance, Bayesian search over prompt space can't find meaningful gains. The remaining errors require understanding *which specific intents* are confused, not just finding "a better prompt" in aggregate.

**GEPA's reflection mechanism broke through.** By explicitly diagnosing failure modes and generating targeted fixes, GEPA found the 15+ percentage points hiding in the tails of the distribution. This is the power of **error-driven learning**—instead of optimizing a scalar objective, you're solving a structured problem.

**LoRA remains the production gold standard.** When you have 9,000 labeled examples and need millisecond latency at zero marginal cost, fine-tuning a small model is unbeatable. The 1.4% gap between LoRA and GEPA might close with more GEPA iterations—but LoRA's inference economics are hard to argue with.

### The Data Efficiency Story

This chart tells the real story—GEPA achieves near-LoRA performance with a fraction of the data:

![Data Efficiency Curve](/blog_assets/plot_02_efficiency_curve.png)

### The Cost-Accuracy Tradeoff

When you factor in inference costs and latency, LoRA's self-hosted advantage becomes clear—but notice how GEPA occupies the same accuracy tier at API prices:

![Cost vs Accuracy Tradeoff](/blog_assets/plot_03_tradeoff_bubble.png)

---

## The Physics of Why This Happened

There's a deeper lesson here about optimization landscapes and the nature of the classification problem.

**The 77-class intent space is highly structured.** It's not 77 random buckets—it's a hierarchy of related concepts (card issues, transfer issues, account issues) with fine-grained distinctions within each cluster. This structure creates a specific kind of optimization challenge:

- **Global optimization** (finding the right general approach) is easy. That's why the baseline is 74.5%.
- **Local refinement** (distinguishing between semantically adjacent classes) is hard. That's where the remaining 25% lives.

MIPROv2 is designed for global optimization—it searches broadly across prompt space. GEPA is designed for local refinement—it zooms in on specific failure modes. LoRA, by directly adjusting the model's internal representations, can do both simultaneously.

This maps onto a familiar physics framework: **coarse-graining vs. fine-graining**. MIPROv2 optimizes the coarse-grained behavior. GEPA refines the fine-grained behavior. LoRA reshapes the underlying microstate distribution.

---

## Key Takeaways & Decision Framework

**LoRA is King for Production.** If you have the data (9,000+ examples), need 91%+ accuracy, and want millisecond latency at zero marginal inference cost, fine-tune a small model like Qwen or Llama. It's consistent, portable, and economically unbeatable once deployed.

**GEPA is the ROI Champion.** We gave GEPA just **50 training examples** and it achieved **90.0% accuracy**—within 1.4% of the fully fine-tuned model. If you're data-constrained but have access to a powerful model like Gemini 2.5 Flash, GEPA extracts remarkable performance through intelligent prompt evolution.

**Don't expect miracles from generic prompt optimization when your baseline is strong.** MIPROv2 is excellent when there's a large gap between naive prompting and optimal prompting. When zero-shot is already at 74.5%, you need error-driven approaches like GEPA to find the remaining performance.

### The Decision Tree

![Which Method Should You Use?](/blog_assets/plot_07_decision_tree.png)

---

## What's Next

A few threads we're pulling on:

1. **GEPA + LoRA hybrid**: Use GEPA to identify the hardest confusion pairs, then fine-tune a small model specifically on those cases. Best of both worlds?

2. **Active learning integration**: Can GEPA's reflection mechanism identify which *unlabeled* examples would be most valuable to annotate?

3. **Cross-domain transfer**: GEPA's discovered rules for Banking77 might transfer to similar financial intent datasets. LoRA's weights won't. There's a portability vs. performance trade-off worth exploring.
