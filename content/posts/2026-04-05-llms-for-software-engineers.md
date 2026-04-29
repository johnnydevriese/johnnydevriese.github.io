---
layout: post
title: "Understanding Large Language Models: A Mental Model for Software Engineers"
date: 2026-04-05
categories: artificial-intelligence engineering education
slug: llms-for-software-engineers
---

# Understanding Large Language Models: A Mental Model for Software Engineers

## The fundamental insight

An LLM is a machine that reads text and predicts what comes next. That's the entire system in one sentence. Everything else—the apparent reasoning, the helpfulness, the creativity, the mistakes—emerges from this simple mechanic repeated billions of times during training and thousands of times during generation.

The elegance of this design is that a single objective (predict the next token) gives rise to surprisingly complex behavior. By seeing enough examples of how humans explain concepts, write code, answer questions, and structure arguments, the model learns to produce continuations that *look like* reasoning, even though it's fundamentally pattern matching at massive scale.

### Probability distributions, not single answers

The key detail: at each step, the model doesn't immediately commit to one next token. Instead, it computes a **probability distribution** over all possible tokens in its vocabulary (typically 50,000-100,000 tokens).

Consider this prompt:

> `The capital of France is`


The model might assign probabilities like:
| **Candidate token** | **Probability** |
| --- | --- |
| ` Paris` | 0.72 |
| ` Lyon` | 0.05 |
| ` Marseille` | 0.04 |
| ` London` | 0.01 |
| `*(49,996 others)`* | 0.18 |


**Greedy decoding** always picks the highest-probability token. **Sampling** occasionally selects lower-probability tokens, trading determinism for variety. Most production systems use sampling with temperature control to balance creativity and coherence.

This probabilistic nature explains several LLM behaviors:
- Why the same prompt can yield different outputs
- Why small prompt changes can have large effects (they shift the probability landscape)
- Why the model sometimes "changes its mind" mid-generation (early low-probability choices constrain later options)

### Your prompt shapes the probability manifold

Here's the mental model that makes prompt engineering intuitive: **your prompt positions the model in a high-dimensional space of possible continuations**.

![A visual intuition for the "probability manifold": prompts move you to regions where different continuations are more/less likely.](/blog_assets/llm_manifold_v4.png)


Imagine all possible text sequences as points in a vast manifold. A vague prompt leaves you in a broad, unfocused region:


> `Write code to process data.`


This could mean Python, JavaScript, SQL, or shell scripts. It could process CSVs, JSON, databases, or binary formats. The probability mass spreads across thousands of plausible but different continuations.

Now consider a detailed prompt:

> `Write a Python function using pandas that reads a CSV file,
filters rows where 'status' equals 'active', computes the mean
of the 'revenue' column, and returns it as a float. Treat missing
revenue values as zero.`


You've moved to a narrow region of the manifold where the model's training data contains similar patterns. The probability mass concentrates on continuations that match your specification.

**The practical upshot:**
- Specificity narrows the probability distribution
- Constraints (format, style, edge cases) geometrically restrict the space of valid continuations
- Examples are extremely efficient navigation tools—they directly show which manifold region you want
- Ambiguity allows the model to drift into "nearby" regions that may not match your intent

This perspective explains why few-shot prompting works: instead of verbally describing your desired output, you provide examples that position the model precisely where you need it.

*The generation loop: compute probabilities, sample a token, append it, repeat until done.*


## From text to numbers: the model's view

To understand how LLMs produce these probability distributions, we need to know what they actually see. Spoiler: it's not the text you typed.

### Tokenization: chunking text into digestible pieces

The model doesn't process characters or words directly. First, a **tokenizer** splits your text into chunks called **tokens**—typically word pieces like `" ing"`, `"un"`, or common short words. Each token maps to a unique integer called a **token ID**.

*Text preprocessing: tokenize, convert to IDs, look up embeddings.*


Example tokenization:
| **Original text** | **Tokens** | **Token IDs** |
| --- | --- | --- |
| `Hello world!` | `["Hello", " world", "!"]` | `[15496, 995, 0]` |
| `printf("hi");` | `["printf", "(", "\"hi\"", ")", ";"]` | `[22210, 7, 4072, 8, 13]` |


Notice that tokenizers often preserve spaces, punctuation, and case. This has practical implications:

**Why tokenization matters:**
- **Cost:** You pay per token, not per character. Longer tokens are more efficient.
- **Behavior:** Different tokenizations activate different patterns. `"New York"` vs `"New_York"` vs `"NewYork"` may tokenize differently and thus behave differently.
- **Context limits:** Models have maximum token counts (e.g., 8K, 128K tokens), not character counts.
- **Prompt engineering:** Small changes like adding a space (`"JSON:"` vs `"JSON :"`) can shift token boundaries and change outputs.

This explains why prompt engineering sometimes feels finicky—you're not just changing semantic meaning; you're changing the exact numerical sequence the model processes.

### Embeddings: giving tokens meaning

Token IDs are just vocabulary indices—meaningless integers. The model immediately looks each ID up in a learned table and replaces it with an **embedding vector**: a list of floating-point numbers (typically 768 to 12,288 dimensions depending on model size).

| **Token ID** | **Embedding (simplified)** |
| --- | --- |
| `15496` | $[0.12, -0.03, 0.44, ...] \in \mathbb{R}^{d}$ |
| `995` | $[-0.07, 0.22, 0.10, ...] \in \mathbb{R}^{d}$ |


Think of embeddings as coordinates in a high-dimensional space where semantic similarity corresponds to geometric proximity. Tokens that appear in similar contexts during training end up near each other:
- `"king"` near `"queen"`, `"monarch"`
- `"printf"` near `"cout"`, `"console.log"`
- `"happy"` near `"joyful"`, `"glad"`

This geometric representation is what lets the model do math with language—it can measure similarity, interpolate between concepts, and transform meanings through learned linear operations.

Critically, these embeddings are **learned during training**, not hand-coded. The model discovers useful representations by trying to predict next tokens billions of times. We'll see how in the training section.

## The transformer architecture: what the model computes

Now we understand the inputs (token embeddings). What does the model actually *do* with them to produce probability distributions over the next token?

The answer: it runs a sequence of transformations called **transformer layers**, each of which refines the representation of every token by looking at context and applying learned transformations.

### Attention: looking back to move forward

The key innovation in transformers is **attention**—a mechanism that lets each token selectively focus on earlier tokens to gather relevant information.

**Why this matters:** To predict the next token, you often need to look back at earlier context:


> `Sam gave Alex the keys because _he` was late._


To continue sensibly, the model must figure out what `he` refers to—Sam or Alex? Attention is the mechanism that lets it "look back" and connect `he` to the most relevant earlier name.

Or consider constraints:

> `Write a haiku about the ocean. It must mention "salt".`


To obey the constraint, the model needs to keep `"salt"` in focus while generating. Attention is how it maintains this focus across many tokens.

**How attention works (without the math):**

At each position $i$, attention computes relevance scores between token $i$ and all earlier tokens $1, 2, ..., i-1$. It then creates a weighted summary of the information from the most relevant earlier positions.

*Attention: score earlier tokens for relevance, then blend information from the highest-scoring ones.*


Think of it like writing an email: when you type the next sentence, you mentally highlight the earlier parts that matter (the recipient's name, the previous question, the deadline). Attention is the model's version of that selective focus.

**Multi-head attention:** In practice, models use multiple attention mechanisms (heads) in parallel, each learning to focus on different patterns. One head might specialize in connecting pronouns to nouns, another in tracking list items, another in matching opening and closing delimiters in code.

### Feed-forward networks: transforming representations

After attention gathers context, each token's representation passes through a **feed-forward network** (FFN), also called an MLP (multi-layer perceptron). This is a simple two-layer neural network:

$$
\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2
$$

In plain English: multiply by learned weights, apply a nonlinearity, multiply by more learned weights. This transforms and refines each token's representation based on patterns learned during training.

You can think of the FFN as a lookup table for patterns: "When I see this combination of features, output this other combination." Researchers have found that individual neurons in FFNs often correspond to interpretable concepts (e.g., one neuron fires for code-related tokens, another for negative sentiment).

### Stacking layers: building depth

The full model repeats this attention + FFN block many times—often 32, 48, or even 96 layers in large models. Each layer refines the representations, building increasingly abstract and contextual features.

*One transformer layer. Stack $L$ of these to build the complete model.*


**Residual connections** (the "$x \leftarrow x + ...$" operations) are a technical trick that helps gradients flow during training. They let information from early layers skip directly to later layers.

**LayerNorm** normalizes the representations to keep numbers in a reasonable range, which stabilizes training.

After all layers, the final representation at the last position passes through one more transformation: a linear projection that maps the high-dimensional representation to a vocabulary-sized vector of **logits** (unnormalized scores). A softmax turns these logits into the probability distribution we saw at the beginning.

### The big picture: forward pass

When you send a prompt to an LLM, here's what happens:


- **Tokenize:** Convert text $\rightarrow$ token IDs
- **Embed:** Look up token IDs $\rightarrow$ embedding vectors
- **Add position info:** Add positional encodings so the model knows token order
- **Transform:** Pass through $L$ layers of (attention + FFN)
- **Project:** Map final representation $\rightarrow$ vocabulary-sized logits
- **Softmax:** Convert logits $\rightarrow$ probability distribution
- **Sample:** Pick next token according to the distribution

This entire process is called the **forward pass**. Every token generation requires one forward pass.

## Training: how the model learns

Now we understand *what* the model computes. But where do all those weights in the attention mechanisms and FFNs come from? How does the model learn to produce useful probability distributions?

The answer: training. We show the model billions of examples and adjust its weights to make better predictions.

### The training objective: next-token prediction

Training is conceptually simple: give the model text with the answer masked, ask it to predict the masked token, then adjust the weights to make the correct token more likely.

**Example:**

> Input: `The capital of France is ____`
Target: `Paris`
Model outputs: `Paris` (72%), `Lyon` (5%), `Marseille` (4%), ...


The model already assigns high probability to the correct answer, so the loss is low. But if it had predicted:

> Model outputs: `Lyon` (40%), `Paris` (15%), `London` (10%), ...


The loss would be high, and we'd adjust the weights to increase the probability of `Paris`.

The standard loss function is **cross-entropy loss**, which measures how well the predicted distribution matches the true distribution (which is 100% on the correct token, 0% on all others).

### Backpropagation: tuning billions of knobs

Here's the remarkable part: modern LLMs have billions or even trillions of parameters (weights). How do we adjust all of them to reduce the loss?

The answer is **backpropagation** + **gradient descent**.

**The intuition:**

- Compute the loss (how wrong is the prediction?)
- For each weight, compute its **gradient**: how would changing this weight affect the loss? (Would increasing it make the loss go up or down, and by how much?)
- Adjust each weight slightly in the direction that decreases the loss
- Repeat millions of times

**Why this works:** Every operation in a transformer (matrix multiplies, softmax, ReLU) is differentiable. This means we can compute gradients efficiently using the chain rule, flowing backwards from the loss through all the layers.

*Training loop: forward pass to predict, compute loss, backward pass for gradients, update weights, repeat.*


**Concretely:** If a particular attention weight consistently increases the loss (makes predictions worse), backprop will compute a negative gradient for that weight, and gradient descent will decrease it. If a weight helps (makes predictions better), the gradient will be positive (or negative depending on convention), and we'll adjust it to amplify its effect.

**Why we need GPUs:** Computing gradients for billions of parameters across millions of training examples requires massive parallelism. GPUs excel at the matrix operations that dominate both forward and backward passes. Training large models requires hundreds or thousands of GPUs running for weeks or months.

**Batch training:** In practice, we don't update weights after every example. Instead, we accumulate gradients over a **batch** of examples (often thousands of sequences), then update once. This is more efficient and produces more stable gradient estimates.

### The three stages of training

Modern LLMs are trained in three phases, each building on the last:

*Three-stage training pipeline: pretraining, instruction tuning, preference optimization.*


**Stage 1: Pretraining (learn language patterns)**

The model sees a massive corpus—web pages, books, Wikipedia, code repositories, Q&A sites—typically hundreds of billions to trillions of tokens. The objective is pure next-token prediction on random slices of this corpus.

This teaches the model:
- Grammar, syntax, and language structure
- Common facts and associations (though not with perfect reliability)
- Programming patterns and idioms
- Stylistic conventions
- Correlations between concepts

After pretraining, the model is good at continuing text but doesn't know how to be helpful. It might continue "Explain quantum computing" with "to a five-year-old using only kitchen metaphors while..." rather than actually explaining.

**Stage 2: Instruction tuning (learn to be an assistant)**

The model is fine-tuned on carefully curated examples of instructions paired with high-quality responses:


> `Instruction: Summarize this article in three bullet points.`
`Response: [a concise, well-structured summary]`


This teaches the model to recognize task structures and produce helpful, relevant outputs rather than just plausible continuations.

**Stage 3: Preference optimization (learn what humans value)**

The final stage addresses a subtle problem: for many prompts, multiple responses are plausible. Some are more helpful, accurate, harmless, or well-formatted than others.

In this phase (often using RLHF—Reinforcement Learning from Human Feedback), the model sees pairs of responses to the same prompt, ranked by human preference. It's trained to favor the higher-ranked response.

This encourages:
- Conciseness and clarity
- Admitting uncertainty rather than guessing
- Following formatting instructions precisely
- Declining harmful or inappropriate requests
- Being truthful about capabilities and limitations

### Scale: why bigger is (often) better

Scaling laws show that model performance improves predictably with:
- **Model size:** More parameters let the model memorize more patterns
- **Data size:** More training tokens provide more diverse examples
- **Compute:** More GPU-hours allow longer training and larger batches

This is why state-of-the-art models keep getting larger (GPT-4 reportedly has trillions of parameters) and more expensive to train (hundreds of millions of dollars in compute).

## Inference: using the trained model

Once training is complete, using the model is straightforward: run the forward pass repeatedly, generating one token at a time.

### The generation loop


- Start with your prompt tokens
- Run forward pass $\rightarrow$ get probability distribution
- Sample next token according to the distribution
- Append the token to the sequence
- Repeat until hitting a stop condition (e.g., generated end-of-sequence token, hit length limit)

*Inference loop. The KV cache stores attention keys/values to avoid recomputing them for earlier tokens.*


**KV caching:** A crucial optimization. Without caching, we'd recompute attention for the entire sequence at every step—extremely wasteful. Instead, we store the keys and values from earlier tokens and only compute attention for the new token. This makes generation much faster.

### One token at a time: no planning ahead

This bears emphasizing because it explains many LLM behaviors: **the model does not plan ahead**. It doesn't compose the full response mentally and then type it out. It generates token by token, committing to each choice before knowing what comes next.

Example: if you prompt:

> `Translate to French: "cat"   Answer:`


Generation might proceed:
| **Step** | **Token added** | **Why** |
| --- | --- | --- |
| 1 | ` chat` | highest-prob continuation |
| 2 | `.` | natural sentence ending |


At step 1, the model hasn't "decided" to add a period—that comes later, after `chat` is already committed.

This is why you sometimes see models:
- Start confidently and then get stuck or contradict themselves
- Produce malformed JSON (they started a structure but couldn't close it properly)
- Fail to satisfy global constraints (like "answer in exactly 50 words")

The model only "sees" where it's been, not where it's going.

### Sampling strategies

Different sampling methods trade off between determinism and diversity:

**Greedy:** Always pick highest-probability token. Deterministic but can be repetitive.

**Temperature sampling:** Divide logits by temperature $T$ before softmax. Lower $T$ makes the distribution sharper (more deterministic), higher $T$ makes it more uniform (more random).

**Top-k sampling:** Only consider the $k$ most likely tokens. Prevents low-probability "tail" tokens from being selected.

**Nucleus (top-p) sampling:** Only consider tokens comprising the top $p$ probability mass. Adapts to distribution shape—sometimes more than $k$ tokens, sometimes fewer.

## Why things go wrong: failure modes

Understanding how LLMs work lets us predict their failure modes. A useful mental model: **the model tries to sound plausible, not to be correct**.

### Hallucinations: confidently making things up

**What it looks like:** The model gives a detailed, confident answer that's partially or completely wrong.

**Why it happens:** The model is a pattern-matching engine, not a fact database. When it lacks sufficient information, it still generates a plausible-sounding continuation based on patterns it saw during training. Since its training objective is "produce text that looks like the training data," it optimizes for sounding right, not being right.

**Example:**

> `User: When did Einstein win the Nobel Prize for relativity?`


Trick question: Einstein won the Nobel Prize for the photoelectric effect, not relativity. But a model that hasn't learned this nuance might confidently generate "1921" (the year he won, but for the wrong reason) or even confabulate a different year entirely.

**Mitigation strategies:**
- Ask for citations and verify them
- Provide needed facts in the prompt (or use retrieval—see RAG below)
- Explicitly give permission to say "I don't know"
- For critical facts, use external tools (search, databases) and have the model quote them
- Use structured outputs with validation

### Context rot: forgetting instructions

**What it looks like:** You set clear rules at the start, but after many turns, the model ignores them.

**Why it happens:** Attention is powerful but finite. In long prompts, earlier tokens become less salient. The model can only focus on so much, and recent text often dominates the attention weights.

**Example:**

> `System: Output JSON only. No extra words.`
`[... 15 messages later ...]`
`User: Return the result.`
`Assistant: Sure! Here is the JSON: {...\`}


The JSON itself might be correct, but the model violated the "no extra words" constraint because that instruction stopped being salient.

**Mitigation strategies:**
- Put critical instructions at the end (recency helps)
- Break long tasks into shorter steps
- Use structured output formats with programmatic validation
- Restate key constraints before critical outputs
- Keep system prompts concise

### Brittleness: small changes, big effects

Because the model processes tokens (not semantic units directly), small variations can have outsized effects:
- `"Answer:"` vs `"Answer: "` (trailing space)
- `"1."` vs `"1)"` (period vs parenthesis in lists)
- `"JSON"` vs `"json"` (case)

These tokenize differently and activate different patterns from training data. This is why prompt engineering can feel like alchemy—you're navigating a statistical landscape where tiny perturbations can shift you to different manifold regions.

### Output variability

The same prompt can produce different outputs because:
- Most systems use sampling (intentional randomness)
- Small floating-point differences in computation can cascade
- Different sampling seeds

If you need determinism, use temperature 0 (greedy decoding) or validate outputs programmatically rather than assuming consistency.

### Lack of true reasoning

The model produces text that *looks like* reasoning by pattern-matching on examples of human reasoning in its training data. But it doesn't maintain logical consistency in the way a formal system does.

Example failure modes:
- Self-contradiction over long outputs
- Correct-sounding but logically invalid arguments
- Difficulty with multi-step reasoning that requires backtracking
- Inability to recognize when a problem is impossible or has no solution

For tasks requiring genuine logical rigor, validate outputs with external tools (type checkers, theorem provers, unit tests).

## Making LLMs reliable: grounding and tools

LLMs are impressive pattern matchers, but to build production systems, we need to ground them in reality and constrain their outputs.

### Retrieval-Augmented Generation (RAG)

The core idea: before asking the LLM to answer, retrieve relevant documents and include them in the prompt. Now the model can answer by summarizing and quoting the evidence rather than guessing.

**The RAG workflow:**

- **Index:** Embed your document corpus using the same embedding model the LLM was trained with (or a compatible one)
- **Query:** Embed the user's question
- **Retrieve:** Find the top-$k$ most similar document chunks (using vector similarity, typically cosine distance)
- **Augment:** Construct a prompt like:
  `Given this evidence: [retrieved text]
  Answer the user's question: [question]`
- **Generate:** Let the model produce a response grounded in the evidence

*RAG pipeline: retrieve relevant evidence, then generate an answer grounded in it.*


**Example (customer support):**

Without RAG:

> `User: What's your refund policy?`
`Assistant: [hallucinates a plausible-sounding policy]`


With RAG:

> `User: What's your refund policy?`
`[System retrieves: "Refunds available within 30 days for annual plans with email confirmation..."]`
`Assistant: According to our policy, refunds are available within 30 days for annual plans. You'll need to email support for confirmation.`


RAG dramatically reduces hallucinations for factual questions because the model now works from source material rather than purely from training patterns.

### Tool use and function calling

Modern LLM systems often augment the model with external tools: calculators, code executors, databases, search engines, APIs.

**The pattern:**

- User asks a question requiring external data
- Model generates a structured request (e.g., JSON specifying a function call)
- System executes the function
- Result is added back to the prompt
- Model continues, now informed by real data

**Example (data analysis):**

> `User: What's the average revenue by product category?`
`[Model generates: {"function": "query_db", "sql": "SELECT category, AVG(revenue) FROM sales GROUP BY category"\`]}
`[System executes query, returns results]`
`Model: Based on the data, here are the average revenues: Electronics: $1,234, ...`


This turns LLMs from text generators into orchestrators of complex workflows.

### Constrained decoding and structured outputs

For system integration, free-form text is often inadequate. We need JSON, XML, or other structured formats.

**Approaches:**
- **Prompting:** Instruct the model to output JSON. Works often but not always.
- **Few-shot examples:** Show the model valid input-output pairs. More reliable.
- **Constrained decoding:** Only allow tokens that keep the output valid according to a schema. Guarantees structure but requires specialized inference infrastructure.
- **Post-processing:** Parse the output, validate, retry if malformed. Robust but slower.

Modern APIs often provide "function calling" or "JSON mode" features that make structured outputs more reliable.

## Practical engineering guidance

If you're building with LLMs, here's what matters:

### Design principles

**Treat outputs as untrusted.** Validate everything with tests, type checkers, linters, schemas, or compilation. Never deploy generated code without review. Never make critical decisions based solely on LLM outputs.

**Use retrieval for facts.** If you need the model to reference specific information, put that information in the prompt (directly or via RAG). Don't rely on memorization.

**Constrain outputs structurally.** Use JSON schemas, function-calling interfaces, or post-processing validation when integrating LLMs into systems. Don't parse free-form text if you can avoid it.

**Prefer small, focused tasks.** One giant prompt is harder to debug and more prone to context rot than several small, well-scoped prompts with validation between them.

**Understand the token boundary.** Costs scale with tokens. Behavior changes with tokenization. Learn your model's tokenizer quirks. Use token counters during development.

**Accept variability.** If you need deterministic outputs, use temperature 0 or validate outputs programmatically. Don't expect identical responses from identical prompts.

**Remember: it's pattern-matching, not reasoning.** Design your system assuming the model will sometimes fail in surprising ways. Build in guard rails, validation, and human oversight for critical paths.

### Prompt engineering best practices

**Be specific.** Vague prompts yield vague outputs. Specify format, style, edge case handling, and constraints explicitly.

**Use examples.** Few-shot prompting is often more effective than lengthy verbal descriptions. Show the model what you want.

**Structure your prompts.** Use clear sections: context, instructions, examples, constraints. Make the structure visually obvious (headings, delimiters).

**Put important instructions at the end.** Recency bias means later text often dominates attention. Restate critical constraints just before asking for output.

**Ask for step-by-step reasoning.** For complex tasks, prompt the model to show its work. This often improves accuracy (the model performs better when it "thinks out loud").

**Iterate.** Prompt engineering is empirical. Test variations, measure what works, refine.

### When to use (and not use) LLMs

**Good use cases:**
- Drafting, summarization, and rewriting
- Code generation for well-defined, common patterns
- Explanation and tutoring
- Classification and sentiment analysis (with validation)
- Extracting structure from unstructured text
- Brainstorming and ideation

**Bad use cases:**
- Critical calculations (use a calculator or symbolic math)
- Fact verification (use search and authoritative sources)
- Tasks requiring perfect consistency (use deterministic code)
- Real-time, low-latency requirements (inference is expensive)
- Safety-critical decisions without human oversight
- Tasks requiring genuine logical proof (use formal verification)

### Recognizing and recovering from broken context

One of the most common failure modes in LLM-assisted development is **context degradation**. You'll recognize this pattern:


> `User: Can you add feature X?`
`Assistant: Yes, of course! Let me try approach A...`
`[code doesn't work]`
`User: That didn't work.`
`Assistant: My apologies! Let me try approach B...`
`[still broken]`
`User: Still failing.`
`Assistant: I see the issue now. Let me try approach C...`


**What's happening:** The context has become polluted with failed attempts, incorrect assumptions, and accumulated misunderstandings. The model is now pattern-matching on a conversation history full of failures rather than working from a clean understanding of the goal.

**The fix: Reset and recontextualize**

When you notice this "yes of course" cycle:

- **Stop** the iteration immediately
- Ask the model: "Can you summarize what we're trying to accomplish and what constraints/requirements we have?"
- **Read that summary carefully**—look for wrong assumptions, misunderstood requirements, or accumulated errors
- Start a **fresh conversation** with:
- The corrected summary as context
- Clear statement of the actual goal
- Any relevant code or documentation


Think of it like a stack overflow in your own thinking—sometimes you need to pop back up to a clean state rather than debugging deeper into a broken frame.

**Why this works:** You're giving the model a clean probability landscape focused on the solution, not weighted down by the history of what didn't work. The fresh context lets it pattern-match on successful implementations rather than failed attempts.

### Maintaining a living context document

A pattern that dramatically improves code quality: **maintain a CLAUDE.md (or similar) file** in your repository that grows with your project.

**What to include:**
- Architectural decisions and patterns your project follows
- Code style preferences and conventions
- Common pitfalls and how to avoid them
- Dependencies and their usage patterns
- Testing strategies and requirements
- Deployment considerations
- Examples of well-written code from your codebase

**The workflow:**

- When starting work, provide CLAUDE.md as context
- When the model makes mistakes or doesn't follow your patterns, add a rule to CLAUDE.md
- When you discover better approaches, update CLAUDE.md
- Treat it as living documentation that codifies your project's knowledge

**Example entry:**

> `## Error Handling`

`Always use Result<T, E> types for fallible operations.`
`Never use unwrap() in production code—use proper error`
`propagation with ? operator.`

`Example:`
`fn load_config() -> Result<Config, ConfigError> {`
` let contents = fs::read_to_string("config.yaml")?;`
` serde_yaml::from_str(&contents)`
` .map_err(ConfigError::ParseError)`
`\`}


**Why this works:** Each session starts with high-quality, project-specific context rather than generic patterns. The model's probability distribution is immediately constrained to your preferred approaches. As the document grows, it becomes a force multiplier—new team members and new sessions benefit from accumulated wisdom.

We've found this approach transforms LLM output from "plausible but needs heavy editing" to "follows our conventions and rarely needs changes."

### The exploration-then-implementation pattern

For complex features, resist the urge to immediately ask for code. Instead, use a two-phase approach:

**Phase 1: Exploration (current context)**

- Describe the problem you're solving
- Ask questions: "What are the main approaches?" "What are the tradeoffs?" "What edge cases matter?"
- Discuss architecture: "How would this fit with our existing system?"
- Explore alternatives until you understand the solution space

**Phase 2: Implementation (fresh context)**

- Once you have clarity, ask: "Can you write a markdown plan for implementing this?"
- Review the plan carefully—this is your last checkpoint
- If the plan looks good, start a **new conversation** with:
- The plan document as primary context
- Relevant code files
- Your CLAUDE.md guidelines
- Ask for implementation following the plan

**Why this works:**
- **Separation of concerns:** Exploration naturally involves dead ends and tangents. Implementation requires focus.
- **Clean context:** The implementation context contains only the distilled plan, not the exploratory back-and-forth.
- **Human checkpoint:** Reviewing the plan lets you catch misunderstandings before they become code.
- **Better prompts:** A detailed plan is a much more specific prompt than "build feature X."

**Example markdown plan:**

> `# Implementation Plan: User Authentication`

`## Overview`
`Add JWT-based authentication with refresh tokens.`

`## Files to modify`
`- src/auth/mod.rs: Add token generation/validation`
`- src/middleware/auth.rs: Create authentication middleware`
`- src/routes/users.rs: Add login/logout endpoints`

`## Implementation steps`
`1. Add dependencies to Cargo.toml (jsonwebtoken, bcrypt)`
`2. Create TokenService for JWT operations`
`3. Implement AuthMiddleware using tower`
`4. Add POST /login and POST /logout routes`
`5. Add tests for happy path and error cases`

`## Edge cases to handle`
`- Expired tokens`
`- Invalid signatures`
`- Concurrent login from multiple devices`

`## Testing strategy`
`Unit tests for token validation, integration tests`
`for auth flow, manual testing with curl`


With this plan as context, the implementation conversation starts with perfect clarity about what to build and how to build it. No exploration noise, no wrong turns—just execution of a reviewed plan.

## Advanced patterns for code generation

Beyond the fundamental workflows, several specific patterns dramatically improve code quality and reduce debugging time.

### Iterative refinement: "make this better"

One of the most underrated patterns is letting the model refine its own output through multiple passes.

**Why this works (from first principles):**

When you prompt "make this better," you fundamentally change the task in the probability manifold:
- **Initial generation:** "write code that solves X"
- **Refinement:** "write code that solves X AND is better than this existing code"

The refinement task is actually harder—the model must understand the existing code, evaluate its quality, and generate improvements while maintaining correctness. But it's also more constrained: the working code provides a strong anchor point.

**The key insight:** Models are often better at *evaluating* code than generating it from scratch. Self-critique activates different training patterns than initial generation.

**The basic workflow:**

> `Round 1: "Write a function to parse CSV with error handling"`
`[model generates working but basic code]`

`Round 2: "Make this better"`
`[model adds: better error messages, edge cases, optimizations]`

`Round 3: "Make this better"`
`[model adds: logging, type hints, documentation]`


Each iteration compounds improvements. The model catches its own mistakes, awkward code, and missed optimizations.

**Directed refinement variants:**
- `"Make this better from a security perspective"`
- `"Make this better from a performance perspective"`
- `"Make this better from a maintainability perspective"`

Each perspective activates different training patterns, finding different improvements.

**Explicit critique variant:**

> `"First, critique this code—what are its weaknesses?
Then improve it addressing those issues."`


Making the reasoning explicit often catches more issues.

**When to use:**
- Initial code works but feels crude
- You know it could be better but aren't sure how
- Teaching the model your quality standards
- When you have time for multiple rounds

**When NOT to use:**
- Code is fundamentally wrong (broken context—start fresh)
- You need specific changes (be explicit instead)
- Late in a long conversation (context degradation)

**Why this is underrated:** Most people either accept first output (missing improvements) or keep asking for different approaches (wasting context). "Make this better" is the goldilocks zone—same approach, incremental refinement, compounding quality.

### Diff-based editing

For modifications to existing code, request diffs rather than full file rewrites.

**The problem with full regeneration:**
- Loses comments, formatting, edge cases you've added
- Model might "fix" things that weren't broken
- Harder to review large file dumps vs. specific changes
- Can introduce regressions in working code

**Better approach:**

> `Bad: "Update this file to add error handling"`
`Good: "Show me a diff that adds try-catch around the database calls"`


Diffs are easier to review, preserve context, and make changes explicit.

### Test-first generation

Generate tests before implementation.

**Why it works:**
- Tests are specifications—they force clarity about edge cases
- You validate tests are correct before code is generated
- Model is less likely to hallucinate when tests exist
- Creates immediate feedback loop

**Workflow:**

> `Phase 1: "Write pytest tests for a function that parses ISO dates,
 handling timezone offsets and invalid formats"`
`Phase 2: Review tests`
`Phase 3: "Now implement the function that makes these tests pass"`


### Decomposition with explicit interfaces

Break complex tasks into modules with strict contracts.

**Why it matters:**
- Each piece can be validated independently
- Type signatures prevent integration failures
- Can regenerate one piece without affecting others
- Forces you to think through architecture

**Example:**

> `"First, define the TypeScript interfaces for: AuthService,
UserRepository, and TokenManager. Show me just the interfaces
with JSDoc."`
`[review interfaces]`
`"Now implement UserRepository following this interface"`


### Validation layer pattern

Generate validators before generating data.

**The pattern:**

- Generate schema/validator (e.g., Zod, JSON Schema)
- Review and validate the schema itself
- Generate data/code that must conform to schema
- Programmatically validate outputs

**Example:**

> `"Write a Zod schema for the API response structure we discussed"`
`[review schema]`
`"Now generate 5 example API responses that match this schema"`
`[validate examples against schema programmatically]`


This creates a programmatic checkpoint—you can validate LLM outputs with LLM-generated validators.

### Constraint enumeration

Before generating code, force the model to list all constraints.

**Why it helps:**
- Model often forgets earlier constraints
- You might realize you forgot important constraints
- Creates explicit checklist for validation
- Makes requirements concrete

**Example:**

> `"Before writing the function, list all constraints:`
` - What inputs are valid/invalid?`
` - What should it return on error?`
` - What are the performance requirements?`
` - What edge cases must be handled?"`


### Style transfer from examples

Provide existing code as style reference, not just verbal instructions.

**Why it works:**
- "Write clean code" is vague; examples are precise
- Matches your actual codebase conventions
- Includes implicit patterns (error handling, naming, etc.)

**Example:**

> `"Here's an existing API handler from our codebase: [example code]
Write a new handler for /api/products following the same patterns"`


Examples position the model precisely in the probability manifold, capturing conventions that would take paragraphs to describe.

### Incremental complexity

Start with the simplest version, then add features one at a time.

**Benefits:**
- Each step can be tested and validated
- Easier to locate where bugs were introduced
- Model less likely to overcomplicate early
- You can stop when "good enough"

**Example:**

> `V1: "Write a function that validates email format"`
`V2: "Add MX record checking"`
`V3: "Add disposable email detection"`


### The confidence check

Ask the model to rate its confidence and explain uncertainty.

**Why it helps:**
- Model may flag potential issues
- Indicates when to double-check or use different approach
- Reveals assumptions
- Helps calibrate trust

**Example:**

> `"Rate your confidence (1-10) on this solution and explain
what you're least certain about"`


The model's uncertainty often correlates with actual problems.

### Rubber duck mode

Have the model explain back what you're trying to do before generating.

**Benefits:**
- Catches your own unclear thinking
- Model might spot issues you missed
- Creates shared understanding before code generation
- Much cheaper than debugging generated code

**Example:**

> `You: "I need to implement user session management with Redis"`
`Model: "Let me make sure I understand: [explanation]"`
`You: "Actually, I realize we need to handle concurrent logins..."`


### Combining patterns for maximum effectiveness

These patterns compound when used together. A robust workflow might look like:


- Start with CLAUDE.md context (living documentation)
- Use exploration-then-implementation (separate phases)
- Define interfaces first (decomposition)
- Generate tests before code (test-first)
- Use "make this better" 2-3 times (iterative refinement)
- Request diffs for changes (preserve context)
- Check confidence on complex parts (uncertainty awareness)
- Watch for "yes of course" cycles (context degradation)
- Reset with summary when needed (context recovery)

The key is understanding *why* each pattern works based on how LLMs actually function—token-by-token generation, probability distributions, attention mechanisms, and context limitations. With this understanding, you can adapt and combine patterns for your specific needs.

## Conclusion

The fundamental simplicity of LLMs—predict the next token—gives rise to remarkably complex and useful behavior. But understanding this simplicity is crucial for using LLMs effectively.

Key takeaways:
- LLMs are pattern-matching systems trained on next-token prediction
- Your prompt positions the model in a probability manifold; specificity narrows the distribution
- Attention lets the model focus on relevant context; layers build increasingly abstract representations
- Training via backpropagation adjusts billions of parameters to minimize prediction error
- The model generates one token at a time with no planning ahead
- Failure modes (hallucination, context rot, brittleness) follow naturally from this architecture
- Reliability requires grounding (RAG), validation, and appropriate use case selection

With this mental model, you can:
- Write better prompts by understanding how they shape probability distributions
- Debug failures by recognizing which architectural limitations are at play
- Design robust systems that use LLMs where they excel and avoid them where they don't
- Set realistic expectations for what these systems can and cannot do

LLMs are powerful tools, but they're tools—not magic, not intelligence, just very sophisticated autocomplete. Understanding how the sausage is made helps you use it effectively.
