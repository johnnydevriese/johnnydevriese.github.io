---
layout: post
title: "Looped Transformers Pass the Bucket"
date: 2026-11-02
categories: artificial-intelligence machine-learning research
slug: looped-transformers-pass-the-bucket
---

# Looped Transformers Pass the Bucket

*Part I of II. A looped transformer runs one block of shared weights over and over before it answers. Forget the fixed-point jargon for a moment. Think of a line of people passing buckets - and watch how far the buckets can travel.*

> ~~"A looped transformer is a weight-tied, recurrent-depth architecture that scales test-time compute by iterating a shared block toward a fixed point of its latent dynamics."~~

Every phrase in that sentence is defensible, and together they hide the two things you actually want to know: what one more loop does, and how many loops a given input needs. So let's put the definition down and build something you can hold in your hands instead - a line of people, each holding a bucket.

## 01 - Shared Weights, Changing State

A standard transformer is a stack of $N$ blocks, each with its own weights. Information passes through exactly $N$ of them on the way to the answer: no more, no fewer.

A looped transformer keeps one block (or a small group of layers) and applies it $r$ times. Each application is a **loop**. Write $e$ for the embedded input, $h_t$ for the state after $t$ loops, and $F_\theta$ for the shared block:

$$
h_{t+1} = F_\theta(h_t, e), \qquad t = 0, 1, \dots, r-1, \qquad \text{answer} = \mathrm{head}(h_r).
$$

Two things are worth keeping apart. The **weights** $\theta$ are fixed once training ends, and every loop uses the same ones. The **state** $h_t$ is a list of numbers the model computes for this particular input, and each loop replaces it with a new list. Nothing is written back into the weights. A loop is not learning; it is computing.

Notice the $e$ inside $F_\theta$. Feeding the input back in on every loop, called *input recall*, is a design choice, not a law. You could instead use the input only as the starting state, $h_0 = e$, and let the loop run on $h$ alone. Section 05 shows that this choice matters more than it looks.

<div class="rope-diagram-wrap">
<span class="caption">Figure 1 - One step of a language model, ordinary and looped</span>
<!-- fig:pipeline -->
<svg class="rope-diagram" viewBox="0 0 600 350" width="600" height="350">
<defs><marker id="pipe-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0.6 L8,4 L0,7.4 z" fill="var(--muted)"/></marker><marker id="pipe-loop" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0.6 L8,4 L0,7.4 z" fill="var(--accent)"/></marker></defs>
<text x="300.0" y="16.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9.5" fill="var(--ink)" text-anchor="middle" letter-spacing="1.5">AN ORDINARY LANGUAGE MODEL</text>
<text x="300.0" y="30.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >one pass through the stack per new token</text>
<text x="49.0" y="86.0" font-family="'Source Serif 4',serif" font-size="11.5" fill="var(--ink)" text-anchor="middle" font-style="italic">“the cat”</text>
<rect x="92.0" y="72.0" width="26.0" height="20.0" rx="4" fill="var(--paper)" stroke="var(--base-150)" stroke-width="0.9"/>
<text x="105.0" y="85.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >the</text>
<rect x="122.0" y="72.0" width="26.0" height="20.0" rx="4" fill="var(--paper)" stroke="var(--base-150)" stroke-width="0.9"/>
<text x="135.0" y="85.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >cat</text>
<rect x="166.0" y="62.0" width="46.0" height="40.0" rx="4" fill="color-mix(in srgb, var(--ink) 4.0%, var(--paper))" stroke="var(--base-150)" stroke-width="0.9"/>
<text x="189.0" y="85.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >embed</text>
<rect x="230.0" y="62.0" width="44.0" height="40.0" rx="4" fill="color-mix(in srgb, var(--ink) 5.0%, var(--paper))" stroke="var(--base-150)" stroke-width="0.9"/>
<text x="252.0" y="80.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >block 1</text>
<text x="252.0" y="92.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="6.5" fill="var(--muted)" text-anchor="middle" >attn · FFN</text>
<rect x="284.0" y="62.0" width="44.0" height="40.0" rx="4" fill="color-mix(in srgb, var(--ink) 11.0%, var(--paper))" stroke="var(--base-150)" stroke-width="0.9"/>
<text x="306.0" y="80.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >block 2</text>
<text x="306.0" y="92.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="6.5" fill="var(--muted)" text-anchor="middle" >attn · FFN</text>
<rect x="356.0" y="62.0" width="44.0" height="40.0" rx="4" fill="color-mix(in srgb, var(--ink) 17.0%, var(--paper))" stroke="var(--base-150)" stroke-width="0.9"/>
<text x="378.0" y="80.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >block N</text>
<text x="378.0" y="92.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="6.5" fill="var(--muted)" text-anchor="middle" >attn · FFN</text>
<line x1="274.0" y1="82.0" x2="284.0" y2="82.0" stroke="var(--muted)" stroke-width="1"/>
<text x="342.0" y="85.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="10" fill="var(--muted)" text-anchor="middle" >⋯</text>
<text x="313.0" y="117.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >N blocks, each with its own weights</text>
<rect x="414.0" y="62.0" width="50.0" height="40.0" rx="4" fill="color-mix(in srgb, var(--ink) 4.0%, var(--paper))" stroke="var(--base-150)" stroke-width="0.9"/>
<text x="439.0" y="85.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >LM head</text>
<rect x="485.0" y="96.0" width="6" height="6" fill="color-mix(in srgb, var(--ink) 30.0%, var(--paper))"/>
<rect x="494.0" y="92.0" width="6" height="10" fill="color-mix(in srgb, var(--ink) 30.0%, var(--paper))"/>
<rect x="503.0" y="72.0" width="6" height="30" fill="var(--accent)"/>
<rect x="512.0" y="94.0" width="6" height="8" fill="color-mix(in srgb, var(--ink) 30.0%, var(--paper))"/>
<rect x="521.0" y="98.0" width="6" height="4" fill="color-mix(in srgb, var(--ink) 30.0%, var(--paper))"/>
<line x1="482.0" y1="102.0" x2="528.0" y2="102.0" stroke="var(--base-150)" stroke-width="0.8"/>
<rect x="546.0" y="72.0" width="30.0" height="20.0" rx="4" fill="color-mix(in srgb, var(--accent) 14%, var(--paper))" stroke="var(--accent)" stroke-width="0.9"/>
<text x="561.0" y="85.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >sat</text>
<text x="49.0" y="52.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >text</text>
<text x="120.0" y="52.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >tokens</text>
<text x="505.0" y="52.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >logits</text>
<text x="561.0" y="52.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >next token</text>
<line x1="77.0" y1="82.0" x2="89.0" y2="82.0" stroke="var(--muted)" stroke-width="1" marker-end="url(#pipe-arrow)"/>
<line x1="151.0" y1="82.0" x2="163.0" y2="82.0" stroke="var(--muted)" stroke-width="1" marker-end="url(#pipe-arrow)"/>
<line x1="215.0" y1="82.0" x2="227.0" y2="82.0" stroke="var(--muted)" stroke-width="1" marker-end="url(#pipe-arrow)"/>
<line x1="403.0" y1="82.0" x2="411.0" y2="82.0" stroke="var(--muted)" stroke-width="1" marker-end="url(#pipe-arrow)"/>
<line x1="467.0" y1="82.0" x2="479.0" y2="82.0" stroke="var(--muted)" stroke-width="1" marker-end="url(#pipe-arrow)"/>
<line x1="531.0" y1="82.0" x2="543.0" y2="82.0" stroke="var(--muted)" stroke-width="1" marker-end="url(#pipe-arrow)"/>
<path d="M561.0,93.0 V122.0 Q561.0,128.0 555.0,128.0 H126.0 Q120.0,128.0 120.0,122.0 V95.0" fill="none" stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 2.5" marker-end="url(#pipe-arrow)"/>
<text x="340.5" y="141.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >append the new token and run the whole stack again</text>
<line x1="24" y1="162" x2="576" y2="162" stroke="var(--base-100)" stroke-width="0.8"/>
<text x="300.0" y="186.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9.5" fill="var(--ink)" text-anchor="middle" letter-spacing="1.5">A LOOPED TRANSFORMER</text>
<text x="300.0" y="200.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >one shared block, run r times before the token is read out</text>
<text x="49.0" y="280.0" font-family="'Source Serif 4',serif" font-size="11.5" fill="var(--ink)" text-anchor="middle" font-style="italic">“the cat”</text>
<rect x="92.0" y="266.0" width="26.0" height="20.0" rx="4" fill="var(--paper)" stroke="var(--base-150)" stroke-width="0.9"/>
<text x="105.0" y="279.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >the</text>
<rect x="122.0" y="266.0" width="26.0" height="20.0" rx="4" fill="var(--paper)" stroke="var(--base-150)" stroke-width="0.9"/>
<text x="135.0" y="279.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >cat</text>
<rect x="166.0" y="256.0" width="46.0" height="40.0" rx="4" fill="color-mix(in srgb, var(--ink) 4.0%, var(--paper))" stroke="var(--base-150)" stroke-width="0.9"/>
<text x="189.0" y="279.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >embed</text>
<rect x="281.0" y="256.0" width="64.0" height="40.0" rx="4" fill="color-mix(in srgb, var(--accent) 14%, var(--paper))" stroke="var(--accent)" stroke-width="0.9"/>
<text x="313.0" y="274.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >block</text>
<text x="313.0" y="286.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >attn · FFN</text>
<path d="M333.0,255.0 C333.0,222.0 293.0,222.0 293.0,253.0" fill="none" stroke="var(--accent)" stroke-width="1.4" marker-end="url(#pipe-loop)"/>
<text x="313.0" y="226.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--accent)" text-anchor="middle" >run r times</text>
<text x="313.0" y="311.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >same weights every pass</text>
<rect x="414.0" y="256.0" width="50.0" height="40.0" rx="4" fill="color-mix(in srgb, var(--ink) 4.0%, var(--paper))" stroke="var(--base-150)" stroke-width="0.9"/>
<text x="439.0" y="279.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >LM head</text>
<rect x="485.0" y="290.0" width="6" height="6" fill="color-mix(in srgb, var(--ink) 30.0%, var(--paper))"/>
<rect x="494.0" y="286.0" width="6" height="10" fill="color-mix(in srgb, var(--ink) 30.0%, var(--paper))"/>
<rect x="503.0" y="266.0" width="6" height="30" fill="var(--accent)"/>
<rect x="512.0" y="288.0" width="6" height="8" fill="color-mix(in srgb, var(--ink) 30.0%, var(--paper))"/>
<rect x="521.0" y="292.0" width="6" height="4" fill="color-mix(in srgb, var(--ink) 30.0%, var(--paper))"/>
<line x1="482.0" y1="296.0" x2="528.0" y2="296.0" stroke="var(--base-150)" stroke-width="0.8"/>
<rect x="546.0" y="266.0" width="30.0" height="20.0" rx="4" fill="color-mix(in srgb, var(--accent) 14%, var(--paper))" stroke="var(--accent)" stroke-width="0.9"/>
<text x="561.0" y="279.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >sat</text>
<line x1="77.0" y1="276.0" x2="89.0" y2="276.0" stroke="var(--muted)" stroke-width="1" marker-end="url(#pipe-arrow)"/>
<line x1="151.0" y1="276.0" x2="163.0" y2="276.0" stroke="var(--muted)" stroke-width="1" marker-end="url(#pipe-arrow)"/>
<line x1="215.0" y1="276.0" x2="278.0" y2="276.0" stroke="var(--muted)" stroke-width="1" marker-end="url(#pipe-arrow)"/>
<line x1="348.0" y1="276.0" x2="411.0" y2="276.0" stroke="var(--muted)" stroke-width="1" marker-end="url(#pipe-arrow)"/>
<line x1="467.0" y1="276.0" x2="479.0" y2="276.0" stroke="var(--muted)" stroke-width="1" marker-end="url(#pipe-arrow)"/>
<line x1="531.0" y1="276.0" x2="543.0" y2="276.0" stroke="var(--muted)" stroke-width="1" marker-end="url(#pipe-arrow)"/>
<path d="M561.0,287.0 V316.0 Q561.0,322.0 555.0,322.0 H126.0 Q120.0,322.0 120.0,316.0 V289.0" fill="none" stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 2.5" marker-end="url(#pipe-arrow)"/>
<text x="340.5" y="335.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >append it, run again: the r loops happen inside every step</text>
</svg>
<!-- /fig:pipeline -->
<span class="note">Top: tokens climb through N different blocks, the LM head scores every possible next token, and the winner is appended before the whole stack runs again. Bottom: one block runs r times before the readout. The extra compute goes into depth - no new tokens, no new weights.</span>
</div>

---

## 02 - Loops Are Not Thinking Tokens

The familiar way to spend more compute on a hard problem is **thinking tokens**: let the model write out intermediate steps before it answers. Each token buys one more trip through the whole stack, but it also lengthens the context, grows the KV cache, and forces the intermediate work into words.

A looped transformer spends its extra compute *inside* a single next-token step. It emits nothing extra, and its intermediate results are vectors, not text. That isn't free - every loop costs latency and FLOPs, and sharing weights cuts the parameter count without automatically cutting activation or cache memory - but the bill arrives in a different place.

Looping also changes what a fixed set of weights can compute. A stack of $N$ blocks always does exactly $N$ steps of work. A loop does as many steps as you choose to run, so the same weights can carry out a procedure whose number of steps grows with the input.

Which raises the question this whole series is about: **how many loops does a given input need?** Stop too early and the answer is wrong. Run too many and you burn compute. Ideally you would know the number before you spend it.

---

## 03 - The Analogy: A Bucket Brigade

Start with a toy that has nothing learned in it at all.

Line up 12 people, one per token. Each person holds a **bucket** - a number playing the role of that token's hidden state - and a slip of paper with their own **item**, playing the role of that token's input. Person $p$'s job is to end up holding the running sum of the items from person 1 through person $p$.

> **Twelve people, twelve items.**
>
> `items:` `1` `1` `1` `0` `0` `0` `0` `0` `0` `1` `1` `1`
>
> `running sums:` `1` `2` `3` `3` `3` `3` `3` `3` `3` `4` `5` `6`
>
> Every bucket starts full of random noise. Each round, everyone acts at once: **take the bucket your left neighbour held at the start of the round, add your own item, and hold the result.** Person 1 has no left neighbour, so they start each round from an empty bucket.

Items never move. Only buckets move, and each one picks up exactly one new item per round.

### A worked trace for the first four people

Before looking at all 12 people, trace the first four by hand. Their items are `1, 1, 1, 0`, so their finished buckets should be `1, 2, 3, 3`. In our seeded run, their buckets start at `0.35, 0.82, 0.33, -1.30`:

| Round $t$ | P1 (item `1`) | P2 (item `1`) | P3 (item `1`) | P4 (item `0`) | What happened on this round |
|---|---|---|---|---|---|
| 0 | 0.35 | 0.82 | 0.33 | -1.30 | Starting noise in every bucket |
| 1 | **1.00** | 1.35 | 1.82 | 0.33 | P1 takes $0 + 1 = 1$; P2 takes $0.35 + 1 = 1.35$; P1's noise (`0.35`) is now in P2's bucket |
| 2 | **1.00** | **2.00** | 2.35 | 1.82 | P2 takes P1's exact $1.00 + 1 = 2.00$; the `0.35` error has moved on to P3 |
| 3 | **1.00** | **2.00** | **3.00** | 2.35 | P3 takes P2's exact $2.00 + 1 = 3.00$; the `0.35` error has reached P4 |
| 4 | **1.00** | **2.00** | **3.00** | **3.00** | P4 takes P3's exact $3.00 + 0 = 3.00$; all four are solved |

Look at where that `0.35` went: P1 at round 0, P2 at round 1 (inside `1.35`), P3 at round 2 (inside `2.35`), P4 at round 3 (inside `2.35`), and off the right edge of our four-person window at round 4. Nobody damped it out. Each person simply handed it to the right and replaced it with a fresh sum from the left.

Now zoom out to all 12 people:

<div class="rope-diagram-wrap">
<span class="caption">Figure 2 - Twelve people, buckets started at noise, one round per row</span>
<!-- fig:brigade -->
<svg class="rope-diagram" viewBox="0 0 600 327" width="600" height="327">
<text x="110.0" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >P1</text>
<text x="110.0" y="36.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >item 1</text>
<text x="146.0" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >P2</text>
<text x="146.0" y="36.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >item 1</text>
<text x="182.0" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >P3</text>
<text x="182.0" y="36.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >item 1</text>
<text x="218.0" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >P4</text>
<text x="218.0" y="36.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >item 0</text>
<text x="254.0" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >P5</text>
<text x="254.0" y="36.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >item 0</text>
<text x="290.0" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >P6</text>
<text x="290.0" y="36.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >item 0</text>
<text x="326.0" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >P7</text>
<text x="326.0" y="36.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >item 0</text>
<text x="362.0" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >P8</text>
<text x="362.0" y="36.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >item 0</text>
<text x="398.0" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >P9</text>
<text x="398.0" y="36.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >item 0</text>
<text x="434.0" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >P10</text>
<text x="434.0" y="36.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >item 1</text>
<text x="470.0" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >P11</text>
<text x="470.0" y="36.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >item 1</text>
<text x="506.0" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >P12</text>
<text x="506.0" y="36.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" text-anchor="middle" >item 1</text>
<text x="82.0" y="64.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >round 0</text>
<rect x="93.0" y="53.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="110.0" y="64.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.3</text>
<rect x="129.0" y="53.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="146.0" y="64.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.8</text>
<rect x="165.0" y="53.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="182.0" y="64.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.3</text>
<rect x="201.0" y="53.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="218.0" y="64.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >-1.3</text>
<rect x="237.0" y="53.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="254.0" y="64.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.9</text>
<rect x="273.0" y="53.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="290.0" y="64.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.4</text>
<rect x="309.0" y="53.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="326.0" y="64.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >-0.5</text>
<rect x="345.0" y="53.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="362.0" y="64.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.6</text>
<rect x="381.0" y="53.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="398.0" y="64.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.4</text>
<rect x="417.0" y="53.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="434.0" y="64.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.3</text>
<rect x="453.0" y="53.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="470.0" y="64.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.0</text>
<rect x="489.0" y="53.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="506.0" y="64.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.5</text>
<text x="82.0" y="83.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >round 1</text>
<rect x="93.0" y="72.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="110.0" y="84.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >1</text>
<rect x="129.0" y="72.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="146.0" y="83.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.3</text>
<rect x="165.0" y="72.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="182.0" y="83.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.8</text>
<rect x="201.0" y="72.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="218.0" y="83.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.3</text>
<rect x="237.0" y="72.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="254.0" y="83.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >-1.3</text>
<rect x="273.0" y="72.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="290.0" y="83.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.9</text>
<rect x="309.0" y="72.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="326.0" y="83.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.4</text>
<rect x="345.0" y="72.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="362.0" y="83.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >-0.5</text>
<rect x="381.0" y="72.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="398.0" y="83.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.6</text>
<rect x="417.0" y="72.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="434.0" y="83.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.4</text>
<rect x="453.0" y="72.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="470.0" y="83.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.3</text>
<rect x="489.0" y="72.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="506.0" y="83.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.0</text>
<text x="82.0" y="102.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >round 2</text>
<rect x="93.0" y="91.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="110.0" y="103.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >1</text>
<rect x="129.0" y="91.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="146.0" y="103.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >2</text>
<rect x="165.0" y="91.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="182.0" y="102.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.3</text>
<rect x="201.0" y="91.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="218.0" y="102.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.8</text>
<rect x="237.0" y="91.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="254.0" y="102.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.3</text>
<rect x="273.0" y="91.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="290.0" y="102.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >-1.3</text>
<rect x="309.0" y="91.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="326.0" y="102.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.9</text>
<rect x="345.0" y="91.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="362.0" y="102.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.4</text>
<rect x="381.0" y="91.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="398.0" y="102.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >-0.5</text>
<rect x="417.0" y="91.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="434.0" y="102.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.6</text>
<rect x="453.0" y="91.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="470.0" y="102.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.4</text>
<rect x="489.0" y="91.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="506.0" y="102.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.3</text>
<text x="82.0" y="121.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >round 3</text>
<rect x="93.0" y="110.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="110.0" y="122.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >1</text>
<rect x="129.0" y="110.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="146.0" y="122.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >2</text>
<rect x="165.0" y="110.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="182.0" y="122.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="201.0" y="110.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="218.0" y="121.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.3</text>
<rect x="237.0" y="110.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="254.0" y="121.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.8</text>
<rect x="273.0" y="110.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="290.0" y="121.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.3</text>
<rect x="309.0" y="110.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="326.0" y="121.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >-1.3</text>
<rect x="345.0" y="110.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="362.0" y="121.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.9</text>
<rect x="381.0" y="110.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="398.0" y="121.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.4</text>
<rect x="417.0" y="110.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="434.0" y="121.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.5</text>
<rect x="453.0" y="110.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="470.0" y="121.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.6</text>
<rect x="489.0" y="110.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="506.0" y="121.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >3.4</text>
<text x="82.0" y="140.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >round 4</text>
<rect x="93.0" y="129.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="110.0" y="141.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >1</text>
<rect x="129.0" y="129.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="146.0" y="141.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >2</text>
<rect x="165.0" y="129.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="182.0" y="141.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="201.0" y="129.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="218.0" y="141.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="237.0" y="129.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="254.0" y="140.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.3</text>
<rect x="273.0" y="129.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="290.0" y="140.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.8</text>
<rect x="309.0" y="129.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="326.0" y="140.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.3</text>
<rect x="345.0" y="129.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="362.0" y="140.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >-1.3</text>
<rect x="381.0" y="129.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="398.0" y="140.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.9</text>
<rect x="417.0" y="129.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="434.0" y="140.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.4</text>
<rect x="453.0" y="129.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="470.0" y="140.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.5</text>
<rect x="489.0" y="129.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="506.0" y="140.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >3.6</text>
<text x="82.0" y="159.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >round 5</text>
<rect x="93.0" y="148.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="110.0" y="160.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >1</text>
<rect x="129.0" y="148.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="146.0" y="160.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >2</text>
<rect x="165.0" y="148.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="182.0" y="160.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="201.0" y="148.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="218.0" y="160.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="237.0" y="148.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="254.0" y="160.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="273.0" y="148.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="290.0" y="159.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.3</text>
<rect x="309.0" y="148.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="326.0" y="159.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.8</text>
<rect x="345.0" y="148.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="362.0" y="159.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.3</text>
<rect x="381.0" y="148.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="398.0" y="159.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >-1.3</text>
<rect x="417.0" y="148.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="434.0" y="159.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.9</text>
<rect x="453.0" y="148.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="470.0" y="159.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.4</text>
<rect x="489.0" y="148.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="506.0" y="159.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.5</text>
<text x="82.0" y="178.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >round 6</text>
<rect x="93.0" y="167.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="110.0" y="179.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >1</text>
<rect x="129.0" y="167.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="146.0" y="179.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >2</text>
<rect x="165.0" y="167.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="182.0" y="179.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="201.0" y="167.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="218.0" y="179.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="237.0" y="167.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="254.0" y="179.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="273.0" y="167.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="290.0" y="179.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="309.0" y="167.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="326.0" y="178.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.3</text>
<rect x="345.0" y="167.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="362.0" y="178.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.8</text>
<rect x="381.0" y="167.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="398.0" y="178.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.3</text>
<rect x="417.0" y="167.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="434.0" y="178.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >-0.3</text>
<rect x="453.0" y="167.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="470.0" y="178.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.9</text>
<rect x="489.0" y="167.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="506.0" y="178.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >3.4</text>
<text x="82.0" y="197.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >round 7</text>
<rect x="93.0" y="186.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="110.0" y="198.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >1</text>
<rect x="129.0" y="186.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="146.0" y="198.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >2</text>
<rect x="165.0" y="186.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="182.0" y="198.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="201.0" y="186.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="218.0" y="198.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="237.0" y="186.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="254.0" y="198.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="273.0" y="186.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="290.0" y="198.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="309.0" y="186.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="326.0" y="198.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="345.0" y="186.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="362.0" y="197.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.3</text>
<rect x="381.0" y="186.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="398.0" y="197.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.8</text>
<rect x="417.0" y="186.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="434.0" y="197.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.3</text>
<rect x="453.0" y="186.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="470.0" y="197.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >0.7</text>
<rect x="489.0" y="186.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="506.0" y="197.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >3.9</text>
<text x="82.0" y="216.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >round 8</text>
<rect x="93.0" y="205.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="110.0" y="217.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >1</text>
<rect x="129.0" y="205.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="146.0" y="217.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >2</text>
<rect x="165.0" y="205.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="182.0" y="217.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="201.0" y="205.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="218.0" y="217.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="237.0" y="205.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="254.0" y="217.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="273.0" y="205.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="290.0" y="217.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="309.0" y="205.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="326.0" y="217.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="345.0" y="205.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="362.0" y="217.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="381.0" y="205.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="398.0" y="216.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.3</text>
<rect x="417.0" y="205.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="434.0" y="216.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.8</text>
<rect x="453.0" y="205.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="470.0" y="216.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >2.3</text>
<rect x="489.0" y="205.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="506.0" y="216.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >1.7</text>
<text x="82.0" y="235.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >round 9</text>
<rect x="93.0" y="224.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="110.0" y="236.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >1</text>
<rect x="129.0" y="224.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="146.0" y="236.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >2</text>
<rect x="165.0" y="224.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="182.0" y="236.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="201.0" y="224.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="218.0" y="236.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="237.0" y="224.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="254.0" y="236.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="273.0" y="224.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="290.0" y="236.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="309.0" y="224.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="326.0" y="236.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="345.0" y="224.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="362.0" y="236.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="381.0" y="224.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="398.0" y="236.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="417.0" y="224.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="434.0" y="235.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >3.3</text>
<rect x="453.0" y="224.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="470.0" y="235.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >3.8</text>
<rect x="489.0" y="224.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="506.0" y="235.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >3.3</text>
<text x="82.0" y="254.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >round 10</text>
<rect x="93.0" y="243.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="110.0" y="255.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >1</text>
<rect x="129.0" y="243.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="146.0" y="255.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >2</text>
<rect x="165.0" y="243.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="182.0" y="255.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="201.0" y="243.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="218.0" y="255.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="237.0" y="243.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="254.0" y="255.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="273.0" y="243.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="290.0" y="255.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="309.0" y="243.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="326.0" y="255.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="345.0" y="243.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="362.0" y="255.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="381.0" y="243.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="398.0" y="255.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="417.0" y="243.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="434.0" y="255.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >4</text>
<rect x="453.0" y="243.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="470.0" y="254.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >4.3</text>
<rect x="489.0" y="243.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="506.0" y="254.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >4.8</text>
<text x="82.0" y="273.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >round 11</text>
<rect x="93.0" y="262.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="110.0" y="274.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >1</text>
<rect x="129.0" y="262.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="146.0" y="274.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >2</text>
<rect x="165.0" y="262.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="182.0" y="274.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="201.0" y="262.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="218.0" y="274.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="237.0" y="262.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="254.0" y="274.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="273.0" y="262.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="290.0" y="274.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="309.0" y="262.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="326.0" y="274.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="345.0" y="262.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="362.0" y="274.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="381.0" y="262.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="398.0" y="274.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="417.0" y="262.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="434.0" y="274.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >4</text>
<rect x="453.0" y="262.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="470.0" y="274.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >5</text>
<rect x="489.0" y="262.0" width="34.0" height="17.0" rx="3" fill="var(--base-100)"/>
<text x="506.0" y="273.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="7" fill="var(--muted)" text-anchor="middle" >5.3</text>
<text x="82.0" y="292.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >round 12</text>
<rect x="93.0" y="281.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="110.0" y="293.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >1</text>
<rect x="129.0" y="281.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="146.0" y="293.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >2</text>
<rect x="165.0" y="281.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="182.0" y="293.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="201.0" y="281.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="218.0" y="293.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="237.0" y="281.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="254.0" y="293.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="273.0" y="281.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="290.0" y="293.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="309.0" y="281.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="326.0" y="293.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="345.0" y="281.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="362.0" y="293.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="381.0" y="281.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="398.0" y="293.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >3</text>
<rect x="417.0" y="281.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="434.0" y="293.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >4</text>
<rect x="453.0" y="281.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="470.0" y="293.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >5</text>
<rect x="489.0" y="281.0" width="34.0" height="17.0" rx="3" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="506.0" y="293.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >6</text>
<path d="M92.0,52.0 L92.0,52.0 L92.0,71.0 L128.0,71.0 L128.0,90.0 L164.0,90.0 L164.0,109.0 L200.0,109.0 L200.0,128.0 L236.0,128.0 L236.0,147.0 L272.0,147.0 L272.0,166.0 L308.0,166.0 L308.0,185.0 L344.0,185.0 L344.0,204.0 L380.0,204.0 L380.0,223.0 L416.0,223.0 L416.0,242.0 L452.0,242.0 L452.0,261.0 L488.0,261.0 L488.0,280.0 L524.0,280.0 L524.0,299.0" fill="none" stroke="var(--accent)" stroke-width="1.8" stroke-dasharray="4 3"/>
<rect x="92.0" y="308.0" width="12" height="12" rx="2" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="110.0" y="318.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="start" >exact running total</text>
<rect x="252.0" y="308.0" width="12" height="12" rx="2" fill="var(--base-100)"/>
<text x="270.0" y="318.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="start" >leftover noise</text>
<line x1="382.0" y1="314.0" x2="404.0" y2="314.0" stroke="var(--accent)" stroke-width="1.8" stroke-dasharray="4 3"/>
<text x="410.0" y="318.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--accent)" text-anchor="start" >front: one person per round</text>
</svg>
<!-- /fig:brigade -->
<span class="note">Exact buckets fill in one person per round, left to right, pushing the starting noise off the far end. The line is done on round 12 and not a round before. After round 0, every grey number is the grey number up and to its left plus that person's item.</span>
</div>

Person 1 is exact after round 1. Person 2 takes that exact bucket in round 2, adds their own item, and is exact too. In general person $p$ is exact after round $p$, because their left neighbour was exact one round earlier. After 12 rounds everyone is exact, *whatever the starting noise was*.

That is a ceiling, not a count: a line that happens to start correct is already done at round 0. But starting from noise, the last person can't be right until the correct running sum has been carried all the way down the line.

### The brigade is a linear loop

The analogy doesn't merely resemble the math; it *is* the math. Write $S$ for the shift matrix that hands every bucket one place to the right: $(Sh)_p = h_{p-1}$, and $(Sh)_1 = 0$. One round is then

$$
h_{t+1} = S\,h_t + x .
$$

That is the looped-transformer equation from Section 01, with $F(h, x) = Sh + x$. Its settled state - the **fixed point**, where one more round changes nothing - satisfies $h^\ast = S h^\ast + x$, which component by component says $h^\ast_p = h^\ast_{p-1} + x_p$: the running sum. Subtract the two equations and the error $\delta_t = h_t - h^\ast$ obeys

$$
\delta_{t+1} = S\,\delta_t, \qquad (\delta_t)_p = (\delta_0)_{p-t}, \quad \text{which is zero once } t \ge p .
$$

The starting error does not shrink in place. It is carried $t$ places to the right and dumped off the end of the line. Since $S^n = 0$, after $n$ rounds there is nothing left to carry.

### Without recall, the brigade forgets

Now drop the $+x$. Hand everyone their item once, as the starting bucket ($h_0 = x$), and from then on only pass: $h_{t+1} = S\,h_t$. Every round the buckets march one place right and the last one falls off the end. After 12 rounds every bucket is zero.

Zero is a perfectly good fixed point - one more round of passing leaves it zero - and in this example it is wrong for every person in the line. The brigade has settled by forgetting its input. Hold on to that; it comes back in Section 05.

---

## 04 - enso: A Trained Brigade

The toy is exact, but nobody trains toys. So I trained a small looped transformer, **enso**, on a task with the same shape: about 100k parameters, small enough to take apart on a laptop.

The task is **prefix parity**: given a string of bits, output at every position whether the bits so far contain an odd number of ones (the running XOR). It is the brigade with XOR in place of addition, and it keeps the property that matters:

> **Prefix parity, five bits.**
>
> `input:` `1` `0` `1` `1` `0`
>
> `target:` `1` `1` `0` `1` `1`
>
> Flip the first bit and *every* output flips. For position 96 to be right, information from bit 1 has to reach position 96.

enso reads the whole bit string at once, starts its state from noise, re-reads the input on every loop, runs its shared block $r$ times, and reads out every position's answer at the end. The shared block is two causal attention layers, and each layer has an attention window of one: a position reads only itself and its left neighbour. That gives the model a built-in speed limit.

<div class="rope-diagram-wrap">
<span class="caption">Figure 3 - What one loop can reach: two layers, window one</span>
<!-- fig:reach -->
<svg class="rope-diagram" viewBox="0 0 600 272" width="600" height="272">
<defs><marker id="reach-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="var(--freq-slow)"/></marker></defs>
<text x="223.0" y="26.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >position 1</text>
<text x="287.0" y="26.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >position 2</text>
<text x="351.0" y="26.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >position 3</text>
<text x="415.0" y="26.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >position 4</text>
<text x="479.0" y="26.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >position 5</text>
<text x="543.0" y="26.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >position 6</text>
<text x="178.0" y="62.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="end" >start of loop</text>
<rect x="196.0" y="44.0" width="54.0" height="30.0" rx="4" fill="var(--base-100)"/>
<rect x="260.0" y="44.0" width="54.0" height="30.0" rx="4" fill="var(--base-100)"/>
<rect x="324.0" y="44.0" width="54.0" height="30.0" rx="4" fill="var(--freq-slow)" fill-opacity="0.22" stroke="var(--freq-slow)" stroke-width="0.8"/>
<rect x="388.0" y="44.0" width="54.0" height="30.0" rx="4" fill="var(--freq-slow)" fill-opacity="0.22" stroke="var(--freq-slow)" stroke-width="0.8"/>
<rect x="452.0" y="44.0" width="54.0" height="30.0" rx="4" fill="var(--freq-slow)" fill-opacity="0.22" stroke="var(--freq-slow)" stroke-width="0.8"/>
<rect x="516.0" y="44.0" width="54.0" height="30.0" rx="4" fill="var(--base-100)"/>
<text x="178.0" y="136.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="end" >after layer 1</text>
<rect x="196.0" y="118.0" width="54.0" height="30.0" rx="4" fill="var(--base-100)"/>
<rect x="260.0" y="118.0" width="54.0" height="30.0" rx="4" fill="var(--base-100)"/>
<rect x="324.0" y="118.0" width="54.0" height="30.0" rx="4" fill="var(--base-100)"/>
<rect x="388.0" y="118.0" width="54.0" height="30.0" rx="4" fill="var(--freq-slow)" fill-opacity="0.22" stroke="var(--freq-slow)" stroke-width="0.8"/>
<rect x="452.0" y="118.0" width="54.0" height="30.0" rx="4" fill="var(--freq-slow)" fill-opacity="0.22" stroke="var(--freq-slow)" stroke-width="0.8"/>
<rect x="516.0" y="118.0" width="54.0" height="30.0" rx="4" fill="var(--base-100)"/>
<text x="178.0" y="210.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="end" >after layer 2</text>
<rect x="196.0" y="192.0" width="54.0" height="30.0" rx="4" fill="var(--base-100)"/>
<rect x="260.0" y="192.0" width="54.0" height="30.0" rx="4" fill="var(--base-100)"/>
<rect x="324.0" y="192.0" width="54.0" height="30.0" rx="4" fill="var(--base-100)"/>
<rect x="388.0" y="192.0" width="54.0" height="30.0" rx="4" fill="var(--base-100)"/>
<rect x="452.0" y="192.0" width="54.0" height="30.0" rx="4" fill="var(--freq-slow)" fill-opacity="0.85"/>
<text x="479.0" y="210.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--paper)" text-anchor="middle" >output</text>
<rect x="516.0" y="192.0" width="54.0" height="30.0" rx="4" fill="var(--base-100)"/>
<line x1="351.0" y1="76.0" x2="415.0" y2="115.0" stroke="var(--freq-slow)" stroke-width="1.2" marker-end="url(#reach-arrow)"/>
<line x1="415.0" y1="76.0" x2="415.0" y2="115.0" stroke="var(--freq-slow)" stroke-width="1.2" marker-end="url(#reach-arrow)"/>
<line x1="415.0" y1="76.0" x2="479.0" y2="115.0" stroke="var(--freq-slow)" stroke-width="1.2" marker-end="url(#reach-arrow)"/>
<line x1="479.0" y1="76.0" x2="479.0" y2="115.0" stroke="var(--freq-slow)" stroke-width="1.2" marker-end="url(#reach-arrow)"/>
<line x1="415.0" y1="150.0" x2="479.0" y2="189.0" stroke="var(--freq-slow)" stroke-width="1.2" marker-end="url(#reach-arrow)"/>
<line x1="479.0" y1="150.0" x2="479.0" y2="189.0" stroke="var(--freq-slow)" stroke-width="1.2" marker-end="url(#reach-arrow)"/>
<text x="300.0" y="248.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--muted)" text-anchor="middle" >each layer: a position reads itself and its left neighbour</text>
<text x="300.0" y="262.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--accent)" text-anchor="middle" >two layers per loop: information moves at most two positions</text>
</svg>
<!-- /fig:reach -->
<span class="note">A schematic of the wiring, not data. Shaded boxes are everything output position 5 can depend on after one loop: positions 3 to 5 at the start of the loop. One loop moves information at most two positions to the right.</span>
</div>

It's a bucket brigade where everyone's arms reach two people. That two-positions-per-loop limit is a ceiling set by the wiring, not a promise that training reaches it.

enso trained on 12-bit strings only, with loop counts drawn at random up to 24. Then I handed it 96-bit strings - eight times longer than anything it had seen - and recorded which positions were right after every loop.

<div class="rope-diagram-wrap">
<span class="caption">Figure 4 - enso solving 96-bit parity, one loop per row</span>
<!-- fig:wavefront -->
<svg class="rope-diagram" viewBox="0 0 600 334" width="600" height="334">
<rect x="70.0" y="34.0" width="480.0" height="286.0" fill="var(--paper)" stroke="var(--base-150)"/>
<rect x="105.0" y="34.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="130.0" y="34.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="175.0" y="34.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="205.0" y="34.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="245.0" y="34.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="285.0" y="34.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="340.0" y="34.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="380.0" y="34.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="395.0" y="34.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="38.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="75.0" y="38.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="80.0" y="38.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.500"/>
<rect x="195.0" y="38.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="215.0" y="38.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="330.0" y="38.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="430.0" y="38.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="465.0" y="38.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="510.0" y="38.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="540.0" y="38.4" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="42.8" width="20.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="90.0" y="42.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.500"/>
<rect x="180.0" y="42.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="205.0" y="42.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="225.0" y="42.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="250.0" y="42.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="340.0" y="42.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="385.0" y="42.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="445.0" y="42.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="475.0" y="42.8" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="520.0" y="42.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="47.2" width="25.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="95.0" y="47.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="100.0" y="47.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.500"/>
<rect x="190.0" y="47.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="215.0" y="47.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="235.0" y="47.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="260.0" y="47.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="380.0" y="47.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="395.0" y="47.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="455.0" y="47.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="485.0" y="47.2" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="530.0" y="47.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="51.6" width="40.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="110.0" y="51.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.500"/>
<rect x="200.0" y="51.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="225.0" y="51.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="245.0" y="51.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="405.0" y="51.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="465.0" y="51.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="495.0" y="51.6" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="540.0" y="51.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="56.0" width="45.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="115.0" y="56.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="120.0" y="56.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.500"/>
<rect x="210.0" y="56.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="235.0" y="56.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="255.0" y="56.0" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="400.0" y="56.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="415.0" y="56.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="475.0" y="56.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="505.0" y="56.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="60.4" width="55.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="125.0" y="60.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="130.0" y="60.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.500"/>
<rect x="220.0" y="60.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="245.0" y="60.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="290.0" y="60.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="395.0" y="60.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="410.0" y="60.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="485.0" y="60.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="515.0" y="60.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="64.8" width="65.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="135.0" y="64.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="140.0" y="64.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.500"/>
<rect x="230.0" y="64.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="255.0" y="64.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="420.0" y="64.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="495.0" y="64.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="525.0" y="64.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="69.2" width="75.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="145.0" y="69.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="150.0" y="69.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.500"/>
<rect x="200.0" y="69.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="240.0" y="69.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="265.0" y="69.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="415.0" y="69.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="505.0" y="69.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="535.0" y="69.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="73.6" width="85.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="155.0" y="73.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="160.0" y="73.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="250.0" y="73.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="275.0" y="73.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="440.0" y="73.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="515.0" y="73.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="545.0" y="73.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="78.0" width="95.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="165.0" y="78.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="170.0" y="78.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="260.0" y="78.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="285.0" y="78.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="310.0" y="78.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="450.0" y="78.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="82.4" width="105.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="175.0" y="82.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="180.0" y="82.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="295.0" y="82.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="315.0" y="82.4" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="86.8" width="115.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="185.0" y="86.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="190.0" y="86.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="305.0" y="86.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="350.0" y="86.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="470.0" y="86.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="485.0" y="86.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="91.2" width="125.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="195.0" y="91.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="200.0" y="91.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="290.0" y="91.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="315.0" y="91.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="340.0" y="91.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="360.0" y="91.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="480.0" y="91.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="495.0" y="91.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="95.6" width="135.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="205.0" y="95.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="210.0" y="95.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="260.0" y="95.6" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="300.0" y="95.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="320.0" y="95.6" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="370.0" y="95.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="490.0" y="95.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="100.0" width="145.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="215.0" y="100.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="220.0" y="100.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="310.0" y="100.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="330.0" y="100.0" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="360.0" y="100.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="440.0" y="100.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="500.0" y="100.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="104.4" width="155.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="225.0" y="104.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="230.0" y="104.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="340.0" y="104.4" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="365.0" y="104.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="375.0" y="104.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="390.0" y="104.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="450.0" y="104.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="495.0" y="104.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="510.0" y="104.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="525.0" y="104.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="108.8" width="165.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="235.0" y="108.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="240.0" y="108.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="330.0" y="108.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="350.0" y="108.8" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="400.0" y="108.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="113.2" width="175.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="245.0" y="113.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="250.0" y="113.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="340.0" y="113.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="360.0" y="113.2" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="410.0" y="113.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="490.0" y="113.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="530.0" y="113.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="117.6" width="185.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="255.0" y="117.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="260.0" y="117.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="350.0" y="117.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="370.0" y="117.6" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="395.0" y="117.6" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="420.0" y="117.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="540.0" y="117.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="122.0" width="195.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="265.0" y="122.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="270.0" y="122.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="320.0" y="122.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="380.0" y="122.0" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="405.0" y="122.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="430.0" y="122.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="126.4" width="205.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="275.0" y="126.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="280.0" y="126.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="390.0" y="126.4" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="415.0" y="126.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="440.0" y="126.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="130.8" width="215.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="285.0" y="130.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="290.0" y="130.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="345.0" y="130.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="400.0" y="130.8" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="425.0" y="130.8" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="450.0" y="130.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="135.2" width="225.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="295.0" y="135.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="300.0" y="135.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="410.0" y="135.2" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="440.0" y="135.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="460.0" y="135.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="139.6" width="235.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="305.0" y="139.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="310.0" y="139.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="420.0" y="139.6" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="450.0" y="139.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="470.0" y="139.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="144.0" width="245.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="315.0" y="144.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="320.0" y="144.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.375"/>
<rect x="430.0" y="144.0" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="460.0" y="144.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="480.0" y="144.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="540.0" y="144.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="148.4" width="255.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="325.0" y="148.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="330.0" y="148.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="440.0" y="148.4" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="465.0" y="148.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="490.0" y="148.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="152.8" width="265.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="335.0" y="152.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="340.0" y="152.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="395.0" y="152.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="455.0" y="152.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="475.0" y="152.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="500.0" y="152.8" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="157.2" width="275.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="345.0" y="157.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="350.0" y="157.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="460.0" y="157.2" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="485.0" y="157.2" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="510.0" y="157.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="161.6" width="285.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="355.0" y="161.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="360.0" y="161.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="470.0" y="161.6" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="495.0" y="161.6" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="166.0" width="295.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="365.0" y="166.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="370.0" y="166.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="480.0" y="166.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="510.0" y="166.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="530.0" y="166.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="170.4" width="305.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="375.0" y="170.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="380.0" y="170.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="435.0" y="170.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="490.0" y="170.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="515.0" y="170.4" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="174.8" width="315.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="385.0" y="174.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="390.0" y="174.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="500.0" y="174.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="530.0" y="174.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="179.2" width="325.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="395.0" y="179.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="400.0" y="179.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="450.0" y="179.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="510.0" y="179.2" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="535.0" y="179.2" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="183.6" width="335.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="405.0" y="183.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="410.0" y="183.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="465.0" y="183.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="520.0" y="183.6" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="545.0" y="183.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="188.0" width="345.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="415.0" y="188.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="420.0" y="188.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="530.0" y="188.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="192.4" width="355.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="425.0" y="192.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="430.0" y="192.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="480.0" y="192.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="540.0" y="192.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="196.8" width="365.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="435.0" y="196.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="440.0" y="196.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="490.0" y="196.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="201.2" width="375.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="445.0" y="201.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="450.0" y="201.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="70.0" y="205.6" width="385.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="455.0" y="205.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="460.0" y="205.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="70.0" y="210.0" width="395.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="465.0" y="210.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="470.0" y="210.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="480.0" y="210.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="214.4" width="405.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="475.0" y="214.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="480.0" y="214.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="530.0" y="214.4" width="10.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="218.8" width="415.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="485.0" y="218.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="490.0" y="218.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="70.0" y="223.2" width="425.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="495.0" y="223.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="500.0" y="223.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="510.0" y="223.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.125"/>
<rect x="70.0" y="227.6" width="435.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="505.0" y="227.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="510.0" y="227.6" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="70.0" y="232.0" width="445.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="515.0" y="232.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="520.0" y="232.0" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="70.0" y="236.4" width="455.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="525.0" y="236.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="530.0" y="236.4" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="70.0" y="240.8" width="460.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="530.0" y="240.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="535.0" y="240.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="540.0" y="240.8" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.250"/>
<rect x="70.0" y="245.2" width="470.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="540.0" y="245.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.875"/>
<rect x="545.0" y="245.2" width="5.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="70.0" y="249.6" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="254.0" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="258.4" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="262.8" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="267.2" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="271.6" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="276.0" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="280.4" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="284.8" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="289.2" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="293.6" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="298.0" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="302.4" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="306.8" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="311.2" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="70.0" y="315.6" width="480.00" height="4.40" shape-rendering="crispEdges" fill="var(--freq-slow)" fill-opacity="1.000"/>
<path d="M70.0,42.8 L75.0,42.8 L75.0,42.8 L80.0,42.8 L80.0,47.2 L85.0,47.2 L85.0,47.2 L90.0,47.2 L90.0,51.6 L95.0,51.6 L95.0,51.6 L100.0,51.6 L100.0,51.6 L105.0,51.6 L105.0,56.0 L110.0,56.0 L110.0,60.4 L115.0,60.4 L115.0,60.4 L120.0,60.4 L120.0,64.8 L125.0,64.8 L125.0,64.8 L130.0,64.8 L130.0,64.8 L135.0,64.8 L135.0,69.2 L140.0,69.2 L140.0,73.6 L145.0,73.6 L145.0,73.6 L150.0,73.6 L150.0,78.0 L155.0,78.0 L155.0,78.0 L160.0,78.0 L160.0,82.4 L165.0,82.4 L165.0,82.4 L170.0,82.4 L170.0,82.4 L175.0,82.4 L175.0,86.8 L180.0,86.8 L180.0,86.8 L185.0,86.8 L185.0,91.2 L190.0,91.2 L190.0,91.2 L195.0,91.2 L195.0,95.6 L200.0,95.6 L200.0,100.0 L205.0,100.0 L205.0,104.4 L210.0,104.4 L210.0,104.4 L215.0,104.4 L215.0,104.4 L220.0,104.4 L220.0,108.8 L225.0,108.8 L225.0,108.8 L230.0,108.8 L230.0,113.2 L235.0,113.2 L235.0,113.2 L240.0,113.2 L240.0,117.6 L245.0,117.6 L245.0,117.6 L250.0,117.6 L250.0,122.0 L255.0,122.0 L255.0,122.0 L260.0,122.0 L260.0,126.4 L265.0,126.4 L265.0,126.4 L270.0,126.4 L270.0,130.8 L275.0,130.8 L275.0,130.8 L280.0,130.8 L280.0,135.2 L285.0,135.2 L285.0,135.2 L290.0,135.2 L290.0,139.6 L295.0,139.6 L295.0,139.6 L300.0,139.6 L300.0,144.0 L305.0,144.0 L305.0,144.0 L310.0,144.0 L310.0,148.4 L315.0,148.4 L315.0,148.4 L320.0,148.4 L320.0,152.8 L325.0,152.8 L325.0,152.8 L330.0,152.8 L330.0,157.2 L335.0,157.2 L335.0,157.2 L340.0,157.2 L340.0,161.6 L345.0,161.6 L345.0,161.6 L350.0,161.6 L350.0,166.0 L355.0,166.0 L355.0,166.0 L360.0,166.0 L360.0,170.4 L365.0,170.4 L365.0,170.4 L370.0,170.4 L370.0,174.8 L375.0,174.8 L375.0,174.8 L380.0,174.8 L380.0,179.2 L385.0,179.2 L385.0,179.2 L390.0,179.2 L390.0,183.6 L395.0,183.6 L395.0,188.0 L400.0,188.0 L400.0,188.0 L405.0,188.0 L405.0,188.0 L410.0,188.0 L410.0,192.4 L415.0,192.4 L415.0,192.4 L420.0,192.4 L420.0,196.8 L425.0,196.8 L425.0,196.8 L430.0,196.8 L430.0,201.2 L435.0,201.2 L435.0,201.2 L440.0,201.2 L440.0,205.6 L445.0,205.6 L445.0,205.6 L450.0,205.6 L450.0,210.0 L455.0,210.0 L455.0,210.0 L460.0,210.0 L460.0,214.4 L465.0,214.4 L465.0,214.4 L470.0,214.4 L470.0,218.8 L475.0,218.8 L475.0,218.8 L480.0,218.8 L480.0,223.2 L485.0,223.2 L485.0,223.2 L490.0,223.2 L490.0,227.6 L495.0,227.6 L495.0,227.6 L500.0,227.6 L500.0,232.0 L505.0,232.0 L505.0,232.0 L510.0,232.0 L510.0,236.4 L515.0,236.4 L515.0,236.4 L520.0,236.4 L520.0,240.8 L525.0,240.8 L525.0,240.8 L530.0,240.8 L530.0,245.2 L535.0,245.2 L535.0,245.2 L540.0,245.2 L540.0,249.6 L545.0,249.6 L545.0,249.6 L550.0,249.6" fill="none" stroke="var(--accent)" stroke-width="1.6"/>
<text x="72.5" y="26.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--muted)" text-anchor="middle" >1</text>
<text x="187.5" y="26.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--muted)" text-anchor="middle" >24</text>
<text x="307.5" y="26.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--muted)" text-anchor="middle" >48</text>
<text x="427.5" y="26.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--muted)" text-anchor="middle" >72</text>
<text x="547.5" y="26.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--muted)" text-anchor="middle" >96</text>
<text x="62.0" y="39.2" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--muted)" text-anchor="end" >0</text>
<text x="62.0" y="109.6" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--muted)" text-anchor="end" >16</text>
<text x="62.0" y="180.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--muted)" text-anchor="end" >32</text>
<text x="62.0" y="250.4" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--muted)" text-anchor="end" >48</text>
<text x="62.0" y="320.8" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--muted)" text-anchor="end" >64</text>
<text x="310.0" y="12.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >TOKEN POSITION (PERSON IN LINE) →</text>
<text x="24.0" y="177.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" transform="rotate(-90 24 177.0)">LOOP ↓</text>
<text x="400.0" y="130.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--muted)" text-anchor="middle" >coin-flip territory</text>
<text x="400.0" y="142.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >(accuracy ≈ 0.5)</text>
<text x="190.0" y="270.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="9" fill="var(--paper)" text-anchor="middle" >solved (accuracy ≥ 0.99)</text>
<text x="220.0" y="64.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--accent)" text-anchor="start" >red: the loop from which each</text>
<text x="220.0" y="75.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--accent)" text-anchor="start" >position stays solved, ≈ ⌈p/2⌉ + 1</text>
<rect x="564.0" y="156.0" width="12" height="14" fill="var(--freq-slow)" fill-opacity="0.000" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="564.0" y="142.0" width="12" height="14" fill="var(--freq-slow)" fill-opacity="0.125" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="564.0" y="128.0" width="12" height="14" fill="var(--freq-slow)" fill-opacity="0.250" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="564.0" y="114.0" width="12" height="14" fill="var(--freq-slow)" fill-opacity="0.375" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="564.0" y="100.0" width="12" height="14" fill="var(--freq-slow)" fill-opacity="0.500" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="564.0" y="86.0" width="12" height="14" fill="var(--freq-slow)" fill-opacity="0.625" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="564.0" y="72.0" width="12" height="14" fill="var(--freq-slow)" fill-opacity="0.750" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="564.0" y="58.0" width="12" height="14" fill="var(--freq-slow)" fill-opacity="0.875" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="564.0" y="44.0" width="12" height="14" fill="var(--freq-slow)" fill-opacity="1.000" stroke="var(--base-150)" stroke-width="0.4"/>
<text x="580.0" y="54.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="start" >1.0</text>
<text x="580.0" y="166.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="start" >0.5</text>
<text x="570.0" y="38.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >acc</text>
</svg>
<!-- /fig:wavefront -->
<span class="note">Accuracy at each position after each loop, averaged over 256 random strings. A front crosses the sequence at two positions per loop: behind it every position is solved, ahead of it the model is flipping coins. The red staircase marks the loop from which each position stays at 99% or better.</span>
</div>

A sharp diagonal front, two positions per loop. Behind it, solved; ahead of it, coin flips. Position $p$ settles at loop $\lceil p/2 \rceil + 1$ for 89 of 96 positions, and every position is within one loop of that. The brigade, learned.

The same rule holds across lengths. Counting a string as correct only if *every* position is right:

| Input length $n$ | $\lceil n/2 \rceil$ | First loop with at least 99% of strings exact | Strings exact at that loop |
|---|---|---|---|
| 12 | 6 | 7 | 256 of 256 |
| 24 | 12 | 14 | 255 of 256 |
| 48 | 24 | 26 | 256 of 256 |
| 96 | 48 | 50 | 254 of 256 |

> Loops needed ≈ the distance the answer has to travel ÷ how far it moves per loop, plus a loop or two of lag.

The lag is measured, not derived. Two more things stand out. enso trained with at most 24 loops and still solves 96 bits in 50. And running past the solve doesn't hurt: at all four lengths it stays at 99% or better through loop 192, eight times the training maximum, and is at 100% there. A plausible reading is that what it learned is a local rule, the same at every position, so more loops simply run more of the same procedure. That is a reading of one model, not a law of looping.

---

## 05 - The Trap: Settling Is Not Correctness

If you don't know a task's dependency chain in advance, how would you decide when to stop looping? The natural rule is to watch the state and halt once one more loop barely changes it, $h_{t+1} \approx h_t$. In other words: stop at the fixed point.

It is tempting to read "the loop has settled" as "the model has finished thinking." Here is why that's dangerous. Alongside enso I trained a sibling *without* input recall: the input is only its starting state, $h_0 = e$, and after that the loop runs on $h$ alone, $h_{t+1} = F_\theta(h_t)$. Same architecture, same training data.

<div class="rope-diagram-wrap">
<span class="caption">Figure 5 - How much each loop moves the state, 48-bit parity</span>
<!-- fig:residual -->
<svg class="rope-diagram" viewBox="0 0 600 260" width="600" height="260">
<rect x="70.0" y="20.0" width="480.0" height="200.0" fill="var(--paper)" stroke="var(--base-150)"/>
<line x1="70.0" y1="220.0" x2="550.0" y2="220.0" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="223.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >10⁻⁸</text>
<line x1="70.0" y1="197.8" x2="550.0" y2="197.8" stroke="var(--base-100)" stroke-width="0.6"/>
<line x1="70.0" y1="175.6" x2="550.0" y2="175.6" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="178.6" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >10⁻⁶</text>
<line x1="70.0" y1="153.3" x2="550.0" y2="153.3" stroke="var(--base-100)" stroke-width="0.6"/>
<line x1="70.0" y1="131.1" x2="550.0" y2="131.1" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="134.1" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >10⁻⁴</text>
<line x1="70.0" y1="108.9" x2="550.0" y2="108.9" stroke="var(--base-100)" stroke-width="0.6"/>
<line x1="70.0" y1="86.7" x2="550.0" y2="86.7" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="89.7" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >10⁻²</text>
<line x1="70.0" y1="64.4" x2="550.0" y2="64.4" stroke="var(--base-100)" stroke-width="0.6"/>
<line x1="70.0" y1="42.2" x2="550.0" y2="42.2" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="45.2" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >1</text>
<line x1="70.0" y1="20.0" x2="550.0" y2="20.0" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="70.0" y="234.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >1</text>
<text x="184.3" y="234.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >16</text>
<text x="306.2" y="234.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >32</text>
<text x="428.1" y="234.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >48</text>
<text x="550.0" y="234.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >64</text>
<text x="310.0" y="250.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >LOOP</text>
<text x="18.0" y="120.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" transform="rotate(-90 18 120.0)">STEP SIZE |h⁺ − h| / |h|</text>
<path d="M70.0,38.8 L77.6,41.4 L85.2,42.4 L92.9,42.2 L100.5,42.1 L108.1,42.2 L115.7,42.4 L123.3,42.7 L131.0,42.9 L138.6,43.2 L146.2,43.5 L153.8,43.9 L161.4,44.2 L169.0,44.5 L176.7,45.0 L184.3,45.5 L191.9,45.9 L199.5,46.5 L207.1,47.2 L214.8,47.9 L222.4,49.0 L230.0,50.1 L237.6,51.7 L245.2,54.1 L252.9,59.1 L260.5,63.0 L268.1,67.9 L275.7,75.4 L283.3,83.4 L291.0,90.6 L298.6,96.8 L306.2,102.8 L313.8,108.7 L321.4,113.8 L329.0,118.7 L336.7,124.1 L344.3,128.5 L351.9,133.8 L359.5,138.2 L367.1,143.3 L374.8,148.0 L382.4,152.1 L390.0,157.1 L397.6,162.1 L405.2,166.0 L412.9,170.5 L420.5,175.0 L428.1,178.7 L435.7,181.5 L443.3,183.3 L451.0,184.5 L458.6,185.1 L466.2,185.3 L473.8,185.6 L481.4,185.7 L489.0,185.7 L496.7,185.8 L504.3,185.8 L511.9,185.8 L519.5,185.8 L527.1,185.8 L534.8,185.7 L542.4,185.8 L550.0,185.8" fill="none" stroke="var(--freq-slow)" stroke-width="2"/>
<path d="M70.0,40.5 L77.6,47.7 L85.2,53.8 L92.9,66.2 L100.5,69.8 L108.1,80.7 L115.7,93.2 L123.3,105.2 L131.0,117.3 L138.6,129.3 L146.2,141.2 L153.8,153.1 L161.4,164.9 L169.0,176.6 L176.7,187.8 L184.3,196.8 L191.9,200.3 L199.5,200.6 L207.1,200.9 L214.8,200.9 L222.4,200.9 L230.0,200.8 L237.6,201.0 L245.2,200.9 L252.9,200.9 L260.5,200.9 L268.1,200.9 L275.7,200.8 L283.3,200.8 L291.0,200.8 L298.6,200.8 L306.2,200.7 L313.8,200.9 L321.4,200.9 L329.0,200.8 L336.7,200.7 L344.3,200.9 L351.9,200.7 L359.5,200.7 L367.1,200.9 L374.8,200.8 L382.4,200.7 L390.0,200.9 L397.6,200.7 L405.2,200.7 L412.9,200.6 L420.5,200.8 L428.1,200.8 L435.7,200.8 L443.3,200.9 L451.0,200.8 L458.6,200.8 L466.2,200.9 L473.8,201.0 L481.4,201.0 L489.0,200.8 L496.7,200.9 L504.3,200.9 L511.9,200.8 L519.5,201.0 L527.1,200.8 L534.8,200.8 L542.4,200.7 L550.0,200.8" fill="none" stroke="var(--freq-fast)" stroke-width="2"/>
<line x1="268.1" y1="20.0" x2="268.1" y2="220.0" stroke="var(--freq-slow)" stroke-dasharray="3 3" stroke-width="1"/>
<text x="272.1" y="32.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--freq-slow)" text-anchor="start" >loop 27: with recall, 99% of</text>
<text x="272.1" y="43.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--freq-slow)" text-anchor="start" >sequences exact, state still moving</text>
<text x="283.3" y="214.8" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--freq-fast)" text-anchor="start" >without recall: settled by loop 14, 0 of 256 exact</text>
<line x1="440.0" y1="76.0" x2="458.0" y2="76.0" stroke="var(--freq-slow)" stroke-width="2"/>
<text x="464.0" y="79.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="start" >with recall</text>
<line x1="440.0" y1="89.0" x2="458.0" y2="89.0" stroke="var(--freq-fast)" stroke-width="2"/>
<text x="464.0" y="92.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="start" >without recall</text>
</svg>
<!-- /fig:residual -->
<span class="note">Median relative step size over 256 strings, log scale. Without recall, the state stops moving early and gets every string wrong. With recall, the state is still moving when 99% of strings are already right, and settles only once the whole sequence is solved.</span>
</div>

Without recall (red), the change per loop drops below $10^{-6}$ by loop 14. The loop has found a rock-solid fixed point, and it gets **0 of 256** strings right. It is the brigade without the $+x$: settled, by washing out its input.

With recall (teal), the state keeps moving while the front crosses the sequence. In this batch 99% of strings are exact by loop 27, and the step size doesn't fall below $10^{-6}$ until loop 48, when all 256 are exact. Settling is what happens *after* the work is done, not a sign that it is.

One honest caveat: the comparison changes two things at once. The recall model starts from noise; the no-recall model starts from the embedded input. So the difference can't be pinned on recall alone. What it does show, cleanly, is the direction of failure: a small step size is evidence that the loop stopped moving, not that it found the answer. Any settling rule needs to be checked against accuracy on labelled inputs before it sets a loop budget.

---

## 06 - The Whole Brigade in Code

After five figures, it is worth seeing how little machinery this actually is. Here is the toy from Section 03, both versions, in plain NumPy:

<!-- code:brigade -->

```python
import numpy as np

n = 12
x = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=float)  # each person's own item
answer = np.cumsum(x)                         # running totals: 1 2 3 3 3 3 3 3 3 4 5 6
S = np.eye(n, k=-1)                           # hand-off: person p takes p-1's bucket

h = np.random.default_rng(1).normal(size=n)   # every bucket starts as noise
rounds = 0
while not np.allclose(h, answer):
    h, rounds = S @ h + x, rounds + 1         # take the bucket on your left, add your item

forgetful = x.copy()                          # no recall: items go in once, at the start
for _ in range(n):
    forgetful = S @ forgetful                 # keep passing, never add again

print(rounds, forgetful.any())                # 12 False
assert rounds == n and not forgetful.any()    # True: n rounds to solve; without recall, nothing is left
```

<!-- /code:brigade -->

That final assertion is the whole first half of this series in one line: starting from noise, a 12-person brigade with recall solves the running sum in exactly 12 rounds; take away the $+x$ and the same line settles to all zeros - still, stable, and wrong.

---

## 07 - The Fine Print

1. **The speed limit is enso's, not every model's.** The two-positions-per-loop front comes from window-one attention. Full causal attention, which looped language models such as Huginn and Ouro use, lets every position read every earlier position within a single layer, so there is no spatial distance to cross. What extra loops buy there is serial *depth* of composition - each pass can chain another step of reasoning onto the results of the last - which is a different kind of bucket line, and nothing here tests it.
2. **One model, one seed.** The enso numbers come from one trained checkpoint, evaluated on 256 random strings per length with a single evaluation seed and no confidence intervals. The one-to-two loops of lag above $\lceil n/2 \rceil$ is a measurement, not a constant.
3. **Extra loops help only if training made them useful.** Looping gives a model room to run a longer procedure. Whether it learns one that keeps making progress is up to training. enso's recipe - keep input recall on every pass so the input never washes out, and sample the loop count at random during training (as Huginn does) so no single depth is baked in - is what makes loop 50 work when training stopped at 24.
4. **Toy versus model.** The brigade's round counts are exact facts about arithmetic. The enso numbers are measurements of one small model. Neither is a measurement of a large language model.

---

## 08 - Coda: The Loop Is a Line

Strip away the notation and three correspondences remain:

| The Bucket | The Hand-Off | The Loop Count |
|---|---|---|
| is the hidden state | is attention to the left | is distance ÷ reach |

A looped transformer doesn't hold its answer in one place and polish it. It passes partial results down the line, one hand-off at a time, and the number of loops it needs is set by how far the answer has to travel.

Which leaves a puzzle. There is a standard tool for asking how fast a loop settles: linearize around the fixed point and read off the eigenvalues of the Jacobian. For enso that tool forecasts about four loops per factor of ten, and its formula has no input length in it at all. Yet enso needs 7 loops at 12 tokens and 50 at 96. Part II, *Eigenvalues Never See the Bucket Change Hands*, shows exactly where the tool goes blind.

### Further reading

- **Geiping et al., 2025** - [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171) - Huginn, a 3.5B recurrent-depth language model, and the random loop-count training recipe enso borrows so the bucket line keeps working past its training depth.
- **Zhu et al., 2025** - [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) - Ouro, looped language models pre-trained at multi-billion scale with full causal attention.
- **Saunshi et al., 2025** - [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416) - the formal case that $r$ passes of a shared block can simulate $r$-step serial chains of thought.
- **Dehghani et al., 2018** - [Universal Transformers](https://arxiv.org/abs/1807.03819) - the original weight-shared, recurrent-in-depth transformer, with a per-token halting rule.
