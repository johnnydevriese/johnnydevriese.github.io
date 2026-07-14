---
layout: post
title: "RoPE Is Just a Clock with a Thousand Hands"
date: 2026-07-14
categories: artificial-intelligence machine-learning engineering
slug: rope-clock-thousand-hands
---

# RoPE Is Just a Clock with a Thousand Hands

*A visual guide to Rotary Position Embeddings. Forget the linear algebra for a moment. Think in rotations, frequencies, and resonance — the way a physicist would.*

> ~~"RoPE encodes absolute position by multiplying query and key vectors by a block-diagonal rotation matrix to preserve relative distance invariance in the inner-product attention kernel."~~

You have read that sentence. Perhaps three times. If you are an engineer at 2 a.m., staring at a context window that should work and doesn't, it has told you nothing. So let's throw it away and start from something you can hold in your hands — a clock.

## 01 — The Problem: Where Am I?

A Transformer, at birth, has no sense of order. Show it *"The dog bit the man"* and it will process every token at once — a simultaneous flash, with no notion that *dog* arrived before *man*. Each word floats in a vacuum, equidistant from every other. Sequence, for the Transformer, does not exist until we build it in.

Without positional encoding, these two sentences produce **identical** internal representations:

> `The` `dog` `bit` `the` `man`
>
> `The` `man` `bit` `the` `dog`

The model sees the same bag of tokens. It cannot distinguish the biter from the bitten. Positional encoding is what breaks this symmetry.

The first generation of solutions was blunt. BERT simply assigned each slot a learnable vector: *"You are position 5. Here is what position 5 looks like."* It worked the way memorising a phone book works — perfectly, until someone gives you a number that isn't in the book. A model trained on 512 positions had no idea what to do with position 513.

RoPE takes a different path entirely. It treats position not as a label to be memorised, but as a **moment in time** — and time, unlike a lookup table, has no upper bound.

## 02 — Analogy One: The Clock with a Thousand Hands

Here is where the machinery becomes physical. Take an embedding vector — say, 4,096 dimensions wide. RoPE pairs those dimensions off: the 1st with the 2nd, the 3rd with the 4th, all the way down. You now have 2,048 pairs. And each pair is no longer two disconnected numbers. It is a coordinate on a circle — an *angle*. A hand on a clock face.

A single token, then, is not a list of numbers. It is **a clock with 2,048 hands**, all mounted on the same spindle, all pointing in different directions.

*(One honest footnote: production models apply RoPE inside each attention head — 128 dimensions, so 64 hands per head — not across the full embedding width. The story is identical at either scale; we'll keep the bigger clock because it makes for better pictures.)*

Now comes the trick that makes the whole scheme sing: **every hand rotates at a different speed.**

<div class="rope-diagram-wrap">
      <span class="caption">Figure 1 — Three hands, three frequencies</span>
      <svg class="rope-diagram" viewBox="0 0 600 240" width="600" height="240">
        <g transform="translate(100,120)">
          <circle cx="0" cy="0" r="72" fill="none" stroke="var(--base-100)" stroke-width="1"/>
          <circle cx="0" cy="0" r="2.5" fill="var(--freq-fast)"/>
          <g stroke="var(--base-150)" stroke-width="0.7"><line x1="0" y1="-72" x2="0" y2="-66"/><line x1="72" y1="0" x2="66" y2="0"/><line x1="0" y1="72" x2="0" y2="66"/><line x1="-72" y1="0" x2="-66" y2="0"/></g>
          <g><line x1="0" y1="0" x2="0" y2="-58" stroke="var(--freq-fast)" stroke-width="2.2" stroke-linecap="round"/><circle cx="0" cy="-58" r="3" fill="var(--freq-fast)" opacity="0.7"/><animateTransform attributeName="transform" type="rotate" from="0 0 0" to="360 0 0" dur="3s" repeatCount="indefinite"/></g>
          <text y="100" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--freq-fast)" letter-spacing="0.08em">HIGH FREQ</text>
          <text y="112" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="9.5" fill="var(--muted)" font-style="italic">pair #1</text>
        </g>
        <g transform="translate(300,120)">
          <circle cx="0" cy="0" r="72" fill="none" stroke="var(--base-100)" stroke-width="1"/>
          <circle cx="0" cy="0" r="2.5" fill="var(--freq-mid)"/>
          <g stroke="var(--base-150)" stroke-width="0.7"><line x1="0" y1="-72" x2="0" y2="-66"/><line x1="72" y1="0" x2="66" y2="0"/><line x1="0" y1="72" x2="0" y2="66"/><line x1="-72" y1="0" x2="-66" y2="0"/></g>
          <g><line x1="0" y1="0" x2="0" y2="-58" stroke="var(--freq-mid)" stroke-width="2.2" stroke-linecap="round"/><circle cx="0" cy="-58" r="3" fill="var(--freq-mid)" opacity="0.7"/><animateTransform attributeName="transform" type="rotate" from="0 0 0" to="360 0 0" dur="8s" repeatCount="indefinite"/></g>
          <text y="100" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--freq-mid)" letter-spacing="0.08em">MID FREQ</text>
          <text y="112" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="9.5" fill="var(--muted)" font-style="italic">pair #1024</text>
        </g>
        <g transform="translate(500,120)">
          <circle cx="0" cy="0" r="72" fill="none" stroke="var(--base-100)" stroke-width="1"/>
          <circle cx="0" cy="0" r="2.5" fill="var(--freq-slow)"/>
          <g stroke="var(--base-150)" stroke-width="0.7"><line x1="0" y1="-72" x2="0" y2="-66"/><line x1="72" y1="0" x2="66" y2="0"/><line x1="0" y1="72" x2="0" y2="66"/><line x1="-72" y1="0" x2="-66" y2="0"/></g>
          <g><line x1="0" y1="0" x2="0" y2="-58" stroke="var(--freq-slow)" stroke-width="2.2" stroke-linecap="round"/><circle cx="0" cy="-58" r="3" fill="var(--freq-slow)" opacity="0.7"/><animateTransform attributeName="transform" type="rotate" from="0 0 0" to="360 0 0" dur="24s" repeatCount="indefinite"/></g>
          <text y="100" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--freq-slow)" letter-spacing="0.08em">LOW FREQ</text>
          <text y="112" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="9.5" fill="var(--muted)" font-style="italic">pair #2048</text>
        </g>
      </svg>
      <span class="note">Watch the hands. The leftmost completes a full revolution in seconds; the rightmost barely moves. Position is encoded in their combined arrangement.</span>
    </div>

### A worked example with real numbers

Let's make this completely concrete. Forget 4,096 dimensions — imagine a tiny embedding with just **4 dimensions**, which gives us **2 pairs**: two clock hands.

**Setup.** Suppose the token **"dog"** has the raw embedding `[1.0, 0.0, 1.0, 0.0]`. We split this into two pairs:

> Pair 1: `(1.0, 0.0)`   Pair 2: `(1.0, 0.0)`

Each pair is a point on a circle. Both hands start pointing straight to the right — the "3 o'clock" position.

Now RoPE assigns each pair a rotation speed. Pair 1 (the fast hand) rotates by 45° per position. Pair 2 (the slow hand) rotates by just 5° per position. When *"dog"* appears at different positions in the sequence, here is what happens:

<div class="rope-diagram-wrap">
      <span class="caption">Figure 2 — The same token at three different positions</span>
      <svg class="rope-diagram" viewBox="0 0 600 310" width="600" height="310">
        <!-- POSITION 0 -->
        <text x="100" y="18" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--ink)" font-weight="500">Position 0</text>
        <text x="100" y="32" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="8.5" fill="var(--muted)" font-style="italic">no rotation</text>
        <g transform="translate(68,108)">
          <circle r="50" fill="none" stroke="var(--base-100)" stroke-width="0.7"/>
          <line x1="0" y1="0" x2="42" y2="0" stroke="var(--accent)" stroke-width="2" stroke-linecap="round"/>
          <circle cx="42" cy="0" r="3" fill="var(--accent)" opacity="0.5"/><circle r="2" fill="var(--ink)"/>
          <text y="68" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--accent)">FAST 0°</text>
        </g>
        <g transform="translate(132,108)">
          <circle r="50" fill="none" stroke="var(--base-100)" stroke-width="0.7"/>
          <line x1="0" y1="0" x2="42" y2="0" stroke="var(--freq-slow)" stroke-width="2" stroke-linecap="round"/>
          <circle cx="42" cy="0" r="3" fill="var(--freq-slow)" opacity="0.5"/><circle r="2" fill="var(--ink)"/>
          <text y="68" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--freq-slow)">SLOW 0°</text>
        </g>
        <text x="100" y="198" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)">(1.0, 0.0, 1.0, 0.0)</text>

        <!-- POSITION 2 -->
        <text x="300" y="18" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--ink)" font-weight="500">Position 2</text>
        <text x="300" y="32" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="8.5" fill="var(--muted)" font-style="italic">fast: 2×45°=90°   slow: 2×5°=10°</text>
        <g transform="translate(268,108)">
          <circle r="50" fill="none" stroke="var(--base-100)" stroke-width="0.7"/>
          <line x1="0" y1="0" x2="0" y2="-42" stroke="var(--accent)" stroke-width="2" stroke-linecap="round"/>
          <circle cx="0" cy="-42" r="3" fill="var(--accent)" opacity="0.5"/><circle r="2" fill="var(--ink)"/>
          <text y="68" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--accent)">FAST 90°</text>
        </g>
        <g transform="translate(332,108)">
          <circle r="50" fill="none" stroke="var(--base-100)" stroke-width="0.7"/>
          <line x1="0" y1="0" x2="41.4" y2="-7.3" stroke="var(--freq-slow)" stroke-width="2" stroke-linecap="round"/>
          <circle cx="41.4" cy="-7.3" r="3" fill="var(--freq-slow)" opacity="0.5"/><circle r="2" fill="var(--ink)"/>
          <text y="68" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--freq-slow)">SLOW 10°</text>
        </g>
        <text x="300" y="198" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)">(0.0, 1.0, 0.98, 0.17)</text>

        <!-- POSITION 8 -->
        <text x="500" y="18" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--ink)" font-weight="500">Position 8</text>
        <text x="500" y="32" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="8.5" fill="var(--muted)" font-style="italic">fast: 8×45°=360°   slow: 8×5°=40°</text>
        <g transform="translate(468,108)">
          <circle r="50" fill="none" stroke="var(--base-100)" stroke-width="0.7"/>
          <line x1="0" y1="0" x2="42" y2="0" stroke="var(--accent)" stroke-width="2" stroke-linecap="round"/>
          <circle cx="42" cy="0" r="3" fill="var(--accent)" opacity="0.5"/><circle r="2" fill="var(--ink)"/>
          <text y="68" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--accent)">FAST 360°</text>
        </g>
        <g transform="translate(532,108)">
          <circle r="50" fill="none" stroke="var(--base-100)" stroke-width="0.7"/>
          <line x1="0" y1="0" x2="32.2" y2="-27" stroke="var(--freq-slow)" stroke-width="2" stroke-linecap="round"/>
          <circle cx="32.2" cy="-27" r="3" fill="var(--freq-slow)" opacity="0.5"/><circle r="2" fill="var(--ink)"/>
          <text y="68" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--freq-slow)">SLOW 40°</text>
        </g>
        <text x="500" y="198" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)">(1.0, 0.0, 0.77, 0.64)</text>

        <line x1="40" y1="225" x2="560" y2="225" stroke="var(--dim)" stroke-width="0.5"/>
        <text x="300" y="250" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="10.5" fill="var(--ink)">Notice: at position 8 the fast hand has completed a full revolution and returned to 0°.</text>
        <text x="300" y="268" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="10.5" fill="var(--ink)">But the slow hand has only reached 40°. The slow hand is what distinguishes position 8 from position 0.</text>
        <text x="300" y="292" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="10.5" fill="var(--accent)" font-style="italic" font-weight="600">This is exactly how the hour hand distinguishes 3:00 AM from 3:00 PM.</text>
      </svg>
    </div>

This is the key insight: the fast pairs cycle rapidly and resolve fine-grained, nearby positions — they can tell apart token 4 from token 5. The slow pairs change glacially and resolve coarse, large-scale distances — they can tell apart token 100 from token 10,000. Working together, the full set of pairs can uniquely identify any position, no matter how far into the sequence.

### Encoding a position

When the model encounters the 500th token in a sequence, it advances every hand to the angle corresponding to *time = 500*. The fastest hand — one radian per token — has already whipped through almost eighty full revolutions; its reading alone is hopelessly ambiguous, like glancing at a second hand with no other context. The slowest hand has drifted barely three degrees off noon. You would scarcely notice it had moved at all.

But the constellation of all 2,048 angles together — some clustered, some flung to opposite sides of the dial — is unique to this position and no other. It is a fingerprint written in geometry, and it changes with every tick of the sequence. No two moments look alike, for the same reason no two instants on a real clock look alike once you read every hand at once.

---

## 03 — Analogy Two: Harmonic Resonance

Why build a clock at all? Why not just tag each token with an integer and move on?

Because the Transformer doesn't read positions. It computes **attention** — and attention is a dot product. The Query of one token, multiplied against the Key of another. In the language of physics, a dot product measures alignment: how much do these two vectors point the same way?

When both vectors have first been rotated by RoPE, something elegant falls out of the algebra. The absolute angles disappear. What remains in the dot product is *only the difference in rotation* — the phase gap between the two tokens. The model never learns where a token sits. It learns how far apart two tokens are.

> Think of two tokens as two bells.
> Attention is the sound they make
> when you ring them together.

<div class="rope-diagram-wrap">
      <span class="caption">Figure 3 — Constructive vs. destructive interference</span>
      <svg class="rope-diagram" viewBox="0 0 600 280" width="600" height="280">
        <defs>
          <linearGradient id="gC" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="var(--constructive)" stop-opacity="0"/><stop offset="50%" stop-color="var(--constructive)" stop-opacity="0.12"/><stop offset="100%" stop-color="var(--constructive)" stop-opacity="0"/></linearGradient>
          <linearGradient id="gD" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="var(--destructive)" stop-opacity="0"/><stop offset="50%" stop-color="var(--destructive)" stop-opacity="0.12"/><stop offset="100%" stop-color="var(--destructive)" stop-opacity="0"/></linearGradient>
        </defs>
        <rect x="20" y="15" width="260" height="120" rx="4" fill="url(#gC)"/>
        <text x="150" y="10" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="8" fill="var(--constructive)" letter-spacing="0.12em">CONSTRUCTIVE — NEARBY TOKENS</text>
        <path class="wave-path" d="M40,75 C60,35 80,35 100,75 C120,115 140,115 160,75 C180,35 200,35 220,75 C240,115 256,115 260,95" fill="none" stroke="var(--constructive)" stroke-width="2" opacity="0.6"/>
        <path class="wave-path delay" d="M40,75 C60,38 80,38 100,75 C120,112 140,112 160,75 C180,38 200,38 220,75 C240,112 256,112 260,92" fill="none" stroke="var(--constructive)" stroke-width="2" opacity="0.9"/>
        <circle class="pulse-dot" cx="160" cy="35" fill="var(--constructive)"/>
        <text x="150" y="148" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="10.5" fill="var(--muted)" font-style="italic">Waves align → signal amplified → high attention</text>

        <rect x="320" y="15" width="260" height="120" rx="4" fill="url(#gD)"/>
        <text x="450" y="10" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="8" fill="var(--destructive)" letter-spacing="0.12em">DESTRUCTIVE — DISTANT TOKENS</text>
        <path class="wave-path" d="M340,75 C360,35 380,35 400,75 C420,115 440,115 460,75 C480,35 500,35 520,75 C540,115 556,115 560,95" fill="none" stroke="var(--destructive)" stroke-width="2" opacity="0.6"/>
        <path class="wave-path delay" d="M340,75 C360,112 380,112 400,75 C420,38 440,38 460,75 C480,112 500,112 520,75 C540,38 556,38 560,58" fill="none" stroke="var(--destructive)" stroke-width="2" opacity="0.9"/>
        <text x="450" y="148" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="10.5" fill="var(--muted)" font-style="italic">Waves oppose → signal cancels → low attention</text>

        <line x1="20" y1="180" x2="580" y2="180" stroke="var(--dim)" stroke-width="0.5"/>
        <text x="300" y="198" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)" letter-spacing="0.1em">FREQUENCY DECOMPOSITION OF ATTENTION</text>
        <path d="M40,245 Q150,210 300,245 Q450,280 560,245" fill="none" stroke="var(--freq-slow)" stroke-width="1.8" opacity="0.7"/>
        <text x="574" y="249" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--freq-slow)">LOW</text>
        <path d="M40,245 C80,225 120,265 160,245 C200,225 240,265 280,245 C320,225 360,265 400,245 C440,225 480,265 520,245 C540,235 555,252 560,245" fill="none" stroke="var(--freq-mid)" stroke-width="1.4" opacity="0.55"/>
        <text x="574" y="236" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--freq-mid)">MID</text>
        <path d="M40,245 C50,235 60,255 70,245 C80,235 90,255 100,245 C110,235 120,255 130,245 C140,235 150,255 160,245 C170,235 180,255 190,245 C200,235 210,255 220,245 C230,235 240,255 250,245 C260,235 270,255 280,245 C290,235 300,255 310,245 C320,235 330,255 340,245 C350,235 360,255 370,245 C380,235 390,255 400,245 C410,235 420,255 430,245 C440,235 450,255 460,245 C470,235 480,255 490,245 C500,235 510,255 520,245 C530,235 540,255 550,245 C555,240 558,248 560,245" fill="none" stroke="var(--freq-fast)" stroke-width="1" opacity="0.4"/>
        <text x="574" y="223" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--freq-fast)">HIGH</text>
      </svg>
      <span class="note">The attention score between two tokens is the sum of all these harmonics. Low frequencies track long-range structure; high frequencies resolve nearby detail.</span>
    </div>

### A concrete example: who attends to whom?

Let's return to our tiny 4D world. Suppose we have three tokens in a sequence:

> `pos 0: The` `pos 1: cat` `pos 2: sat`

We want to know: when **"sat"** (position 2) looks back at the other tokens, how much attention does it pay to each one?

After RoPE rotates the vectors, the dot product between "sat" and each earlier token depends entirely on the **angular gap** between them — not their content, just their distance. Here is the intuition in numbers:

| | "sat" → "cat" (1 step apart) | "sat" → "The" (2 steps apart) |
|---|---|---|
| **Score** | **0.85** | **0.49** |
| Fast pair gap | 45° | 90° |
| Slow pair gap | 5° | 10° |
| Arithmetic | (cos 45° + cos 5°) / 2 ≈ 0.85 | (cos 90° + cos 10°) / 2 ≈ 0.49 |
| Why | Both pairs mostly aligned → high score | Fast pair fully orthogonal → score drops |

With this toy embedding, each pair contributes exactly the cosine of its angular gap, so you can check both numbers on a pocket calculator. In a real model with 2,048 pairs at varying speeds, this distance signal is far richer — a high-dimensional chord that the model learns to interpret. Nearby tokens tend to score higher by default, which gives the Transformer a natural bias toward local context. But the model is free to override this: if "sat" is semantically related to a distant token, the content-based component of the dot product can still push the score up.

Suppose token A sits at position 100 and token B at position 105. The gap is small. Their slowest-spinning hands — the low frequencies — are nearly identical in angle, and those harmonics reinforce each other: constructive interference, a clean tone. Their fastest-spinning hands may have landed on opposite sides of the dial, cancelling out: destructive interference, silence. The attention mechanism hears the resulting chord — a particular blend of reinforcement and cancellation across all 2,048 frequencies — and that chord *is* the distance. Five steps, expressed as sound.

---

## 04 — The Mathematics: RoPE as a Fourier Basis

We have been speaking in analogies — clocks, bells, chords. Now let's open the hood and look at the algebra. What follows is the mathematical backbone that makes the intuition precise. If you are comfortable with Fourier analysis, this section will feel like coming home.

### A one-paragraph Fourier refresher

Fourier's central insight is that **any sufficiently well-behaved function can be decomposed into a sum of sinusoids**, each oscillating at a different frequency. A square wave, a heartbeat, a vowel sound — all of them are secretly a stack of sine and cosine waves superimposed on one another. The frequencies form the *basis*; the amplitudes and phases form the *coefficients*. To reconstruct the original signal, you simply add the components back together.

<div class="rope-diagram-wrap">
      <span class="caption">Figure 4 — Fourier decomposition: a complex signal is a sum of simple sinusoids</span>
      <svg class="rope-diagram" viewBox="0 0 600 260" width="600" height="260">
        <text x="26" y="14" font-family="'JetBrains Mono',monospace" font-size="7.5" fill="var(--ink)" letter-spacing="0.08em">COMPOSITE SIGNAL</text>
        <path d="M30,55 C45,25 55,30 70,55 C80,72 85,68 95,45 C105,22 115,35 130,55 C140,68 150,72 165,55 C175,38 185,25 200,55 C210,72 220,68 235,45 C245,22 255,35 270,55 C280,68 290,72 305,55 C315,38 325,25 340,55 C350,72 360,68 375,45 C385,22 395,35 410,55 C420,68 430,72 445,55 C455,38 465,25 480,55 C490,72 500,68 515,45 C525,30 535,35 545,55" fill="none" stroke="var(--ink)" stroke-width="2"/>
        <text x="300" y="95" text-anchor="middle" font-family="'Instrument Serif',serif" font-size="18" fill="var(--dim)">=</text>
        <text x="26" y="120" font-family="'JetBrains Mono',monospace" font-size="6.5" fill="var(--freq-slow)" letter-spacing="0.08em">ω₁ (LOW FREQUENCY)</text>
        <path d="M30,155 Q160,120 300,155 Q440,190 545,155" fill="none" stroke="var(--freq-slow)" stroke-width="1.8" opacity="0.8"/>
        <text x="300" y="182" text-anchor="middle" font-family="'Instrument Serif',serif" font-size="16" fill="var(--dim)">+</text>
        <text x="26" y="195" font-family="'JetBrains Mono',monospace" font-size="6.5" fill="var(--freq-mid)" letter-spacing="0.08em">ω₂ (MID FREQUENCY)</text>
        <path d="M30,220 C65,200 100,240 135,220 C170,200 205,240 240,220 C275,200 310,240 345,220 C380,200 415,240 450,220 C485,200 520,240 545,220" fill="none" stroke="var(--freq-mid)" stroke-width="1.5" opacity="0.7"/>
        <text x="300" y="247" text-anchor="middle" font-family="'Instrument Serif',serif" font-size="16" fill="var(--dim)">+  ···</text>
      </svg>
      <span class="note">Any signal — no matter how complex — can be rebuilt as a weighted sum of sinusoids. This is Fourier's theorem. RoPE exploits this principle to encode position.</span>
    </div>

### The rotation matrix

Now, how does RoPE actually encode position? For each dimension pair $k$ (what we called "clock hand $k$"), RoPE defines a 2×2 rotation matrix. If a token sits at position $m$, pair $k$ is rotated by angle $m \cdot \omega_k$:

$$
R_k(m) = \begin{pmatrix} \cos(m\omega_k) & -\sin(m\omega_k) \\ \sin(m\omega_k) & \cos(m\omega_k) \end{pmatrix}
$$

This is a standard 2D rotation by angle $m\omega_k$. Nothing exotic — the same matrix you'd use to rotate a point on a plane.

The full RoPE operation applies this rotation independently to each of the $d/2$ pairs, forming a block-diagonal matrix. But all the interesting structure lives in the scalar $\omega_k$ — the frequency assigned to each pair.

### The frequency schedule

Here is where RoPE makes its most important design choice. The frequencies are not uniform. They are **geometrically spaced**:

$$
\omega_k = \frac{1}{10000^{2k/d}}
$$

where $k = 0, 1, 2, \ldots, d/2 - 1$ is the pair index and $d$ is the embedding dimension (e.g. 4096).

This means the first pair rotates fast (high $\omega$) and the last pair rotates slow (low $\omega$). More precisely, the frequencies span several orders of magnitude in a smooth geometric progression:

| Pair index k | Frequency ω<sub>k</sub> | Period (tokens per full rotation) | Analogy |
|---|---|---|---|
| 0 | 1.0 | ≈ 6.3 | Second hand |
| 512 | 0.1 | ≈ 63 | Minute hand |
| 1024 | 0.01 | ≈ 628 | Hour hand |
| 1536 | 0.001 | ≈ 6,283 | Day hand |
| 2047 | ≈ 0.0001 | ≈ 62,800 | Calendar hand |

The geometric spacing gives **log-uniform coverage** of the frequency spectrum — analogous to the mel scale in audio processing or octaves in music. Each octave of frequency gets roughly the same number of dimension pairs devoted to it. Notice the ladder: every 512 steps of $k$ slows the hand by exactly a factor of ten.

Why geometric and not linear? A linearly spaced set of frequencies would waste most of its resolution on either local or global patterns. Geometric spacing ensures the model has equally fine discrimination at every scale — it can distinguish positions 5 apart just as precisely as positions 5,000 apart, because different pairs "own" different ranges of distance.

<div class="rope-diagram-wrap">
      <span class="caption">Figure 5 — Log-scale frequency coverage: each pair covers a different range of distance</span>
      <svg class="rope-diagram" viewBox="0 0 600 180" width="600" height="180">
        <line x1="50" y1="130" x2="560" y2="130" stroke="var(--ink)" stroke-width="1"/>
        <text x="50" y="155" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--muted)">1</text>
        <text x="177" y="155" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--muted)">10</text>
        <text x="305" y="155" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--muted)">100</text>
        <text x="432" y="155" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--muted)">1,000</text>
        <text x="560" y="155" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--muted)">10,000</text>
        <text x="305" y="175" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="9" fill="var(--muted)" font-style="italic">Distance between tokens (log scale)</text>
        <line x1="50" y1="130" x2="50" y2="135" stroke="var(--ink)" stroke-width="0.7"/><line x1="177" y1="130" x2="177" y2="135" stroke="var(--ink)" stroke-width="0.7"/><line x1="305" y1="130" x2="305" y2="135" stroke="var(--ink)" stroke-width="0.7"/><line x1="432" y1="130" x2="432" y2="135" stroke="var(--ink)" stroke-width="0.7"/><line x1="560" y1="130" x2="560" y2="135" stroke="var(--ink)" stroke-width="0.7"/>

        <rect x="50" y="55" width="155" height="65" rx="3" fill="var(--freq-fast)" opacity="0.08"/>
        <rect x="50" y="55" width="155" height="65" rx="3" fill="none" stroke="var(--freq-fast)" stroke-width="1" opacity="0.3"/>
        <text x="127" y="45" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--freq-fast)" letter-spacing="0.08em">PAIRS 0–512</text>
        <text x="127" y="92" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="9" fill="var(--freq-fast)" font-style="italic">Resolve local</text>
        <text x="127" y="105" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="9" fill="var(--freq-fast)" font-style="italic">syntax &amp; grammar</text>

        <rect x="177" y="55" width="180" height="65" rx="3" fill="var(--freq-mid)" opacity="0.06"/>
        <rect x="177" y="55" width="180" height="65" rx="3" fill="none" stroke="var(--freq-mid)" stroke-width="1" opacity="0.3"/>
        <text x="267" y="45" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--freq-mid)" letter-spacing="0.08em">PAIRS 512–1024</text>
        <text x="267" y="92" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="9" fill="var(--freq-mid)" font-style="italic">Resolve paragraph</text>
        <text x="267" y="105" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="9" fill="var(--freq-mid)" font-style="italic">&amp; sentence structure</text>

        <rect x="355" y="55" width="205" height="65" rx="3" fill="var(--freq-slow)" opacity="0.06"/>
        <rect x="355" y="55" width="205" height="65" rx="3" fill="none" stroke="var(--freq-slow)" stroke-width="1" opacity="0.3"/>
        <text x="457" y="45" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7" fill="var(--freq-slow)" letter-spacing="0.08em">PAIRS 1024–2048</text>
        <text x="457" y="92" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="9" fill="var(--freq-slow)" font-style="italic">Resolve document</text>
        <text x="457" y="105" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="9" fill="var(--freq-slow)" font-style="italic">&amp; cross-passage</text>

        <line x1="50" y1="32" x2="560" y2="32" stroke="var(--dim)" stroke-width="0.7" stroke-dasharray="3,3"/>
        <text x="305" y="26" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="6.5" fill="var(--muted)" letter-spacing="0.1em">EQUAL NUMBER OF PAIRS PER DECADE → EQUAL RESOLUTION AT EVERY SCALE</text>
      </svg>
    </div>

### The key derivation: attention as a Fourier series

Now we arrive at the mathematical heart of RoPE. The attention score between a Query at position $m$ and a Key at position $n$ involves the dot product of their rotated vectors. Let's write this out for a single pair $k$:

**Step 1 — Apply rotation to both vectors.** The query pair $(q_{2k}, q_{2k+1})$ is rotated by angle $m \cdot \omega_k$. The key pair $(k_{2k}, k_{2k+1})$ is rotated by angle $n \cdot \omega_k$.

**Step 2 — Compute the dot product of the rotated pair.** Using the identity that the dot product of two rotated 2D vectors depends only on the *difference* of their rotation angles, the contribution from pair $k$ works out to:

$$
(q_{2k}k_{2k} + q_{2k+1}k_{2k+1})\cos(\omega_k \Delta) + (q_{2k+1}k_{2k} - q_{2k}k_{2k+1})\sin(\omega_k \Delta)
$$

where $\Delta = m - n$ is the relative position — the only positional quantity that survives.

**Step 3 — Sum over all pairs.** The full attention dot product is the sum over all $d/2$ pairs:

$$
q_m^\top k_n = \sum_k \big[\, a_k \cos(\omega_k \Delta) + b_k \sin(\omega_k \Delta) \,\big]
$$

Read that sum carefully. The coefficients $a_k$ and $b_k$ are determined by the *content* of the query and key (the meaning of the tokens). The sinusoids are functions of $\omega_k \cdot \Delta$ — the frequency times the *relative distance*.

> This is a Fourier series in Δ.
> The attention kernel is, literally, a trigonometric polynomial in the relative position.

This is the deep reason the clock analogy works: RoPE doesn't merely *resemble* Fourier analysis — it *is* Fourier analysis. The dimension pairs are the harmonic basis functions. The token embeddings supply the coefficients. And the position difference is the variable over which the series is evaluated.

### Why this is powerful

Fourier's theorem guarantees that *any smooth, periodic function* can be approximated to arbitrary precision by a sufficiently long trigonometric sum. Applied to RoPE, this means the attention mechanism can learn **any distance-dependent pattern** the data requires. The model is not locked into a fixed decay curve or a fixed window. By adjusting the query and key weights (which determine $a_k$ and $b_k$), each attention head can sculpt its own distance kernel:

- **Local attention** — high-frequency coefficients dominate, producing a sharp peak at small Δ. The head attends mostly to its neighbours. Useful for syntax and local agreement.
- **Periodic attention** — a single frequency dominates, producing a repeating pattern. Useful for structured data like code indentation or tabular formats.
- **Long-range decay** — low-frequency coefficients dominate, producing a broad, slowly decaying kernel. Useful for document-level coherence and coreference.

All three patterns — and infinitely many others — emerge from the same mechanism. The Fourier basis is universal; the learned coefficients specialise it.

<div class="rope-diagram-wrap">
      <span class="caption">Figure 6 — Three kernels, one mechanism</span>
      <svg class="rope-diagram" viewBox="0 0 600 185" width="600" height="185">
        <text x="110" y="20" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--freq-fast)" letter-spacing="0.1em">LOCAL</text>
        <text x="110" y="34" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="8.5" fill="var(--muted)" font-style="italic">syntax &amp; agreement</text>
        <line x1="40" y1="95" x2="180" y2="95" stroke="var(--dim)" stroke-width="0.6" stroke-dasharray="3,3"/>
        <path d="M40.0,53.0 L41.8,53.9 L43.5,56.6 L45.2,60.7 L47.0,65.6 L48.8,70.8 L50.5,75.6 L52.2,79.6 L54.0,82.5 L55.8,84.4 L57.5,85.3 L59.2,85.5 L61.0,85.5 L62.8,85.5 L64.5,85.8 L66.2,86.6 L68.0,87.7 L69.8,89.2 L71.5,90.7 L73.2,92.0 L75.0,93.1 L76.8,93.7 L78.5,94.0 L80.2,93.9 L82.0,93.8 L83.8,93.7 L85.5,93.8 L87.2,94.2 L89.0,94.9 L90.8,95.7 L92.5,96.6 L94.2,97.4 L96.0,98.0 L97.8,98.3 L99.5,98.3 L101.2,98.1 L103.0,97.9 L104.8,97.7 L106.5,97.7 L108.2,98.0 L110.0,98.5 L111.8,99.1 L113.5,99.7 L115.2,100.3 L117.0,100.6 L118.8,100.7 L120.5,100.6 L122.2,100.3 L124.0,99.9 L125.8,99.7 L127.5,99.7 L129.2,99.8 L131.0,100.2 L132.8,100.8 L134.5,101.3 L136.2,101.7 L138.0,102.0 L139.8,101.9 L141.5,101.6 L143.2,101.2 L145.0,100.8 L146.8,100.4 L148.5,100.3 L150.2,100.4 L152.0,100.8 L153.8,101.3 L155.5,101.9 L157.2,102.3 L159.0,102.5 L160.8,102.4 L162.5,102.0 L164.2,101.3 L166.0,100.7 L167.8,100.2 L169.5,99.9 L171.2,100.0 L173.0,100.4 L174.8,101.0 L176.5,101.8 L178.2,102.4 L180.0,102.7" fill="none" stroke="var(--freq-fast)" stroke-width="1.8"/>
        <text x="235" y="20" text-anchor="start" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--freq-mid)" letter-spacing="0.1em">PERIODIC</text>
        <text x="235" y="34" text-anchor="start" font-family="'Source Serif 4',serif" font-size="8.5" fill="var(--muted)" font-style="italic">code &amp; tabular structure</text>
        <line x1="235" y1="95" x2="375" y2="95" stroke="var(--dim)" stroke-width="0.6" stroke-dasharray="3,3"/>
        <path d="M235.0,53.0 L236.8,56.3 L238.5,65.7 L240.2,79.8 L242.0,96.2 L243.8,112.5 L245.5,126.0 L247.2,134.6 L249.0,136.9 L250.8,132.7 L252.5,122.5 L254.2,107.9 L256.0,91.3 L257.8,75.3 L259.5,62.4 L261.2,54.7 L263.0,53.3 L264.8,58.5 L266.5,69.4 L268.2,84.4 L270.0,101.1 L271.8,116.8 L273.5,129.1 L275.2,135.9 L277.0,136.4 L278.8,130.2 L280.5,118.6 L282.2,103.2 L284.0,86.5 L285.8,71.1 L287.5,59.6 L289.2,53.6 L291.0,54.1 L292.8,61.2 L294.5,73.5 L296.2,89.3 L298.0,105.9 L299.8,120.8 L301.5,131.7 L303.2,136.8 L305.0,135.2 L306.8,127.3 L308.5,114.3 L310.2,98.3 L312.0,81.7 L313.8,67.3 L315.5,57.2 L317.2,53.1 L319.0,55.6 L320.8,64.3 L322.5,77.9 L324.2,94.1 L326.0,110.6 L327.8,124.5 L329.5,133.8 L331.2,137.0 L333.0,133.5 L334.8,124.0 L336.5,109.9 L338.2,93.4 L340.0,77.2 L341.8,63.8 L343.5,55.3 L345.2,53.1 L347.0,57.5 L348.8,67.8 L350.5,82.4 L352.2,99.0 L354.0,115.0 L355.8,127.8 L357.5,135.4 L359.2,136.7 L361.0,131.3 L362.8,120.3 L364.5,105.2 L366.2,88.5 L368.0,72.9 L369.8,60.7 L371.5,54.0 L373.2,53.7 L375.0,60.0" fill="none" stroke="var(--freq-mid)" stroke-width="1.8"/>
        <text x="500" y="20" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--freq-slow)" letter-spacing="0.1em">LONG-RANGE</text>
        <text x="500" y="34" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="8.5" fill="var(--muted)" font-style="italic">coreference &amp; theme</text>
        <line x1="430" y1="95" x2="570" y2="95" stroke="var(--dim)" stroke-width="0.6" stroke-dasharray="3,3"/>
        <path d="M430.0,53.0 L433.5,53.1 L437.0,53.4 L440.5,53.8 L444.0,54.4 L447.5,55.2 L451.0,56.1 L454.5,57.2 L458.0,58.4 L461.5,59.8 L465.0,61.3 L468.5,62.9 L472.0,64.5 L475.5,66.3 L479.0,68.2 L482.5,70.0 L486.0,72.0 L489.5,73.9 L493.0,75.9 L496.5,77.8 L500.0,79.8 L503.5,81.7 L507.0,83.6 L510.5,85.4 L514.0,87.1 L517.5,88.8 L521.0,90.4 L524.5,91.9 L528.0,93.3 L531.5,94.6 L535.0,95.8 L538.5,96.9 L542.0,97.9 L545.5,98.7 L549.0,99.5 L552.5,100.1 L556.0,100.7 L559.5,101.1 L563.0,101.4 L566.5,101.7 L570.0,101.9" fill="none" stroke="var(--freq-slow)" stroke-width="1.8"/>
        <line x1="40" y1="155" x2="570" y2="155" stroke="var(--dim)" stroke-width="0.5"/>
        <text x="305" y="175" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="9.5" fill="var(--muted)" font-style="italic">Δ, distance between tokens → (each panel: Δ from 0 to 40, dashed line = zero attention contribution)</text>
      </svg>
      <span class="note">Each curve is a genuine trigonometric sum Σ&thinsp;a<sub>k</sub>&thinsp;cos(ω<sub>k</sub>Δ) — computed, not sketched. A head chooses its kernel by learning its query and key weights; the basis never changes.</span>
    </div>

### Connection to the original sinusoidal encoding

If this reminds you of the positional encoding from the original 2017 Transformer paper ("Attention Is All You Need"), that is no coincidence. Vaswani et al. used fixed sinusoidal functions — *sin* and *cos* at geometrically spaced frequencies — as **additive** position embeddings. RoPE keeps the same Fourier backbone but applies it **multiplicatively**, as a rotation. The crucial consequence is that the additive scheme encodes absolute position in a way that only *approximately* preserves relative distance, while the rotational scheme preserves it *exactly*. The dot product factors cleanly into content and distance components — no approximation, no leakage.

### The whole trick in code

After six figures and a derivation, it is worth seeing how little machinery this actually is. Here is RoPE, complete, with a numerical proof of the invariance we just derived:

```python
import numpy as np

def rope(x, pos, base=10000):
    d = x.shape[-1]
    freqs = base ** (-np.arange(0, d, 2) / d)  # ω_k = base^(−2k/d)
    theta = pos * freqs        # advance every hand to time = pos
    cos, sin = np.cos(theta), np.sin(theta)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * cos - x[1::2] * sin  # rotate each pair…
    out[1::2] = x[0::2] * sin + x[1::2] * cos  # …by its own angle
    return out

q, k = np.random.randn(128), np.random.randn(128)
np.allclose(rope(q, 3) @ rope(k, 7),
            rope(q, 1003) @ rope(k, 1007))  # True — only the gap survives
```

That final assertion is the entire payoff of this article, executable in four lines: the same query and key, placed a thousand positions later, produce a bit-for-bit identical attention score — because the gap between them never changed.

---

## 05 — The Payoff: Relative Distance, for Free

This property — that only the gap survives — is why RoPE has quietly become the positional encoding of the modern era. Llama, Mistral, Gemma, PaLM, DeepSeek: the architectures differ, but inside every one, the same clock is ticking.

<div class="rope-diagram-wrap">
      <span class="caption">Figure 7 — Absolute positions differ, relative angle is invariant</span>
      <svg class="rope-diagram" viewBox="0 0 600 200" width="600" height="200">
        <g transform="translate(120,100)">
          <g transform="translate(-55,0)">
            <circle r="42" fill="none" stroke="var(--base-100)" stroke-width="1"/>
            <line x1="0" y1="0" x2="0" y2="-34" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" transform="rotate(30)"/>
            <circle r="2" fill="var(--ink)"/>
            <text y="58" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="8" fill="var(--muted)">pos 1</text>
          </g>
          <g transform="translate(55,0)">
            <circle r="42" fill="none" stroke="var(--base-100)" stroke-width="1"/>
            <line x1="0" y1="0" x2="0" y2="-34" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" transform="rotate(120)"/>
            <circle r="2" fill="var(--ink)"/>
            <text y="58" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="8" fill="var(--muted)">pos 4</text>
          </g>
          <path d="M-30,-26 A55,55 0 0,1 30,26" fill="none" stroke="var(--accent)" stroke-width="1.2" stroke-dasharray="3,3"/>
          <text x="0" y="-48" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--accent)" font-weight="500">Δ = 90°</text>
        </g>
        <text x="300" y="105" text-anchor="middle" font-family="'Instrument Serif',serif" font-size="28" fill="var(--dim)">=</text>
        <g transform="translate(480,100)">
          <g transform="translate(-55,0)">
            <circle r="42" fill="none" stroke="var(--base-100)" stroke-width="1"/>
            <line x1="0" y1="0" x2="0" y2="-34" stroke="var(--freq-slow)" stroke-width="2" stroke-linecap="round" transform="rotate(200)"/>
            <circle r="2" fill="var(--ink)"/>
            <text y="58" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="8" fill="var(--muted)">pos 1001</text>
          </g>
          <g transform="translate(55,0)">
            <circle r="42" fill="none" stroke="var(--base-100)" stroke-width="1"/>
            <line x1="0" y1="0" x2="0" y2="-34" stroke="var(--freq-slow)" stroke-width="2" stroke-linecap="round" transform="rotate(290)"/>
            <circle r="2" fill="var(--ink)"/>
            <text y="58" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="8" fill="var(--muted)">pos 1004</text>
          </g>
          <path d="M-22,-38 A55,55 0 0,1 38,20" fill="none" stroke="var(--freq-slow)" stroke-width="1.2" stroke-dasharray="3,3"/>
          <text x="0" y="-48" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--freq-slow)" font-weight="500">Δ = 90°</text>
        </g>
      </svg>
      <span class="note">Positions 1→4 and 1001→1004 produce the same angular gap. The model perceives only the gap — never the absolute position. This is the rotational invariance that makes RoPE work.</span>
    </div>

Hold a clock at 1:00 in your left hand and another at 1:15 in your right. The angle between the two minute hands is 90°. Now set them to 5:00 and 5:15. The absolute positions have changed completely, but the angle — the relationship — is still 90°.

**Why this matters for language.** Consider the word **"it"** referring back to its antecedent, in two different documents:

> `pos 3: cat` `pos 4: was` `pos 5: hungry` `pos 6: so` `pos 7: it` `pos 8: ate`
>
> `pos 4,031: cat` `pos 4,032: was` `pos 4,033: hungry` `pos 4,034: so` `pos 4,035: it` `pos 4,036: ate`

In both cases, *"it"* is 4 positions after *"cat"*. RoPE produces **the same angular gap** for both pairs. The model's ability to resolve the pronoun does not degrade just because the sentence appears later in the context window. Position 4,035 works exactly as well as position 7.

This is the invariance that RoPE buys you. Because position is encoded as rotation, the attention mechanism never sees an index. It cannot tell whether two tokens live at the start of a sentence or buried deep in a 128,000-token context window. All it perceives is the angle between them — the distance — and if that distance is right, the attention score will spike regardless of where in the sequence the tokens happen to fall.

---

## 06 — The Fine Print: Stretching the Clock

One caveat, and it deserves the same honesty as everything above. RoPE generalises beautifully across the distances it has *seen* — but not into distances it hasn't. Train a model on 4,096-token contexts and ask it to read position 6,000, and quality falls off a cliff. The reason is visible on the clock face. The fast hands are fine: they have lapped the dial thousands of times during training, and one more lap is nothing new. It is the **slow hands** that betray you. The hour hand has simply never been past four o'clock. The attention kernel is handed angles it never learned coefficients for, scores drift out of distribution, and the Fourier series extrapolates the way polynomials do: badly.

<div class="rope-diagram-wrap">
      <span class="caption">Figure 8 — Context extension is clock repair</span>
      <svg class="rope-diagram" viewBox="0 0 600 230" width="600" height="230">
        <text x="150" y="18" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--ink)" font-weight="500">VANILLA ROPE</text>
        <text x="150" y="32" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="8.5" fill="var(--muted)" font-style="italic">trained on 4,096 tokens, asked for position 6,000</text>
        <g transform="translate(150,115)">
          <path d="M0,0 L0,-60 A60,60 0 1,1 -52,30 Z" fill="var(--constructive)" opacity="0.09"/>
          <path d="M0,0 L-52,30 A60,60 0 0,1 0,-60 Z" fill="var(--destructive)" opacity="0.10"/>
          <circle r="60" fill="none" stroke="var(--base-150)" stroke-width="1"/>
          <text x="30" y="26" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="6.5" fill="var(--constructive)" letter-spacing="0.08em">SEEN</text>
          <text x="-26" y="-30" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="6.5" fill="var(--destructive)" letter-spacing="0.08em">NEVER SEEN</text>
          <line x1="0" y1="0" x2="0" y2="-48" stroke="var(--destructive)" stroke-width="2.2" stroke-linecap="round" transform="rotate(280)"/>
          <circle r="2.5" fill="var(--ink)"/>
        </g>
        <text x="150" y="200" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7.5" fill="var(--destructive)">hand enters the unseen zone → scores go feral</text>
        <text x="300" y="112" text-anchor="middle" font-family="'Instrument Serif',serif" font-size="26" fill="var(--dim)">→</text>
        <text x="300" y="132" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7.5" fill="var(--muted)">m → m/2</text>
        <text x="450" y="18" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="9" fill="var(--ink)" font-weight="500">POSITION INTERPOLATION</text>
        <text x="450" y="32" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="8.5" fill="var(--muted)" font-style="italic">every hand slowed ×½ — position 6,000 reads as 3,000</text>
        <g transform="translate(450,115)">
          <path d="M0,0 L0,-60 A60,60 0 1,1 -52,30 Z" fill="var(--constructive)" opacity="0.09"/>
          <path d="M0,0 L-52,30 A60,60 0 0,1 0,-60 Z" fill="var(--destructive)" opacity="0.10"/>
          <circle r="60" fill="none" stroke="var(--base-150)" stroke-width="1"/>
          <text x="30" y="26" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="6.5" fill="var(--constructive)" letter-spacing="0.08em">SEEN</text>
          <text x="-26" y="-30" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="6.5" fill="var(--destructive)" letter-spacing="0.08em">NEVER SEEN</text>
          <line x1="0" y1="0" x2="0" y2="-48" stroke="var(--constructive)" stroke-width="2.2" stroke-linecap="round" transform="rotate(140)"/>
          <circle r="2.5" fill="var(--ink)"/>
        </g>
        <text x="450" y="200" text-anchor="middle" font-family="'JetBrains Mono',monospace" font-size="7.5" fill="var(--constructive)">hand back on familiar angles → attention intact</text>
      </svg>
      <span class="note">Trained on 4,096 tokens, the slow hands have swept only part of the dial. Rescaling the frequencies folds new positions back onto angles the model already understands.</span>
    </div>

The remarkable part is the fix. Because position lives entirely in the frequency schedule — that single line defining $\omega_k$ — you do not retrain the model. You repair the clock:

- **Position Interpolation** — slow *every* hand by the same factor, $m \to m/s$. An 8k document now fits inside the angles a 4k model already knows. The cost: the fast hands lose fine positional resolution, so a brief fine-tune is needed to recalibrate.
- **NTK-aware scaling** — raise the base instead (Llama 3 ships with 500,000 rather than 10,000). This slows the lazy hands dramatically while leaving the fast hands almost untouched: long-range reach grows, local acuity survives.
- **YaRN** — the connoisseur's option. Leave the fastest hands alone, fully interpolate the slowest, blend smoothly in between, and add a small temperature correction to attention.

All three are one-line edits to the frequency schedule. No new parameters, no architectural surgery. When a model card advertises a 128k context window, this is usually what happened: someone re-geared the clock.

---

## 07 — Coda: RoPE Is Signal Processing

Strip away the Greek letters. Forget the matrix notation. What remains are three ideas, each borrowed wholesale from physics:

| 〰 The Vector | ◐ The Dimensions | ↻ The Position |
|---|---|---|
| is a waveform | are frequencies | is time |

By recasting sequence position as rotation through a multi-frequency space, RoPE gives the Transformer something it was never born with: a sense of distance that is mathematically exact, computationally almost free, and — because it lives in a frequency schedule rather than a lookup table — adjustable long after training ends.

No learnable parameters. No phone book of positions. Just a clock with a thousand hands, each spinning at its own quiet speed, encoding the geometry of language in the angles between them.

### Further reading

- **Su et al., 2021** — [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — the original RoPE paper.
- **Vaswani et al., 2017** — [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the sinusoidal ancestor.
- **Chen et al., 2023** — [Extending Context Window of LLMs via Position Interpolation](https://arxiv.org/abs/2306.15595) — slowing every hand.
- **Peng et al., 2023** — [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071) — per-band clock repair.
- **EleutherAI, 2021** — [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) — implementation notes and history.

*— fin —*
