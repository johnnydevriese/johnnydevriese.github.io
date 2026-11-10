---
layout: post
title: "Eigenvalues Never See the Bucket Change Hands"
date: 2026-11-09
categories: artificial-intelligence machine-learning research
slug: eigenvalues-never-see-the-bucket
---

# Eigenvalues Never See the Bucket Change Hands

*Part II of II. The textbook way to ask how fast a loop settles is to read the eigenvalues of its Jacobian. For a looped transformer that answer is true, precise, and about the wrong question. Here is why, with matrices small enough to check by hand and one trained model that behaves exactly like them.*

> ~~"Near a fixed point the iteration converges at a rate set by the spectral radius of the Jacobian, so reaching tolerance ε takes about ln(1/ε) / ln(1/ρ) iterations, independent of the input."~~

That sentence is correct, and it is the reason you might budget enso about four loops per decimal digit, whatever the input length. Part I showed the model actually needs 7 loops at 12 tokens and 50 at 96. Nothing is wrong with the math. It answers a different question than the one a loop budget needs, and the bucket brigade from Part I shows exactly which question.

A quick recap. A looped transformer runs one shared block $F$ repeatedly, $h_{t+1} = F(h_t, e)$. Its toy model is a line of people passing buckets, $h_{t+1} = S h_t + x$, which needs exactly $n$ rounds for $n$ people. The trained model, enso, moves information at most two positions per loop and needs about $n/2$ loops.

## 01 - Two Experiments on the Same Loop

You can ask a loop two different questions.

**Solve.** Start from a blank state - zeros or noise - and run passes until the output matches the task's answer. This measures whether, and how fast, the loop *computes* the answer.

**Nudge.** Start from a state that has already settled, add a small disturbance, and run passes until the disturbance is gone. This measures how the loop *recovers*.

Both involve a **fixed point** $h^\ast = F(h^\ast, e)$, a state one more pass leaves unchanged. But reaching a fixed point is not the same as being right - Part I's no-recall model settled to a wrong answer by loop 14 - and a correct output can appear while the state is still moving. Stability theory is a theory of the second experiment. A loop budget needs the first.

---

## 02 - The Jacobian, by Hand

Take a loop with two stored numbers, $a$ and $b$:

$$
a_{t+1} = \tfrac12 a_t + 5, \qquad b_{t+1} = \tfrac14 b_t + 3 .
$$

The point $(10, 4)$ is a fixed point: $10/2 + 5 = 10$ and $4/4 + 3 = 4$. Start next to it at $(11, 5)$ and track the **disturbance** $\delta_t = h_t - h^\ast$:

| Pass $t$ | State | Disturbance $\delta_t$ |
|---|---|---|
| 0 | (11, 5) | (1, 1) |
| 1 | (10.5, 4.25) | (0.5, 0.25) |
| 2 | (10.25, 4.0625) | (0.25, 0.0625) |
| 3 | (10.125, 4.015625) | (0.125, 0.015625) |

The constants 5 and 3 cancel out of the disturbance, because the fixed point obeys the same rules. What's left is a table of multipliers - how much each next number moves per unit change in each current number. That table is the **Jacobian**:

$$
\delta_{t+1} = J\,\delta_t, \qquad J = \frac{\partial F}{\partial h}\bigg|_{h^\ast} = \begin{pmatrix} 0.5 & 0 \\ 0 & 0.25 \end{pmatrix}.
$$

Row $i$, column $j$ answers: if I nudge current number $j$ by one, how far does next number $i$ move? For this affine loop the equation is exact for a nudge of any size. For a nonlinear loop like a transformer it holds only for small nudges near $h^\ast$.

**Eigenvectors** are the nudge patterns that $J$ only rescales, $Jv = \lambda v$. Here $(1, 0)$ shrinks by $\lambda = 0.5$ each pass and $(0, 1)$ by $\lambda = 0.25$. The mixed nudge $(1, 1)$ is not an eigenvector: it becomes $(0.5, 0.25)$ and changes shape. The **spectral radius** is the largest eigenvalue magnitude, $\rho = \max_i \lvert \lambda_i \rvert = 0.5$.

If $\rho < 1$, every nudge eventually dies, and in the long run its size shrinks like $\rho^t$. So the standard forecast for how many passes it takes to shrink a nudge by a factor $\varepsilon$ is

$$
r_{\text{stab}} \approx \frac{\ln(1/\varepsilon)}{\ln(1/\rho)} .
$$

For $\rho = 0.5$ and $\varepsilon = 10^{-6}$ that is 19.9 passes. Notice what the formula doesn't contain: the length of the input. For enso, the radii measured at settled 24-token inputs work out to 3.5 to 3.9 loops per factor of ten, and nothing in the formula would change that for a longer input.

---

## 03 - A Line Whose Eigenvalues Are All Zero

Now hand the stability test the bucket brigade from Part I, $h_{t+1} = S h_t + x$. Its Jacobian is the hand-off itself, $J = S$. For four people:

$$
J = S = \begin{pmatrix} 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{pmatrix}.
$$

Every entry above the diagonal is zero, because nobody passes a bucket backward. The matrix is triangular, and the eigenvalues of a triangular matrix are its diagonal entries: all zero. So $\rho = 0$, and the forecast says the line settles *instantly*.

Part I showed it takes exactly $n$ rounds. Trace a line of three people and you can see why:

| Pass $t$ | Solve from zero, items $(1, 0, 0)$ | Nudge person 1, on top of the answer |
|---|---|---|
| 0 | (0, 0, 0) | (1, 0, 0) |
| 1 | (1, 0, 0) | (0, 1, 0) |
| 2 | (1, 1, 0) | (0, 0, 1) |
| 3 | (1, 1, 1) | (0, 0, 0) |

The nudge never shrinks while it travels. It moves one person per pass until it falls off the end. Zero eigenvalues and a three-pass journey are both true at once because $S$ is **nilpotent**: $S^3 = 0$, so any $Sv = \lambda v$ forces $\lambda^3 v = 0$, and $\lambda = 0$. The only patterns $S$ merely rescales are nudges sitting on the *last* person. A nudge at person 1 is not one of them; it has to walk the line before it can vanish.

### Give everyone a leaky grip

A real network also keeps part of its own old state, roughly what residual connections do. So let each person keep half their old bucket and take half of the hand-off:

$$
h_{t+1} = \tfrac12 h_t + \tfrac12 \left(S h_t + x\right), \qquad J = \tfrac12 (I + S).
$$

Now there is $\tfrac12$ on the diagonal, so every eigenvalue is $\tfrac12$ and $\rho = \tfrac12$: the 19.9-pass forecast from Section 02, at every length. But $I$ and $S$ commute, so the binomial theorem gives $J^t$ exactly:

$$
J^t = 2^{-t} \sum_{k=0}^{t} \binom{t}{k} S^k, \qquad (J^t)_{pq} = \binom{t}{p-q}\, 2^{-t}.
$$

The eigenvalue shows up only as the factor $2^{-t}$. The hand-off shows up in $\binom{t}{p-q}$, which is *exactly zero* until $t \ge p - q$. However fast the eigenvalue says things decay, person $p$ hears nothing from person $q$ until the buckets have had time to get there.

<div class="rope-diagram-wrap">
<span class="caption">Figure 1 - Rounds to settle versus line length, two toy brigades</span>
<!-- fig:toy-scaling -->
<svg class="rope-diagram" viewBox="0 0 600 284" width="600" height="284">
<rect x="70.0" y="24.0" width="470.0" height="220.0" fill="var(--paper)" stroke="var(--base-150)"/>
<line x1="70.0" y1="244.0" x2="540.0" y2="244.0" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="247.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >0</text>
<line x1="70.0" y1="200.0" x2="540.0" y2="200.0" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="203.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >100</text>
<line x1="70.0" y1="156.0" x2="540.0" y2="156.0" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="159.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >200</text>
<line x1="70.0" y1="112.0" x2="540.0" y2="112.0" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="115.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >300</text>
<line x1="70.0" y1="68.0" x2="540.0" y2="68.0" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="71.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >400</text>
<line x1="70.0" y1="24.0" x2="540.0" y2="24.0" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="27.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >500</text>
<text x="98.2" y="258.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >12</text>
<text x="182.8" y="258.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >48</text>
<text x="295.6" y="258.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >96</text>
<text x="521.2" y="258.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >192</text>
<text x="305.0" y="274.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >PEOPLE IN LINE (SEQUENCE LENGTH n)</text>
<text x="18.0" y="134.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" transform="rotate(-90 18 134.0)">ROUNDS UNTIL EVERY BUCKET IS WITHIN 10⁻⁶</text>
<line x1="70.0" y1="235.2" x2="540.0" y2="235.2" stroke="var(--ink)" stroke-width="1.4" stroke-dasharray="5 3"/>
<text x="211.0" y="229.2" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="start" >stability forecast for ρ = ½: 19.9 rounds at every n</text>
<path d="M98.2,218.5 L126.4,203.5 L182.8,176.2 L295.6,125.2 L521.2,28.0" fill="none" stroke="var(--freq-fast)" stroke-width="2"/>
<circle cx="98.2" cy="218.5" r="3.2" fill="var(--freq-fast)"/>
<text x="92.2" y="211.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--freq-fast)" text-anchor="end" >58</text>
<circle cx="126.4" cy="203.5" r="3.2" fill="var(--freq-fast)"/>
<text x="120.4" y="196.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--freq-fast)" text-anchor="end" >92</text>
<circle cx="182.8" cy="176.2" r="3.2" fill="var(--freq-fast)"/>
<text x="176.8" y="169.2" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--freq-fast)" text-anchor="end" >154</text>
<circle cx="295.6" cy="125.2" r="3.2" fill="var(--freq-fast)"/>
<text x="289.6" y="118.2" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--freq-fast)" text-anchor="end" >270</text>
<circle cx="521.2" cy="28.0" r="3.2" fill="var(--freq-fast)"/>
<text x="515.2" y="21.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--freq-fast)" text-anchor="end" >491</text>
<path d="M98.2,238.7 L126.4,233.4 L182.8,222.9 L295.6,201.8 L521.2,159.5" fill="none" stroke="var(--freq-slow)" stroke-width="2"/>
<circle cx="98.2" cy="238.7" r="3.2" fill="var(--freq-slow)"/>
<text x="92.2" y="231.7" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--freq-slow)" text-anchor="end" >12</text>
<circle cx="126.4" cy="233.4" r="3.2" fill="var(--freq-slow)"/>
<text x="120.4" y="226.4" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--freq-slow)" text-anchor="end" >24</text>
<circle cx="182.8" cy="222.9" r="3.2" fill="var(--freq-slow)"/>
<text x="176.8" y="215.9" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--freq-slow)" text-anchor="end" >48</text>
<circle cx="295.6" cy="201.8" r="3.2" fill="var(--freq-slow)"/>
<text x="289.6" y="194.8" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--freq-slow)" text-anchor="end" >96</text>
<circle cx="521.2" cy="159.5" r="3.2" fill="var(--freq-slow)"/>
<text x="515.2" y="152.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--freq-slow)" text-anchor="end" >192</text>
<text x="93.5" y="98.8" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--freq-fast)" text-anchor="start" >keep half, take half: J = ½(I + S), ρ = ½</text>
<text x="375.5" y="202.2" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--freq-slow)" text-anchor="start" >pure hand-off: J = S, ρ = 0</text>
</svg>
<!-- /fig:toy-scaling -->
<span class="note">Rounds until every bucket is within 10⁻⁶ of its answer, starting from random noise. The stability forecast is flat; both measured counts grow linearly with the length of the line. Computed, not sketched.</span>
</div>

| $n$ | Stability forecast, $\rho = \tfrac12$ | Measured, $J = \tfrac12(I+S)$ | Measured, $J = S$, $\rho = 0$ |
|---|---|---|---|
| 12 | 20 | 58 | 12 |
| 24 | 20 | 92 | 24 |
| 48 | 20 | 154 | 48 |
| 96 | 20 | 270 | 96 |
| 192 | 20 | 491 | 192 |

The leak changed the constants - a slower, softer front means more rounds - but not the scaling. In both brigades the loop count is set by the hand-offs, and the hand-offs are exactly the part of $J$ the eigenvalues ignore.

> Eigenvalues measure how fast each person lets go of their own bucket.
> They never see a bucket change hands.

---

## 04 - Why the Eigenvalues Are Blind

The toy isn't a special case. Every causal looped transformer has the same structure, and the derivation is three steps.

**Step 1 - Group the Jacobian by token.** Block $J_{pq}$ measures how much token $p$'s next state responds to token $q$'s current state. Because attention is causal, token $p$ never reads a future token $q > p$. Every block above the diagonal is zero, so $J$ is **block lower triangular**.

**Step 2 - Read off the spectrum.** The determinant of a block lower triangular matrix is the product of the determinants of its diagonal blocks, $\det(J - \lambda I) = \prod_p \det(J_{pp} - \lambda I)$. So the eigenvalues of $J$ are just the eigenvalues of the diagonal blocks:

$$
\operatorname{spec}(J) = \bigcup_{p} \operatorname{spec}(J_{pp}).
$$

The blocks below the diagonal, $J_{pq}$ with $q < p$, are the hand-offs - the only place information moves between tokens. They drop out of the spectrum entirely, however large they are. An eigenvalue measures what comes back to where it started, and in a one-way line nothing comes back.

**Step 3 - Count how far the buckets can reach.** What does see the hand-offs is a blunter argument about wiring. Suppose one loop can move information at most $w$ positions to the right. Then after $t$ loops,

$$
h_t[p] \ \text{depends only on} \ h_0[p - tw \,..\, p] \ \text{and} \ e[p - tw \,..\, p].
$$

For prefix parity, the answer at position $p$ flips whenever bit 1 flips. Until $tw \ge p - 1$, the readout at $p$ cannot depend on bit 1 at all, and the starting noise is independent of the input, so over random strings it is right exactly half the time. Hence

$$
r \;\ge\; \frac{n-1}{w}.
$$

No fixed points, no eigenvalues. Just how far the buckets can travel per round.

---

## 05 - Inside enso

This is not just an analogy; you can open enso up. Take a settled 24-token input. Each token carries 64 numbers, so the Jacobian is a $1536 \times 1536$ matrix: a $24 \times 24$ grid of $64 \times 64$ blocks. Measure the size of each block and you get Figure 2.

<div class="rope-diagram-wrap">
<span class="caption">Figure 2 - enso's Jacobian at a settled 24-token input, one cell per token-to-token block</span>
<!-- fig:blocks -->
<svg class="rope-diagram" viewBox="0 0 600 320" width="600" height="320">
<rect x="150.0" y="30.0" width="264.0" height="264.0" fill="var(--paper)" stroke="var(--base-150)"/>
<rect x="150.0" y="30.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="1.000"/>
<rect x="150.0" y="41.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.625"/>
<rect x="161.0" y="41.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.788"/>
<rect x="150.0" y="52.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.359"/>
<rect x="161.0" y="52.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.649"/>
<rect x="172.0" y="52.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.810"/>
<rect x="161.0" y="63.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.561"/>
<rect x="172.0" y="63.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.684"/>
<rect x="183.0" y="63.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.787"/>
<rect x="172.0" y="74.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.359"/>
<rect x="183.0" y="74.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.561"/>
<rect x="194.0" y="74.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.770"/>
<rect x="183.0" y="85.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.476"/>
<rect x="194.0" y="85.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.600"/>
<rect x="205.0" y="85.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.777"/>
<rect x="194.0" y="96.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.359"/>
<rect x="205.0" y="96.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.616"/>
<rect x="216.0" y="96.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.967"/>
<rect x="205.0" y="107.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.451"/>
<rect x="216.0" y="107.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.743"/>
<rect x="227.0" y="107.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.945"/>
<rect x="216.0" y="118.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.448"/>
<rect x="227.0" y="118.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.630"/>
<rect x="238.0" y="118.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.867"/>
<rect x="227.0" y="129.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.488"/>
<rect x="238.0" y="129.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.694"/>
<rect x="249.0" y="129.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.935"/>
<rect x="238.0" y="140.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.574"/>
<rect x="249.0" y="140.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.761"/>
<rect x="260.0" y="140.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.939"/>
<rect x="249.0" y="151.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.392"/>
<rect x="260.0" y="151.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.609"/>
<rect x="271.0" y="151.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.747"/>
<rect x="260.0" y="162.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.395"/>
<rect x="271.0" y="162.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.676"/>
<rect x="282.0" y="162.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.946"/>
<rect x="271.0" y="173.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.427"/>
<rect x="282.0" y="173.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.600"/>
<rect x="293.0" y="173.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.757"/>
<rect x="282.0" y="184.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.337"/>
<rect x="293.0" y="184.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.610"/>
<rect x="304.0" y="184.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.794"/>
<rect x="293.0" y="195.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.457"/>
<rect x="304.0" y="195.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.661"/>
<rect x="315.0" y="195.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.951"/>
<rect x="304.0" y="206.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.439"/>
<rect x="315.0" y="206.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.728"/>
<rect x="326.0" y="206.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.945"/>
<rect x="315.0" y="217.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.363"/>
<rect x="326.0" y="217.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.583"/>
<rect x="337.0" y="217.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.747"/>
<rect x="326.0" y="228.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.378"/>
<rect x="337.0" y="228.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.672"/>
<rect x="348.0" y="228.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.832"/>
<rect x="337.0" y="239.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.544"/>
<rect x="348.0" y="239.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.682"/>
<rect x="359.0" y="239.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.785"/>
<rect x="348.0" y="250.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.368"/>
<rect x="359.0" y="250.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.555"/>
<rect x="370.0" y="250.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.765"/>
<rect x="359.0" y="261.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.465"/>
<rect x="370.0" y="261.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.593"/>
<rect x="381.0" y="261.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.774"/>
<rect x="370.0" y="272.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.358"/>
<rect x="381.0" y="272.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.533"/>
<rect x="392.0" y="272.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.750"/>
<rect x="381.0" y="283.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.390"/>
<rect x="392.0" y="283.0" width="11.0" height="11.0" fill="var(--freq-mid)" fill-opacity="0.612"/>
<rect x="403.0" y="283.0" width="11.0" height="11.0" fill="var(--freq-slow)" fill-opacity="0.961"/>
<text x="155.5" y="24.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >1</text>
<text x="144.0" y="38.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >1</text>
<text x="276.5" y="24.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >12</text>
<text x="144.0" y="159.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >12</text>
<text x="408.5" y="24.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >24</text>
<text x="144.0" y="291.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >24</text>
<text x="282.0" y="310.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" >OLD BUCKET OF TOKEN q</text>
<text x="120.0" y="162.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" transform="rotate(-90 120.0 162.0)">NEW BUCKET OF TOKEN p</text>
<text x="432.0" y="60.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--muted)" text-anchor="start" >above the diagonal: exactly 0</text>
<text x="432.0" y="72.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="start" >(nobody passes backward)</text>
<rect x="432.0" y="122.0" width="10" height="10" fill="var(--freq-slow)"/>
<text x="448.0" y="131.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--freq-slow)" text-anchor="start" >diagonal: keeps own bucket</text>
<text x="448.0" y="143.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="start" >eigenvalues live only here</text>
<rect x="432.0" y="170.0" width="10" height="10" fill="var(--freq-mid)"/>
<text x="448.0" y="179.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--freq-mid)" text-anchor="start" >1 and 2 back: the hand-offs</text>
<text x="448.0" y="191.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="start" >invisible to eigenvalues</text>
<text x="432.0" y="244.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--muted)" text-anchor="start" >3+ back: exactly 0</text>
<text x="432.0" y="256.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="start" >(arms reach two people)</text>
</svg>
<!-- /fig:blocks -->
<span class="note">Only three bands are nonzero: the diagonal, and the hand-offs from one and two tokens back. Everything above the diagonal, and everything three or more tokens back, is exactly zero. Window-one attention over two layers: arms that reach two people.</span>
</div>

There's the bucket brigade, inside a trained network. The hand-off bands are large - median block norms of 1.730 one token back and 0.743 two back, against 2.549 on the diagonal - and they are invisible to the spectrum. The full matrix has $\rho(J) = 0.5547$, which matches the largest diagonal-block radius to 12 decimal places. That agreement is the block-triangular theorem at work; the numerical match only confirms the measurement.

The reach bound holds too, with $w = 2$. Look back at Part I's Figure 4. In every cell where loop $t < (p-1)/2$ - before bit 1 can possibly have arrived - accuracy averages 0.497 across 2,304 cells. Single cells range from 0.41 to 0.58, about the spread you'd expect from 256 random strings. No position is solved before its information can get there, and every one is solved within three loops after.

Now the nudge experiment, the thing eigenvalues do describe. Take a settled 64-token state, add a nudge of size 1 at token 1, and run the linearized loop $v \leftarrow Jv$:

<div class="rope-diagram-wrap">
<span class="caption">Figure 3 - A nudge on token 1 of a settled 64-token input</span>
<!-- fig:front -->
<svg class="rope-diagram" viewBox="0 0 600 282" width="600" height="282">
<rect x="70.0" y="30.0" width="448.0" height="240.0" fill="var(--paper)" stroke="var(--base-150)"/>
<rect x="70.0" y="30.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="1.000"/>
<rect x="84.0" y="30.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.922"/>
<rect x="70.0" y="35.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.922"/>
<rect x="98.0" y="35.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="70.0" y="40.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.922"/>
<rect x="112.0" y="40.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="70.0" y="45.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="77.0" y="45.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.922"/>
<rect x="112.0" y="45.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="126.0" y="45.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="70.0" y="50.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="77.0" y="50.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.922"/>
<rect x="112.0" y="50.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="140.0" y="50.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="70.0" y="55.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="77.0" y="55.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.922"/>
<rect x="91.0" y="55.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="98.0" y="55.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.922"/>
<rect x="112.0" y="55.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="140.0" y="55.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="154.0" y="55.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="70.0" y="60.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="77.0" y="60.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="84.0" y="60.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.922"/>
<rect x="91.0" y="60.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="98.0" y="60.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.922"/>
<rect x="112.0" y="60.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="119.0" y="60.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.922"/>
<rect x="126.0" y="60.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="154.0" y="60.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="168.0" y="60.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="70.0" y="65.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="77.0" y="65.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="119.0" y="65.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.922"/>
<rect x="126.0" y="65.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="154.0" y="65.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="161.0" y="65.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="168.0" y="65.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="182.0" y="65.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="70.0" y="70.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="77.0" y="70.0" width="77.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="154.0" y="70.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="161.0" y="70.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="168.0" y="70.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="196.0" y="70.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="70.0" y="75.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="77.0" y="75.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="84.0" y="75.0" width="70.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="154.0" y="75.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="161.0" y="75.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="168.0" y="75.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="196.0" y="75.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="210.0" y="75.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="70.0" y="80.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="77.0" y="80.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="98.0" y="80.0" width="70.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="168.0" y="80.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="210.0" y="80.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="224.0" y="80.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="70.0" y="85.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="77.0" y="85.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="105.0" y="85.0" width="77.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="182.0" y="85.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="210.0" y="85.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="231.0" y="85.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="238.0" y="85.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="70.0" y="90.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="77.0" y="90.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="119.0" y="90.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="133.0" y="90.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="140.0" y="90.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="168.0" y="90.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="224.0" y="90.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="238.0" y="90.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="245.0" y="90.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="252.0" y="90.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="70.0" y="95.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="77.0" y="95.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="84.0" y="95.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="91.0" y="95.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="98.0" y="95.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="119.0" y="95.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="168.0" y="95.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="210.0" y="95.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="217.0" y="95.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="224.0" y="95.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="245.0" y="95.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="259.0" y="95.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="266.0" y="95.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="70.0" y="100.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="77.0" y="100.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="98.0" y="100.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="119.0" y="100.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="126.0" y="100.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="147.0" y="100.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="154.0" y="100.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="161.0" y="100.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="168.0" y="100.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="175.0" y="100.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="182.0" y="100.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="224.0" y="100.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="252.0" y="100.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="259.0" y="100.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="273.0" y="100.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="280.0" y="100.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="70.0" y="105.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="77.0" y="105.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="98.0" y="105.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="140.0" y="105.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.843"/>
<rect x="154.0" y="105.0" width="77.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="231.0" y="105.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="252.0" y="105.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="266.0" y="105.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="280.0" y="105.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="294.0" y="105.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="110.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="77.0" y="110.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="84.0" y="110.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="119.0" y="110.0" width="112.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="231.0" y="110.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="266.0" y="110.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="273.0" y="110.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="287.0" y="110.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="308.0" y="110.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="115.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="77.0" y="115.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="98.0" y="115.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="119.0" y="115.0" width="105.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="224.0" y="115.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="266.0" y="115.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="280.0" y="115.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="294.0" y="115.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="315.0" y="115.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="322.0" y="115.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="120.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="77.0" y="120.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="98.0" y="120.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="119.0" y="120.0" width="119.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="238.0" y="120.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="273.0" y="120.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="294.0" y="120.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="308.0" y="120.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="322.0" y="120.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="336.0" y="120.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="125.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="77.0" y="125.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="98.0" y="125.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="126.0" y="125.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="133.0" y="125.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="140.0" y="125.0" width="91.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="231.0" y="125.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="266.0" y="125.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="294.0" y="125.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="308.0" y="125.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="336.0" y="125.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="350.0" y="125.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="130.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="77.0" y="130.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="98.0" y="130.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="119.0" y="130.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="147.0" y="130.0" width="77.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="224.0" y="130.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="273.0" y="130.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="308.0" y="130.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="322.0" y="130.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="336.0" y="130.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="357.0" y="130.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="364.0" y="130.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="135.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="77.0" y="135.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="98.0" y="135.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="119.0" y="135.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="147.0" y="135.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="154.0" y="135.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="161.0" y="135.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="168.0" y="135.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="175.0" y="135.0" width="70.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="245.0" y="135.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="280.0" y="135.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="322.0" y="135.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="336.0" y="135.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="364.0" y="135.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="378.0" y="135.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="140.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="77.0" y="140.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="98.0" y="140.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="119.0" y="140.0" width="77.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="196.0" y="140.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="210.0" y="140.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="217.0" y="140.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="231.0" y="140.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="273.0" y="140.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="315.0" y="140.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="343.0" y="140.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="350.0" y="140.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="364.0" y="140.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="378.0" y="140.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="392.0" y="140.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="145.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="77.0" y="145.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="91.0" y="145.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="98.0" y="145.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="112.0" y="145.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="119.0" y="145.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="133.0" y="145.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="140.0" y="145.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="196.0" y="145.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="203.0" y="145.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="217.0" y="145.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.765"/>
<rect x="224.0" y="145.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="273.0" y="145.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="322.0" y="145.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="350.0" y="145.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="364.0" y="145.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="371.0" y="145.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="392.0" y="145.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="399.0" y="145.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="70.0" y="150.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="77.0" y="150.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="84.0" y="150.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="91.0" y="150.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="98.0" y="150.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="119.0" y="150.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="140.0" y="150.0" width="140.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="280.0" y="150.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="322.0" y="150.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="357.0" y="150.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="371.0" y="150.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="378.0" y="150.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="392.0" y="150.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="406.0" y="150.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="70.0" y="155.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="77.0" y="155.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="98.0" y="155.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="119.0" y="155.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="140.0" y="155.0" width="140.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="280.0" y="155.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="287.0" y="155.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="294.0" y="155.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="336.0" y="155.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="357.0" y="155.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="378.0" y="155.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="392.0" y="155.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="399.0" y="155.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="420.0" y="155.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="427.0" y="155.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="160.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="77.0" y="160.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="98.0" y="160.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="119.0" y="160.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="161.0" y="160.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="168.0" y="160.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="175.0" y="160.0" width="98.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="273.0" y="160.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="280.0" y="160.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="294.0" y="160.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="336.0" y="160.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="357.0" y="160.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="385.0" y="160.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="392.0" y="160.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="406.0" y="160.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="427.0" y="160.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="434.0" y="160.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="165.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="77.0" y="165.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="91.0" y="165.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="98.0" y="165.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="119.0" y="165.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="133.0" y="165.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="140.0" y="165.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="196.0" y="165.0" width="84.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="280.0" y="165.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="329.0" y="165.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="336.0" y="165.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="350.0" y="165.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="364.0" y="165.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="392.0" y="165.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="406.0" y="165.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="420.0" y="165.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="427.0" y="165.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="441.0" y="165.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="170.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="77.0" y="170.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="98.0" y="170.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="119.0" y="170.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="126.0" y="170.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="133.0" y="170.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="140.0" y="170.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="196.0" y="170.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="210.0" y="170.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="217.0" y="170.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="273.0" y="170.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="287.0" y="170.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="294.0" y="170.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="336.0" y="170.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="343.0" y="170.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="350.0" y="170.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="371.0" y="170.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="399.0" y="170.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="406.0" y="170.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="420.0" y="170.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="441.0" y="170.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="455.0" y="170.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="175.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="77.0" y="175.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="98.0" y="175.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="119.0" y="175.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="140.0" y="175.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="196.0" y="175.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="203.0" y="175.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="217.0" y="175.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="224.0" y="175.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="231.0" y="175.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="273.0" y="175.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="280.0" y="175.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="294.0" y="175.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="350.0" y="175.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="378.0" y="175.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="399.0" y="175.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="420.0" y="175.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="434.0" y="175.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="441.0" y="175.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="455.0" y="175.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="180.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="77.0" y="180.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="119.0" y="180.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="147.0" y="180.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="154.0" y="180.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="161.0" y="180.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="168.0" y="180.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="175.0" y="180.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="217.0" y="180.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="224.0" y="180.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="231.0" y="180.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="273.0" y="180.0" width="77.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="350.0" y="180.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="385.0" y="180.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="399.0" y="180.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="427.0" y="180.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="441.0" y="180.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="448.0" y="180.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="469.0" y="180.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="185.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="77.0" y="185.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="84.0" y="185.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="91.0" y="185.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="98.0" y="185.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="119.0" y="185.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="175.0" y="185.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="182.0" y="185.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="189.0" y="185.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="217.0" y="185.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="224.0" y="185.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="238.0" y="185.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="245.0" y="185.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="259.0" y="185.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="273.0" y="185.0" width="84.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="357.0" y="185.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="385.0" y="185.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="406.0" y="185.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="434.0" y="185.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="448.0" y="185.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="455.0" y="185.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="469.0" y="185.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="190.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="77.0" y="190.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="98.0" y="190.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="119.0" y="190.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="140.0" y="190.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="196.0" y="190.0" width="63.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="259.0" y="190.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.687"/>
<rect x="266.0" y="190.0" width="91.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="357.0" y="190.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="385.0" y="190.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="406.0" y="190.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="420.0" y="190.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="427.0" y="190.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="441.0" y="190.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="455.0" y="190.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="469.0" y="190.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="195.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="77.0" y="195.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="119.0" y="195.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="140.0" y="195.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="196.0" y="195.0" width="154.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="350.0" y="195.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="385.0" y="195.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="420.0" y="195.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="434.0" y="195.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="462.0" y="195.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="469.0" y="195.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="483.0" y="195.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="200.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="77.0" y="200.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="119.0" y="200.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="140.0" y="200.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="196.0" y="200.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="203.0" y="200.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="217.0" y="200.0" width="140.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="357.0" y="200.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="392.0" y="200.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="427.0" y="200.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="441.0" y="200.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="469.0" y="200.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="483.0" y="200.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="497.0" y="200.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="205.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="77.0" y="205.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="98.0" y="205.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="105.0" y="205.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="112.0" y="205.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="119.0" y="205.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="161.0" y="205.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="168.0" y="205.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="175.0" y="205.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="182.0" y="205.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="189.0" y="205.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="217.0" y="205.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="224.0" y="205.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="231.0" y="205.0" width="105.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="336.0" y="205.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="343.0" y="205.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="357.0" y="205.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="399.0" y="205.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="420.0" y="205.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="441.0" y="205.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="469.0" y="205.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="497.0" y="205.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="504.0" y="205.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="210.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="77.0" y="210.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="119.0" y="210.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="140.0" y="210.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="196.0" y="210.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="217.0" y="210.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="224.0" y="210.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="231.0" y="210.0" width="119.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="350.0" y="210.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="392.0" y="210.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="420.0" y="210.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="455.0" y="210.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="462.0" y="210.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="476.0" y="210.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="483.0" y="210.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="504.0" y="210.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="511.0" y="210.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="70.0" y="215.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="77.0" y="215.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="119.0" y="215.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="140.0" y="215.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="196.0" y="215.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="238.0" y="215.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="252.0" y="215.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="259.0" y="215.0" width="98.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="357.0" y="215.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="399.0" y="215.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="427.0" y="215.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="455.0" y="215.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="469.0" y="215.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="504.0" y="215.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="70.0" y="220.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="77.0" y="220.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="91.0" y="220.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="98.0" y="220.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="119.0" y="220.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="140.0" y="220.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="196.0" y="220.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="238.0" y="220.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="245.0" y="220.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="259.0" y="220.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="273.0" y="220.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="280.0" y="220.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="336.0" y="220.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="343.0" y="220.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="357.0" y="220.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="399.0" y="220.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="434.0" y="220.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="448.0" y="220.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="455.0" y="220.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="462.0" y="220.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="483.0" y="220.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="504.0" y="220.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="70.0" y="225.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="77.0" y="225.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="98.0" y="225.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="119.0" y="225.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="133.0" y="225.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="140.0" y="225.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="175.0" y="225.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="210.0" y="225.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="259.0" y="225.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="273.0" y="225.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="280.0" y="225.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="294.0" y="225.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="301.0" y="225.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="329.0" y="225.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="343.0" y="225.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="350.0" y="225.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="399.0" y="225.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="441.0" y="225.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="455.0" y="225.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="490.0" y="225.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="511.0" y="225.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="70.0" y="230.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="77.0" y="230.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="98.0" y="230.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="119.0" y="230.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="140.0" y="230.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="196.0" y="230.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="217.0" y="230.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="259.0" y="230.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="266.0" y="230.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="308.0" y="230.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="322.0" y="230.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="343.0" y="230.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.608"/>
<rect x="350.0" y="230.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="399.0" y="230.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="434.0" y="230.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="462.0" y="230.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="497.0" y="230.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="504.0" y="230.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="70.0" y="235.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="77.0" y="235.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="105.0" y="235.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="112.0" y="235.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="119.0" y="235.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="140.0" y="235.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="196.0" y="235.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="217.0" y="235.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="224.0" y="235.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="238.0" y="235.0" width="168.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="406.0" y="235.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="441.0" y="235.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="469.0" y="235.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="504.0" y="235.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="70.0" y="240.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="84.0" y="240.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="91.0" y="240.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="98.0" y="240.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="119.0" y="240.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="161.0" y="240.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="196.0" y="240.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="238.0" y="240.0" width="161.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="399.0" y="240.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="441.0" y="240.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="469.0" y="240.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="504.0" y="240.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="70.0" y="245.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="98.0" y="245.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="119.0" y="245.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="126.0" y="245.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="133.0" y="245.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="140.0" y="245.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="175.0" y="245.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="196.0" y="245.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="203.0" y="245.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="210.0" y="245.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="238.0" y="245.0" width="154.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="392.0" y="245.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="441.0" y="245.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="462.0" y="245.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="476.0" y="245.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="483.0" y="245.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="497.0" y="245.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="70.0" y="250.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="98.0" y="250.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="119.0" y="250.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="140.0" y="250.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="196.0" y="250.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="217.0" y="250.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="238.0" y="250.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="245.0" y="250.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="259.0" y="250.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="273.0" y="250.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="280.0" y="250.0" width="119.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="399.0" y="250.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="441.0" y="250.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="469.0" y="250.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="504.0" y="250.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="70.0" y="255.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="98.0" y="255.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="119.0" y="255.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="140.0" y="255.0" width="56.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="196.0" y="255.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="217.0" y="255.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="259.0" y="255.0" width="98.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="357.0" y="255.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="364.0" y="255.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="385.0" y="255.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="392.0" y="255.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="399.0" y="255.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="441.0" y="255.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="469.0" y="255.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="476.0" y="255.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="483.0" y="255.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="511.0" y="255.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="70.0" y="260.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="119.0" y="260.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="161.0" y="260.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="168.0" y="260.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="175.0" y="260.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="196.0" y="260.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="217.0" y="260.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="224.0" y="260.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="231.0" y="260.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="259.0" y="260.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="266.0" y="260.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="287.0" y="260.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="294.0" y="260.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="301.0" y="260.0" width="91.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="392.0" y="260.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="441.0" y="260.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="483.0" y="260.0" width="28.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="511.0" y="260.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="70.0" y="265.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.060"/>
<rect x="119.0" y="265.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="126.0" y="265.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="133.0" y="265.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.138"/>
<rect x="140.0" y="265.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<rect x="189.0" y="265.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="196.0" y="265.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="203.0" y="265.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="217.0" y="265.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="238.0" y="265.0" width="49.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="287.0" y="265.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="294.0" y="265.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="308.0" y="265.0" width="84.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.530"/>
<rect x="392.0" y="265.0" width="42.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.452"/>
<rect x="434.0" y="265.0" width="35.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="469.0" y="265.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="476.0" y="265.0" width="7.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.373"/>
<rect x="483.0" y="265.0" width="21.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.295"/>
<rect x="504.0" y="265.0" width="14.00" height="5.00" shape-rendering="crispEdges" fill="var(--freq-fast)" fill-opacity="0.217"/>
<path d="M91.0,32.5 L105.0,37.5 L119.0,42.5 L133.0,47.5 L147.0,52.5 L161.0,57.5 L175.0,62.5 L189.0,67.5 L203.0,72.5 L217.0,77.5 L231.0,82.5 L245.0,87.5 L259.0,92.5 L273.0,97.5 L287.0,102.5 L301.0,107.5 L315.0,112.5 L329.0,117.5 L343.0,122.5 L357.0,127.5 L371.0,132.5 L385.0,137.5 L399.0,142.5 L413.0,147.5 L427.0,152.5 L441.0,157.5 L455.0,162.5 L469.0,167.5 L483.0,172.5 L497.0,177.5 L511.0,182.5 L518.0,187.5 L518.0,192.5 L518.0,197.5 L518.0,202.5 L518.0,207.5 L518.0,212.5 L518.0,217.5 L518.0,222.5 L518.0,227.5 L518.0,232.5 L518.0,237.5 L518.0,242.5 L518.0,247.5 L518.0,252.5 L518.0,257.5 L518.0,262.5 L518.0,267.5" fill="none" stroke="var(--ink)" stroke-dasharray="4 3" stroke-width="1.2"/>
<text x="73.5" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >1</text>
<text x="178.5" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >16</text>
<text x="290.5" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >32</text>
<text x="402.5" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >48</text>
<text x="514.5" y="22.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >64</text>
<text x="62.0" y="35.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >1</text>
<text x="62.0" y="90.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >12</text>
<text x="62.0" y="150.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >24</text>
<text x="62.0" y="210.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >36</text>
<text x="62.0" y="270.5" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >48</text>
<text x="294.0" y="12.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >TOKEN POSITION →</text>
<text x="24.0" y="150.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" transform="rotate(-90 24 150.0)">LOOPS AFTER NUDGE ↓</text>
<text x="370.0" y="80.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--muted)" text-anchor="start" >exactly zero: the nudge</text>
<text x="370.0" y="91.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--muted)" text-anchor="start" >cannot outrun 2 tokens/loop</text>
<rect x="532.0" y="204.0" width="12" height="14" fill="var(--freq-fast)" fill-opacity="0.060" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="532.0" y="190.0" width="12" height="14" fill="var(--freq-fast)" fill-opacity="0.138" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="532.0" y="176.0" width="12" height="14" fill="var(--freq-fast)" fill-opacity="0.217" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="532.0" y="162.0" width="12" height="14" fill="var(--freq-fast)" fill-opacity="0.295" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="532.0" y="148.0" width="12" height="14" fill="var(--freq-fast)" fill-opacity="0.373" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="532.0" y="134.0" width="12" height="14" fill="var(--freq-fast)" fill-opacity="0.452" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="532.0" y="120.0" width="12" height="14" fill="var(--freq-fast)" fill-opacity="0.530" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="532.0" y="106.0" width="12" height="14" fill="var(--freq-fast)" fill-opacity="0.608" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="532.0" y="92.0" width="12" height="14" fill="var(--freq-fast)" fill-opacity="0.687" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="532.0" y="78.0" width="12" height="14" fill="var(--freq-fast)" fill-opacity="0.765" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="532.0" y="64.0" width="12" height="14" fill="var(--freq-fast)" fill-opacity="0.843" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="532.0" y="50.0" width="12" height="14" fill="var(--freq-fast)" fill-opacity="0.922" stroke="var(--base-150)" stroke-width="0.4"/>
<rect x="532.0" y="36.0" width="12" height="14" fill="var(--freq-fast)" fill-opacity="1.000" stroke="var(--base-150)" stroke-width="0.4"/>
<text x="548.0" y="46.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="start" >1</text>
<text x="548.0" y="130.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="start" >10⁻⁶</text>
<text x="548.0" y="214.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="start" >10⁻¹²</text>
</svg>
<!-- /fig:front -->
<span class="note">Size of the disturbance at each position after each loop, log scale. The nudge rides the same two-tokens-per-loop cone as the solve and fades as it goes. Everything outside the cone is exactly zero.</span>
</div>

Same cone as Part I's Figure 4, completely different experiment. The total size of the disturbance shrinks on every loop, to about $2 \times 10^{-6}$ of where it started after 48 loops. That is the endgame the eigenvalues talk about: a finished answer absorbing a small shock. The solve is the other thing. From noise, token 96 can't settle until the hand-offs from token 1 have physically arrived - 48 loops at two tokens per loop - plus a couple more to lock in.

---

## 06 - Radius Below One, Norms Above One

There's one more way the eigenvalue forecast misleads, and it's the hand-offs again. The spectral radius describes the long run. The **operator norm** $\lVert J \rVert_2$ - the most any unit-size nudge can grow in a single pass - describes the short run. For a triangular matrix with large hand-offs, the two can disagree wildly.

Take two people, each keeping half their own error, with person 2 also taking in twice person 1's:

$$
J = \begin{pmatrix} 0.5 & 0 \\ 2 & 0.5 \end{pmatrix}, \qquad \rho(J) = 0.5, \qquad \lVert J \rVert_2 \approx 2.12 .
$$

Nudge person 1 and follow the size of the disturbance:

| Pass $t$ | Disturbance $\delta_t$ | Size $\lVert \delta_t \rVert$ |
|---|---|---|
| 0 | (1, 0) | 1.000 |
| 1 | (0.5, 2) | 2.062 |
| 2 | (0.25, 2) | 2.016 |
| 3 | (0.125, 1.5) | 1.505 |
| 4 | (0.0625, 1) | 1.002 |
| 5 | (0.03125, 0.625) | 0.626 |

The eigenvalue promises halving every pass. Instead the nudge doubles, and stays bigger than it started for four passes. The growth *is* the hand-off: person 1's error arriving at person 2, on top of person 2's own. Matrices like this, which can't be diagonalized by a rotation, are called **non-normal**, and transient growth is their signature.

enso's Jacobian is non-normal in exactly this way:

<div class="rope-diagram-wrap">
<span class="caption">Figure 4 - Worst-case growth of a nudge, eight settled enso inputs</span>
<!-- fig:transient -->
<svg class="rope-diagram" viewBox="0 0 600 280" width="600" height="280">
<rect x="70.0" y="20.0" width="470.0" height="220.0" fill="var(--paper)" stroke="var(--base-150)"/>
<line x1="70.0" y1="240.0" x2="540.0" y2="240.0" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="243.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >10⁻¹²</text>
<line x1="70.0" y1="223.1" x2="540.0" y2="223.1" stroke="var(--base-100)" stroke-width="0.6"/>
<line x1="70.0" y1="206.2" x2="540.0" y2="206.2" stroke="var(--base-100)" stroke-width="0.6"/>
<line x1="70.0" y1="189.2" x2="540.0" y2="189.2" stroke="var(--base-100)" stroke-width="0.6"/>
<line x1="70.0" y1="172.3" x2="540.0" y2="172.3" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="175.3" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >10⁻⁸</text>
<line x1="70.0" y1="155.4" x2="540.0" y2="155.4" stroke="var(--base-100)" stroke-width="0.6"/>
<line x1="70.0" y1="138.5" x2="540.0" y2="138.5" stroke="var(--base-100)" stroke-width="0.6"/>
<line x1="70.0" y1="121.5" x2="540.0" y2="121.5" stroke="var(--base-100)" stroke-width="0.6"/>
<line x1="70.0" y1="104.6" x2="540.0" y2="104.6" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="107.6" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >10⁻⁴</text>
<line x1="70.0" y1="87.7" x2="540.0" y2="87.7" stroke="var(--base-100)" stroke-width="0.6"/>
<line x1="70.0" y1="70.8" x2="540.0" y2="70.8" stroke="var(--base-100)" stroke-width="0.6"/>
<line x1="70.0" y1="53.8" x2="540.0" y2="53.8" stroke="var(--base-100)" stroke-width="0.6"/>
<line x1="70.0" y1="36.9" x2="540.0" y2="36.9" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="39.9" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >1</text>
<line x1="70.0" y1="20.0" x2="540.0" y2="20.0" stroke="var(--base-100)" stroke-width="0.6"/>
<text x="64.0" y="23.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >10¹</text>
<line x1="70.0" y1="36.9" x2="540.0" y2="36.9" stroke="var(--muted)" stroke-width="0.9"/>
<text x="70.0" y="254.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >1</text>
<text x="178.5" y="254.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >10</text>
<text x="299.0" y="254.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >20</text>
<text x="419.5" y="254.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >30</text>
<text x="540.0" y="254.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="middle" >40</text>
<text x="305.0" y="270.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="middle" >LOOPS k OF THE FROZEN LINEAR MAP</text>
<text x="18.0" y="130.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--ink)" text-anchor="middle" transform="rotate(-90 18 130.0)">WORST-CASE GROWTH ‖Jᵏ‖₂</text>
<path d="M70.0,41.4 L82.1,45.9 L94.1,50.4 L106.2,54.9 L118.2,59.4 L130.3,63.9 L142.3,68.4 L154.4,72.9 L166.4,77.4 L178.5,81.9 L190.5,86.4 L202.6,90.9 L214.6,95.4 L226.7,99.9 L238.7,104.4 L250.8,108.9 L262.8,113.4 L274.9,117.9 L286.9,122.4 L299.0,126.9 L311.0,131.4 L323.1,135.9 L335.1,140.4 L347.2,144.9 L359.2,149.4 L371.3,153.9 L383.3,158.3 L395.4,162.8 L407.4,167.3 L419.5,171.8 L431.5,176.3 L443.6,180.8 L455.6,185.3 L467.7,189.8 L479.7,194.3 L491.8,198.8 L503.8,203.3 L515.9,207.8 L527.9,212.3 L540.0,216.8" fill="none" stroke="var(--ink)" stroke-width="1.3" stroke-dasharray="5 3"/>
<path d="M70.0,29.3 L82.1,29.9 L94.1,31.5 L106.2,33.7 L118.2,37.6 L130.3,39.0 L142.3,41.3 L154.4,44.9 L166.4,47.1 L178.5,50.0 L190.5,53.3 L202.6,55.6 L214.6,58.6 L226.7,62.1 L238.7,64.4 L250.8,67.3 L262.8,70.9 L274.9,73.3 L286.9,76.2 L299.0,80.1 L311.0,82.6 L323.1,85.6 L335.1,89.6 L347.2,92.8 L359.2,95.7 L371.3,99.6 L383.3,103.7 L395.4,106.6 L407.4,110.3 L419.5,115.0 L431.5,118.3 L443.6,121.9 L455.6,126.6 L467.7,130.8 L479.7,134.3 L491.8,138.9 L503.8,143.9 L515.9,147.4 L527.9,151.7 L540.0,157.0" fill="none" stroke="var(--freq-mid)" stroke-width="1.1" stroke-opacity="0.85"/>
<path d="M70.0,29.1 L82.1,29.0 L94.1,30.2 L106.2,32.2 L118.2,36.1 L130.3,37.5 L142.3,39.7 L154.4,43.3 L166.4,45.8 L178.5,49.0 L190.5,51.8 L202.6,54.5 L214.6,58.2 L226.7,60.6 L238.7,62.9 L250.8,66.5 L262.8,69.3 L274.9,71.4 L286.9,74.7 L299.0,77.9 L311.0,80.2 L323.1,83.4 L335.1,86.9 L347.2,89.3 L359.2,92.3 L371.3,96.2 L383.3,98.9 L395.4,101.5 L407.4,105.4 L419.5,109.2 L431.5,111.4 L443.6,114.6 L455.6,119.1 L467.7,122.1 L479.7,124.7 L491.8,128.6 L503.8,133.1 L515.9,135.7 L527.9,138.9 L540.0,143.4" fill="none" stroke="var(--freq-mid)" stroke-width="1.1" stroke-opacity="0.85"/>
<path d="M70.0,28.9 L82.1,29.0 L94.1,30.7 L106.2,32.9 L118.2,34.8 L130.3,36.5 L142.3,38.6 L154.4,40.3 L166.4,42.4 L178.5,44.5 L190.5,46.7 L202.6,49.3 L214.6,51.5 L226.7,54.1 L238.7,57.0 L250.8,59.5 L262.8,62.3 L274.9,65.5 L286.9,68.2 L299.0,71.1 L311.0,74.4 L323.1,77.4 L335.1,80.1 L347.2,83.2 L359.2,86.5 L371.3,89.1 L383.3,92.1 L395.4,95.5 L407.4,98.2 L419.5,101.2 L431.5,104.6 L443.6,107.5 L455.6,110.6 L467.7,113.8 L479.7,117.0 L491.8,120.4 L503.8,123.5 L515.9,126.8 L527.9,130.5 L540.0,133.7" fill="none" stroke="var(--freq-mid)" stroke-width="1.1" stroke-opacity="0.85"/>
<path d="M70.0,29.2 L82.1,29.4 L94.1,31.0 L106.2,34.2 L118.2,36.5 L130.3,38.8 L142.3,41.1 L154.4,44.0 L166.4,46.3 L178.5,48.8 L190.5,51.9 L202.6,54.2 L214.6,56.4 L226.7,59.5 L238.7,62.0 L250.8,64.1 L262.8,67.3 L274.9,69.9 L286.9,72.1 L299.0,75.4 L311.0,78.3 L323.1,80.6 L335.1,83.9 L347.2,87.4 L359.2,89.8 L371.3,93.0 L383.3,96.8 L395.4,99.6 L407.4,102.6 L419.5,106.5 L431.5,109.9 L443.6,112.8 L455.6,116.6 L467.7,120.6 L479.7,123.6 L491.8,127.2 L503.8,131.6 L515.9,134.8 L527.9,138.4 L540.0,142.6" fill="none" stroke="var(--freq-mid)" stroke-width="1.1" stroke-opacity="0.85"/>
<path d="M70.0,28.8 L82.1,29.0 L94.1,30.6 L106.2,34.7 L118.2,37.8 L130.3,40.2 L142.3,43.7 L154.4,47.1 L166.4,49.8 L178.5,53.2 L190.5,56.9 L202.6,59.2 L214.6,62.0 L226.7,66.0 L238.7,69.6 L250.8,71.9 L262.8,75.3 L274.9,79.6 L286.9,82.7 L299.0,85.6 L311.0,89.5 L323.1,93.5 L335.1,96.2 L347.2,99.6 L359.2,103.8 L371.3,107.2 L383.3,109.9 L395.4,113.3 L407.4,117.5 L419.5,121.3 L431.5,123.8 L443.6,127.1 L455.6,131.3 L467.7,135.6 L479.7,138.2 L491.8,141.5 L503.8,145.9 L515.9,150.6 L527.9,153.3 L540.0,156.8" fill="none" stroke="var(--freq-mid)" stroke-width="1.1" stroke-opacity="0.85"/>
<path d="M70.0,29.1 L82.1,29.5 L94.1,30.7 L106.2,32.6 L118.2,35.3 L130.3,37.8 L142.3,40.0 L154.4,43.2 L166.4,46.0 L178.5,49.1 L190.5,52.8 L202.6,55.8 L214.6,59.1 L226.7,62.9 L238.7,66.5 L250.8,69.8 L262.8,73.7 L274.9,77.5 L286.9,80.8 L299.0,84.5 L311.0,88.2 L323.1,91.1 L335.1,94.4 L347.2,97.8 L359.2,100.7 L371.3,103.9 L383.3,107.5 L395.4,110.4 L407.4,113.7 L419.5,117.5 L431.5,120.7 L443.6,124.0 L455.6,128.0 L467.7,131.6 L479.7,135.0 L491.8,139.0 L503.8,143.2 L515.9,146.6 L527.9,150.6 L540.0,155.1" fill="none" stroke="var(--freq-mid)" stroke-width="1.1" stroke-opacity="0.85"/>
<path d="M70.0,29.0 L82.1,29.0 L94.1,30.7 L106.2,34.2 L118.2,36.3 L130.3,38.9 L142.3,42.1 L154.4,44.6 L166.4,47.0 L178.5,49.9 L190.5,52.3 L202.6,54.5 L214.6,57.4 L226.7,59.8 L238.7,61.9 L250.8,64.8 L262.8,67.3 L274.9,69.4 L286.9,72.3 L299.0,75.0 L311.0,77.2 L323.1,80.1 L335.1,83.0 L347.2,85.3 L359.2,88.2 L371.3,91.3 L383.3,93.9 L395.4,96.9 L407.4,100.0 L419.5,102.8 L431.5,106.0 L443.6,109.1 L455.6,112.2 L467.7,115.5 L479.7,118.8 L491.8,121.9 L503.8,125.3 L515.9,129.0 L527.9,132.1 L540.0,135.5" fill="none" stroke="var(--freq-mid)" stroke-width="1.1" stroke-opacity="0.85"/>
<path d="M70.0,28.8 L82.1,26.8 L94.1,26.2 L106.2,26.9 L118.2,28.6 L130.3,29.7 L142.3,30.6 L154.4,32.3 L166.4,34.3 L178.5,35.8 L190.5,37.6 L202.6,40.1 L214.6,42.8 L226.7,44.6 L238.7,47.1 L250.8,50.3 L262.8,52.5 L274.9,55.1 L286.9,58.3 L299.0,61.0 L311.0,63.7 L323.1,67.0 L335.1,70.0 L347.2,72.9 L359.2,76.1 L371.3,79.3 L383.3,82.3 L395.4,85.5 L407.4,89.0 L419.5,92.1 L431.5,95.3 L443.6,98.9 L455.6,102.3 L467.7,105.4 L479.7,109.2 L491.8,113.0 L503.8,116.0 L515.9,119.8 L527.9,124.1 L540.0,127.2" fill="none" stroke="var(--freq-mid)" stroke-width="1.1" stroke-opacity="0.85"/>
<text x="94.1" y="155.4" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--freq-mid)" text-anchor="start" >‖Jᵏ‖₂, eight inputs: about 3× growth on loop 1</text>
<text x="534.0" y="232.0" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8.5" fill="var(--ink)" text-anchor="end" >dashed: ρᵏ with ρ = 0.5423</text>
<text x="534.0" y="31.9" font-family="'IBM Plex Mono','JetBrains Mono',monospace" font-size="8" fill="var(--muted)" text-anchor="end" >starting size</text>
</svg>
<!-- /fig:transient -->
<span class="note">The worst-case growth after k loops for eight 24-token inputs, each at its own fixed point, against ρᵏ at the largest of their radii (dashed). Every input can grow a nudge roughly threefold on the first loop before decay sets in, and every curve stays above the dashed line.</span>
</div>

Across eight other settled 24-token inputs, $\rho$ runs from 0.5178 to 0.5423, yet $\lVert J \rVert_2$ runs from 2.80 to 3.04: the worst unit nudge roughly triples in one loop. Over the first 40 loops, $\lVert J^k \rVert_2$ peaks as high as 4.32. The worst case first drops below 1 somewhere between k = 5 and k = 11, and every curve sits above $\rho^k$ the whole way.

Radius and norm are both true, and both are about the nudge experiment. Neither one is a solve-time guarantee.

---

## 07 - Solve Time and Loop Budgets

Put the pieces side by side for enso:

- **Solving from noise** is reach-limited: about $n/2$ loops for the front to cross, plus one or two. At 96 tokens, 99% of strings are exact at loop 50, against a floor of 48.
- **Recovering from a nudge** is what $\rho$ and $\lVert J^k \rVert_2$ describe - locally, to first order, near a settled state. It is fast, and it has no $n$ in it.
- **Settling** is neither. In the length-48 run from Part I, 99% of strings are exact at loop 27 while the median step size at loop 26 is still 0.116; the step size first drops below $10^{-6}$ at loop 48. The no-recall model drops below $10^{-6}$ at loop 14 and never gets a string right.
- **Damping slows the line down.** A textbook way to tame a fixed-point iteration is Krasnoselskii-Mann damping: blend each new bucket with the old one, $h_{t+1} = (1-\alpha) h_t + \alpha F_\theta(h_t, e)$. That is the leaky grip from Section 03 applied at test time. On 48-bit parity, reaching 99% exact strings takes 27 / 29 / 40 loops for $\alpha = 1 / 0.75 / 0.5$, and at $\alpha = 0.25$ the slowed front only reaches 98.4% by loop 192. Making people hold on to their old bucket doesn't make the answer better; it makes the hand-offs take longer.

So a loop budget has to come from measured task accuracy, at the lengths and loop counts you plan to serve. A residual measures a step size, not an answer, and nothing here supports a halting rule built on one.

Here is the central claim of this post in plain NumPy: one spectral radius, and loop counts that grow with the line anyway.

<!-- code:leaky -->

```python
import math
import numpy as np

def loops_to_settle(n, tol=1e-6):
    x = np.random.default_rng(0).integers(0, 2, n).astype(float)  # each person's own item
    S = np.eye(n, k=-1)                                           # hand-off from the left
    J = 0.5 * (np.eye(n) + S)                                     # keep half your bucket, take half
    h, answer, t = np.random.default_rng(1).normal(size=n), np.cumsum(x), 0
    while np.abs(h - answer).max() > tol:
        h, t = J @ h + 0.5 * x, t + 1
    return t

rho = 0.5                                     # J is triangular: every eigenvalue is its diagonal, 1/2
forecast = math.log(1e-6) / math.log(rho)     # loops to shrink a nudge a millionfold
loops = [loops_to_settle(n) for n in (12, 24, 48, 96, 192)]

print(round(forecast, 1), loops)              # 19.9 [58, 92, 154, 270, 491]
assert min(loops) > forecast and loops == sorted(set(loops))  # True: one radius, loops grow with n
```

<!-- /code:leaky -->

Those are the numbers in Figure 1 and its table: the same $\rho$ at every length, and a loop count that keeps climbing.

---

## 08 - The Fine Print

1. **Window attention only.** The banded Jacobian and the two-tokens-per-loop front come from enso's window-one attention over two layers. With full causal attention, $w$ is effectively $n$ and the reach bound says nothing. The Jacobian is still block lower triangular, so the eigenvalues are still blind to the hand-offs, but what extra loops buy in that setting isn't tested here.
2. **A bound, not a formula.** $(n-1)/w$ is a lower bound. The pure hand-off toy comes within one round of it; the leaky toy and enso exceed it, and enso's lag of one to two loops is measured, not derived.
3. **Local means local.** Every Jacobian statement here holds near one fixed point, to first order. The toy's global argument works because the toy is affine. It does not carry over to a nonlinear model started far from its fixed point.
4. **One checkpoint, separate batches.** The enso numbers come from one trained checkpoint. The length-scaling runs, the length-48 step-size run, and the Jacobian probes are separate evaluation batches - which is why both 26 and 27 appear as the 99% loop at length 48. Compare timings within a batch, not across batches.
5. **Teaching examples are arithmetic.** The two-number loop, the three-person relay, and the $2 \times 2$ matrix are exact illustrations chosen to explain the vocabulary, not evidence about enso.

---

## 09 - Coda: Letting Go Versus Passing On

Two questions, two answers, one Jacobian:

| | Stability | Reach |
|---|---|---|
| Question | How fast does a finished answer absorb a nudge? | How long until token 1's information reaches token $n$? |
| Formula | $\ln(1/\varepsilon) \,/\, \ln(1/\rho)$ | $(n-1) \,/\, w$ |
| Lives in | the diagonal blocks $J_{pp}$ | the hand-off blocks $J_{pq}$, $q < p$ |
| Grows with $n$ | no | linearly |

In every example here, the loop count is governed by reach. Stability is still right about what it measures: how fast a nudge dies once the answer is in place.

A looped transformer doesn't hold its answer and polish it. It passes it down the line. The eigenvalues tell you how fast each person lets go of their own bucket once the answer has arrived. The hand-offs tell you how long the answer takes to get there.

### Further reading

- **Part I of this series** - *Looped Transformers Pass the Bucket*: the bucket brigade, enso, and why settling is not correctness.
- **Trefethen & Embree, 2005** - *Spectra and Pseudospectra: The Behavior of Nonnormal Matrices and Operators* (Princeton University Press) - the definitive treatment of why eigenvalues miss transient growth when hand-offs are large.
- **Bai, Kolter & Koltun, 2019** - [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) - networks defined directly by their fixed point $h^\ast = F_\theta(h^\ast, x)$, where the Jacobian at $h^\ast$ governs both forward solvers and backward gradients.
- **Geiping et al., 2025** - [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171) - Huginn, a recurrent-depth language model with full causal attention, where the spatial reach bound above does not apply.
