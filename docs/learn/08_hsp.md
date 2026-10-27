# The hidden subgroup problem

**The problem.** A function $f$ on a group $G$ hides a subgroup $H$ if it is constant on every coset of $H$ and
distinct on different cosets. Find $H$. Almost every exponential quantum speed-up is an instance of this one
problem: Deutsch–Jozsa and Simon ($G=\mathbb Z_2^n$), period and order finding, Shor's factoring and discrete
logarithms ($G=\mathbb Z_N$, $\mathbb Z_N\times\mathbb Z_N$) [Jozsa 2001; Childs & van Dam 2010].

<!-- fig:hsp -->

## Fourier sampling

For an **abelian** group the same three steps solve every instance:

1. Prepare $\frac{1}{\sqrt{|G|}}\sum_{x\in G}|x\rangle|f(x)\rangle$ and (conceptually) measure the output register:
   the input register collapses to a **coset state** $\frac{1}{\sqrt{|H|}}\sum_{h\in H}|x_0+h\rangle$.
2. Apply the quantum Fourier transform of $G$. Translations by $x_0$ only change phases, so the result is the
   uniform superposition of the characters $\chi_y$ that are **trivial on $H$** (the annihilator $H^\perp$).
3. Measure $y\in H^\perp$. About $\log|G|$ random elements of $H^\perp$ determine $H$ classically.

$$\Pr(y)=\begin{cases}|H|/|G| & y\in H^\perp\\ 0 & \text{otherwise.}\end{cases}$$

## In the app

The template *Hidden subgroup problem (abelian)* offers two groups:

* **$G=\mathbb Z_2^n$ (Simon-type).** $H=\mathrm{span}\{h_1,h_2\}$ (set $h_2=0$ for a one-dimensional $H$, which
  is Simon's problem). The oracle computes $f(x)=Ax$ with the rows of $A$ a basis of $H^\perp$, so $\ker f=H$:
  one CNOT per non-zero entry of $A$. The Fourier transform of $\mathbb Z_2^n$ is $H^{\otimes n}$. Every outcome
  satisfies $y\cdot h=0$ for all $h\in H$; the observable $P(y\in H^\perp)$ is ideally 1, and each generator is a
  $Z$-parity that **symmetry verification** checks (all generators at once).
* **$G=\mathbb Z_N$, $N=2^n$ (cyclic).** $H=\langle 2^j\rangle$ and $f(x)=x\bmod 2^j$ (the low $j$ bits, copied
  with CNOTs). The QFT over $\mathbb Z_N$ maps the coset state to peaks at the multiples of $N/2^j$ — **period
  finding**, the subroutine at the heart of Shor's algorithm. The observable $P(y\equiv0 \bmod N/2^j)$ is ideally 1.

## What noise does

Noise leaks probability out of $H^\perp$: under global depolarizing noise $P(y\in H^\perp)\to|H^\perp|/|G|$, the
value of a random guess. In the Simon-type instance the circuit is shallow (Hadamards and CNOTs), so the leak is
dominated by two-qubit and readout errors; in the cyclic instance the QFT's controlled phases add coherent and
incoherent error that spreads the peaks.

## Which mitigation fits

* **Symmetry verification** is natural for $\mathbb Z_2^n$: every generator $h$ of $H$ gives a parity
  $\prod_{i\in h}Z_i=+1$ that the ideal outcomes obey, and post-selection discards shots that violate any of them.
  (It is *exact* for this observable: post-selection on all generators keeps only $y\in H^\perp$, so it trades
  bias for discarded shots — a useful lesson in reading the cost column.)
* ZNE and readout mitigation apply to both instances; the QFT instance benefits from twirling, which turns the
  coherent phase errors into stochastic ones that extrapolate cleanly.

## Try it

1. Load *Hidden subgroup problem* with the default $H=\{000,011,101,110\}$ and the *Depolarizing* preset; compare
   *Unmitigated*, *Symmetry verification* and *ZNE · Richardson*. Read the accepted fraction.
2. Switch to the cyclic group with $n=4$, $j=2$ and look at the *output distribution*: four peaks at
   $0,4,8,12$.
3. In *Batch run*, sweep `p2: 0.005, 0.01, 0.02` and read which method keeps $P(y\in H^\perp)$ closest to 1.
