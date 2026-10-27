# Grover's search

**The problem.** Find the one marked item $m$ among $N=2^n$ items, given only an oracle that recognizes it. A
classical search needs about $N/2$ queries on average; Grover's algorithm (1996) needs about
$\tfrac{\pi}{4}\sqrt N$ — a quadratic speed-up that is provably optimal for unstructured search
[Bennett et al. 1997].

<!-- fig:grover -->

## The idea: a rotation in a two-dimensional plane

Start from the uniform superposition $|s\rangle=H^{\otimes n}|0\rangle^{\otimes n}$ and write it in the plane spanned
by the marked state $|m\rangle$ and the normalized rest $|m^\perp\rangle$:

$$|s\rangle=\sin\theta\,|m\rangle+\cos\theta\,|m^\perp\rangle,\qquad \sin\theta=\frac{1}{\sqrt N}.$$

One **Grover iteration** $G=D\,O_m$ is the product of two reflections:

* the **oracle** $O_m=I-2|m\rangle\langle m|$ flips the sign of the marked amplitude (a reflection about
  $|m^\perp\rangle$);
* the **diffuser** $D=2|s\rangle\langle s|-I$ reflects about the uniform state.

Two reflections make a rotation by $2\theta$, so after $k$ iterations

$$P(m)=\sin^2\!\big((2k+1)\theta\big),\qquad k^\ast=\Big\lfloor\frac{\pi}{4\theta}\Big\rfloor\approx\frac{\pi}{4}\sqrt N .$$

Iterating further rotates *past* the target: Grover's algorithm is periodic, not monotone.

## In the app

The template *Grover search* (FTQC · deep algorithms) builds the oracle from X gates around a multi-controlled
$Z$ (the bits of $m$ that are 0 are flipped, so the controlled $Z$ fires exactly on $|m\rangle$) and the diffuser as
$H^{\otimes n}X^{\otimes n}\,C^{n-1}Z\,X^{\otimes n}H^{\otimes n}$. The multi-controlled $Z$ is a CZ for $n=2$, the
six-CNOT Clifford+T decomposition of CCZ for $n=3$, and Qiskit's synthesis beyond. The observable is $P(m)$, with
ideal value $\sin^2((2k+1)\theta)$: $1$ for $n=2$, $0.9453$ for $n=3$, $0.9613$ for $n=4$, $0.9992$ for $n=5$ at
$k^\ast$. Choose *iterations = custom* to watch the rotation overshoot.

## What noise does

Every iteration contains two $n$-qubit controlled-$Z$ gates, whose two-qubit cost grows quickly with $n$ (24 CNOTs at
$n=3$, 84 at $n=4$, 216 at $n=5$ for $k^\ast$ iterations). Depolarizing noise pulls the state toward the maximally
mixed state, for which $P(m)=1/N$. Because $k^\ast\propto\sqrt N$, the total noise grows like $\sqrt N\cdot$(gates per
iteration): the speed-up survives only while the error per iteration is far below $1/\sqrt N$.

## Which mitigation fits

$P(m)$ is a projector, so readout mitigation applies directly and ZNE extrapolates the decay of the amplitude.
PEC's overhead $\gamma$ grows exponentially with the CNOT count and becomes the limiting cost from $n=4$ on. There is
no conserved symmetry to post-select on. The *output distribution* view shows how the marked peak drains into a
uniform background.

## Try it

1. Load *Grover search* with $n=3$ and the *Hardware-realistic* preset; compare *Unmitigated*, *ZNE · Richardson*
   and *Readout · M3*.
2. Set *iterations = custom*, $k=1, 2, 3$: the ideal $P(m)$ goes $0.78\to0.95\to0.33$ — the overshoot.
3. In *Batch run*, sweep `n: 2, 3, 4` with the same methods and read the error against $n$.
