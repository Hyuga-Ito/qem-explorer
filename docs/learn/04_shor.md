# Shor's algorithm: order finding for N = 15

**The problem.** Factor an integer $N$. Shor (1994) reduced factoring to **order finding**: for a random $a$
coprime to $N$, find the smallest $r>0$ with

$$a^r\equiv1 \pmod N .$$

If $r$ is even and $a^{r/2}\not\equiv-1$, then $\gcd(a^{r/2}\pm1,N)$ are non-trivial factors. Only order finding
needs a quantum computer; everything else is classical.

## The quantum part: phase estimation of modular multiplication

$U_a|y\rangle=|ay\bmod N\rangle$ is a permutation of the work register. Its eigenvalues are $e^{2\pi is/r}$,
$s=0,\dots,r-1$, and $|1\rangle$ is an equal superposition of the corresponding eigenvectors. **Quantum phase
estimation** with $t$ counting qubits therefore returns, for a random $s$,

$$y\approx\frac{s}{r}\,2^{t} ,$$

and the **continued-fraction expansion** of $y/2^t$ reveals $r$ (as the denominator, or a multiple of it).

1. Counting register: $H^{\otimes t}$. Work register: $|1\rangle$.
2. Controlled-$U_a^{2^k}$ from counting qubit $k$ (the phase kick-back writes $s/r$ into the counting register).
3. Inverse QFT on the counting register, then measure $y$.

<!-- fig:shor -->

## In the app

The template *Shor order finding, N = 15* uses $a=7$ by default: $7^1=7$, $7^2=49\equiv4$, $7^3\equiv13$,
$7^4\equiv1$, so $r=4$. With $t=3$ counting qubits (q0–q2) and a 4-qubit work register (q3–q6), the ideal outcomes
are $y\in\{0,2,4,6\}$ with probability ¼ each. Continued fractions give $r=4$ from $y=2$ ($\tfrac14$) and $y=6$
($\tfrac34$), and from $y=4$ ($\tfrac12$) after checking the multiple $2\cdot2$; $y=0$ gives nothing. The observable
is the **success probability** $P(y\text{ gives } r)$, ideally $\tfrac34$. Then
$\gcd(7^2-1,15)=\gcd(48,15)=3$ and $\gcd(7^2+1,15)=\gcd(50,15)=5$.

Two options show how much of the textbook demonstration is "compiled":

* *precomputed* powers multiply by $a^{2^k}\bmod 15$ directly — this uses the answer, as in the first
  experimental demonstrations; *repeated* applies $U_a$ $2^k$ times, as an honest circuit would.
* the *permutation* multiplier is the short SWAP/NOT network; *synthesized* lets Qiskit build the controlled
  permutation from a generic unitary (hundreds of CX per call).

## What noise does

Order finding is deep: dozens of two-qubit gates even for $N=15$. Noise flattens the distribution of $y$ toward
uniform; a uniform $y$ over 8 values would succeed with probability $3/8$. The success probability therefore
falls from $0.75$ toward $0.375$ (and lower once the work register decoheres).

## Which mitigation fits

The observable is a probability (a projector), so every method applies. ZNE recovers part of the loss; PEC's
overhead grows quickly with the gate count; readout mitigation is essential because the answer *is* the bit
string. With *repeated* powers or the *synthesized* multiplier, the circuit is far beyond what mitigation can
rescue — a direct illustration of why factoring needs error correction.

## Try it

1. Load the template with *precomputed* powers and compare *Unmitigated* with *ZNE · Richardson*.
2. Switch to *repeated* powers: the two-qubit count jumps; watch the fidelity collapse in *Time travel*.
3. Compile on a heavy-hex lattice and look at the SWAPs in *Topology & layout*.
