# QAOA for MaxCut

**The problem.** Split the nodes of a graph $G=(V,E)$ into two sets so that as many edges as possible cross between
them (**MaxCut**). With a bit $z_i\in\{0,1\}$ per node, the number of cut edges is

$$C(z)=\sum_{(i,j)\in E}z_i\oplus z_j=\sum_{(i,j)\in E}\frac{1-Z_iZ_j}{2}\quad\text{(as an operator)} .$$

MaxCut is NP-hard in general, a standard benchmark for quantum optimization.

**The idea.** The Quantum Approximate Optimization Algorithm (Farhi, Goldstone and Gutmann, 2014) alternates the
cost and a *mixer* $B=\sum_iX_i$, $p$ times, starting from the uniform superposition:

$$|\vec\gamma,\vec\beta\rangle=\prod_{\ell=1}^{p}e^{-i\beta_\ell B}\,e^{-i\gamma_\ell C}\,|+\rangle^{\otimes n} .$$

The $2p$ angles are tuned (by a classical optimizer, as in VQE) to maximize $\langle C\rangle$; sampling the final
state then gives good cuts. As $p\to\infty$ QAOA can reach the optimum (it approximates adiabatic evolution).

## The circuit

* $H$ on every qubit prepares $|+\rangle^{\otimes n}$.
* The cost layer $e^{-i\gamma C}$ is, up to a global phase, one $R_{ZZ}(-\gamma)$ per edge:
  $e^{-i\gamma(1-Z_iZ_j)/2}=e^{-i\gamma/2}R_{ZZ}(-\gamma)$.
* The mixer $e^{-i\beta B}$ is $R_X(2\beta)$ on every qubit.

Because $C$ has integer eigenvalues, $\langle C\rangle$ is periodic in $\gamma$ with period $2\pi$; because $C$ is
unchanged by flipping all bits, it is periodic in $\beta$ with period $\pi/2$.

<!-- fig:qaoa -->

## In the app

The template *QAOA · MaxCut* offers ring, path, 3-regular, complete and random graphs and any depth $p$. The
observable is the **approximation ratio** $\langle C\rangle/C_{\max}$. For the 4-node ring ($C_{\max}=4$) with
$p=1$ the optimum is $0.75$, and random guessing gives $0.5$.

In *Time travel*, the $\gamma$ and $\beta$ sliders move the state live, and *Energy landscape → 2D: γ × β* maps the
whole ratio landscape, ideal next to noisy, with the best grid point starred.

## What noise does

The maximally mixed state cuts every edge with probability ½, so noise pulls the ratio toward $|E|/(2C_{\max})$
(0.5 for the ring). Under depolarizing noise the landscape is scaled, not moved: the best angles stay almost the
same — a reason QAOA angles can be trained on a noisy device. Readout errors flip bits directly and bias the cut
count.

Routing matters: a ring needs a cycle of couplers. IBM's heavy-hex lattice has only 12-cycles, so a 4-ring needs
SWAPs; complete graphs need many (swap networks). *Topology & layout* shows where they go.

## Which mitigation fits

* **Readout mitigation** first: the observable is a function of the measured bits.
* **ZNE** on $\langle C\rangle$ restores the depth of the landscape.
* **Twirling + ZNE** (the *ZNE + RC + TREX* preset) is a robust default on hardware-like noise.

## Try it

1. Load *QAOA · MaxCut* (ring, $n=4$, $p=1$). In *Time travel*, scan the 2D landscape.
2. Switch the noise to *Hardware-realistic*, scan again: the maximum is lower but at the same place.
3. Press *Move the sliders to the noisy optimum*, then *Use these angles in the circuit* and compare methods.
