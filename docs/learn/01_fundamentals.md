# Quantum computing fundamentals for error mitigation

This chapter collects the ideas every other part of the app relies on. Each section ends with **In the app**, the
place where you can see it happen.

## 1. Qubits, states and the Bloch sphere

A qubit is a two-level system with basis states $|0\rangle$ and $|1\rangle$. A pure state is

$$|\psi\rangle=\cos\tfrac{\theta}{2}\,|0\rangle+e^{i\varphi}\sin\tfrac{\theta}{2}\,|1\rangle ,$$

a point on the unit sphere with polar angle $\theta$ and azimuth $\varphi$ (the **Bloch sphere**). Its Bloch vector
is $\vec r=(\langle X\rangle,\langle Y\rangle,\langle Z\rangle)$ with the Pauli matrices

$$X=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad Y=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.$$

$n$ qubits live in a $2^n$-dimensional space. Most states of several qubits are **entangled**: they are not a
product of one-qubit states, and the state of one qubit on its own is then *mixed* (see §3).

**In the app:** *Playground → Time travel* draws every qubit's Bloch vector after each gate, ideal and noisy.

## 2. Gates, circuits and what hardware really runs

Gates are unitary matrices. One-qubit gates rotate the Bloch sphere, e.g. $R_Z(\lambda)=e^{-i\lambda Z/2}$ and
$R_X(\lambda)=e^{-i\lambda X/2}$; $H$ swaps the $X$ and $Z$ axes. Two-qubit gates such as CX (CNOT),
CZ and $R_{ZZ}(\lambda)=e^{-i\lambda Z\otimes Z/2}$ create entanglement.

A processor implements only a small **native gate set**. IBM's is $\{R_Z(\lambda),\sqrt X, X\}$ plus one
two-qubit gate (CZ on Heron, ECR on Eagle). Every other gate is rewritten, e.g.

$$H=e^{i\pi/4}\,R_Z(\tfrac\pi2)\,\sqrt X\,R_Z(\tfrac\pi2),\qquad \mathrm{CX}=(I\otimes H)\,\mathrm{CZ}\,(I\otimes H).$$

$R_Z$ is *virtual*: the control electronics shift the phase of later pulses instead of playing a pulse, so it
takes no time and adds no error. Only $\sqrt X$, $X$ and the two-qubit gates are physical pulses.

Two-qubit gates work only between **coupled** qubits (the coupling graph; IBM uses the heavy-hex lattice, where
a qubit has at most three neighbors). When a circuit needs a gate between uncoupled qubits, the compiler inserts
**SWAP** gates, each worth three two-qubit gates:
$\mathrm{SWAP}=\mathrm{CX}_{ab}\mathrm{CX}_{ba}\mathrm{CX}_{ab}$. Choosing *which* physical qubits hold the circuit
(the **layout**) decides how many SWAPs are needed and how good the used qubits are.

**In the app:** the Composer's *Show the compiled (physical) circuit* toggle aligns every logical gate with its
native gates and marks the routing SWAPs; *Topology & layout* shows the coupling graph and ranks physical qubits
by calibrated error.

<!-- fig:bell -->

## 3. Mixed states and density matrices

A noisy device does not prepare one pure state but a statistical mixture. The **density matrix**

$$\rho=\sum_k p_k\,|\psi_k\rangle\langle\psi_k|,\qquad \rho\succeq 0,\ \ \mathrm{Tr}\,\rho=1,$$

describes it completely. For one qubit $\rho=\tfrac12(I+\vec r\cdot\vec\sigma)$ with $|\vec r|\le 1$: pure states
sit on the sphere, mixed states inside it. The **purity** $\mathrm{Tr}\,\rho^2$ is 1 for a pure state and $2^{-n}$
for the maximally mixed one. The **fidelity** to the ideal pure state is
$F=\langle\psi_{\rm ideal}|\rho|\psi_{\rm ideal}\rangle$.

**In the app:** *Time travel* shows the density matrix (Hinton diagram) and the fidelity and purity after every
step.

## 4. Noise is a quantum channel

Any physical process, noise included, is a **channel** $\mathcal E(\rho)=\sum_k K_k\rho K_k^\dagger$ with
$\sum_k K_k^\dagger K_k=I$ (Kraus form). The important ones:

| channel | effect | Kraus operators / action |
|---|---|---|
| depolarizing ($p$) | shrinks the Bloch vector toward the center | $\rho\mapsto(1-p)\rho+p\,I/2$ |
| amplitude damping ($\gamma$) | energy relaxation toward $\lvert 0\rangle$ ($T_1$) | $K_0=\begin{pmatrix}1&0\\0&\sqrt{1-\gamma}\end{pmatrix},\ K_1=\begin{pmatrix}0&\sqrt\gamma\\0&0\end{pmatrix}$ |
| dephasing ($p$) | destroys phase, keeps populations ($T_2$) | $\rho\mapsto(1-p)\rho+p\,Z\rho Z$: coherences $\times(1-2p)$ |
| coherent error | a wrong rotation, e.g. $R_X(\epsilon)$ after every gate | unitary: the state stays pure but wrong |
| readout error | a wrong classical bit | $P(1\mid0)\ne0$, $P(0\mid1)\ne0$ |
| crosstalk | a gate on one qubit disturbs an idle neighbor | $R_Z(\delta)$ and $e^{-i\zeta Z_aZ_b/2}$ on spectators |

**Incoherent** errors (the first three) shrink the Bloch vector and lower the purity. **Coherent** errors
rotate it: they add up *in amplitude* ($N$ steps of angle $\epsilon$ give $N\epsilon$), so they grow faster than
incoherent ones but can be undone by symmetry tricks — twirling turns them into incoherent Pauli noise, and
dynamical decoupling refocuses slow phases.

On hardware, idle qubits also decay: $T_1$ (energy relaxation) and $T_2$ (dephasing) make the error of a qubit
grow with the time it waits, which is why scheduling and dynamical decoupling matter.

**In the app:** the sidebar's noise injector has every one of these, and presets such as *Crosstalk
(spectator)* and *Hardware-realistic*; a fake IBM device uses a real calibration snapshot instead.

## 5. Measurement, expectation values and shots

Measuring qubit $q$ in the computational basis returns a bit; many repetitions (**shots**) estimate
probabilities. An **observable** is a Hermitian matrix $O$, and its expectation value
$\langle O\rangle=\mathrm{Tr}(\rho O)$ is what algorithms usually need. For a Pauli string such as $Z_0Z_1$, each
shot gives $\pm1$; the average of $N$ shots has standard error $\sqrt{(1-\langle O\rangle^2)/N}$, so halving the
error costs four times the shots.

A general **Hamiltonian** $H=\sum_k c_kP_k$ is measured term by term. Terms that are *qubit-wise commuting*
(on every qubit the same Pauli or the identity) share one measurement setting: rotate $X\to Z$ with $H$ and
$Y\to Z$ with $S^\dagger H$, then measure all qubits in $Z$.

**In the app:** *Observable → Hamiltonian* accepts text such as `0.5*Z0*Z1 + 0.2*X0 - 0.1*Y1`, groups the terms
into settings and shows them; every setting is a separate circuit for every method.

## 6. What error mitigation is — and is not

A noisy device returns $\tilde O=\mathrm{Tr}[\mathcal E(\rho)\,O]\ne\langle O\rangle$. **Error correction** encodes
one logical qubit in many physical ones and removes errors during the computation; it needs many qubits and
fast feedback. **Error mitigation** leaves the device alone: it runs *more* or *modified* circuits and combines
their results classically into an estimator $\hat O$ whose **bias** $b=\mathbb E[\hat O]-\langle O\rangle$ is smaller.
The price is **variance**:

$$\mathrm{MSE}=b^2+\frac{\gamma^2\sigma^2}{N},$$

where $\gamma\ge1$ is the method's sampling overhead: $\gamma^2$ times more shots give the same precision. For
methods that remove *all* noise, $\gamma$ grows exponentially with the total error of the circuit — mitigation
is a bridge for circuits of modest size, not a replacement for error correction.

| family | idea | main assumption | cost |
|---|---|---|---|
| ZNE | run at amplified noise, extrapolate to zero | the noise can be scaled (folding) and the curve is smooth | a few circuits, $\gamma\approx$ 2–10 |
| PEC | sample the inverse of each gate's noise | the noise model is known | $\gamma=\prod_i\gamma_i$, exponential in gates |
| learning (CDR, NIL) | learn the map noisy → ideal on similar, classically simulable circuits | the training circuits see the same noise | training circuits |
| readout (tensored, M3, TREX) | invert or randomize the assignment matrix | readout errors are local | calibration circuits |
| twirling, DD | change the noise's *shape* (Pauli, refocused) | — | a little extra depth |
| symmetry verification, VD | project onto a known symmetry / purify with copies | a symmetry exists / extra qubits | discarded shots / $M$ copies |

**In the app:** *Mitigate & compare* runs any set of these side by side; *Cost vs precision*, *Bias–variance*
and *Batch run* show when each one is worth its cost.

Methods also **combine**. A *chained* contender (*Build a custom contender → Chained mitigation*) runs an outer
estimator (ZNE, CDR or NIL) on circuits that an inner method (symmetry verification, virtual distillation or PEC) has
already mitigated. The overheads multiply, so a chain pays only when the two methods remove different errors.

## 7. Three ideas the research topics build on

The *Research topics* tab follows the questions a thesis asks once the basics are in place. Three ideas recur:

* **Compare at a fixed total budget.** A method's training and calibration shots count against the same budget as its
  target shots. "Best" then depends on the budget: cheap, biased methods win small budgets, low-bias methods win
  large ones, and sometimes nothing reaches the precision you need (*Comparing methods fairly*).
* **The structure of the noise matters, not just its strength.** Two noise models with the same fault rate can favor
  different methods. A coherent error grows in amplitude, an idle error grows with time, and folding amplifies each
  differently (*Inside ZNE*).
* **A device is not its calibration.** A calibration measures each gate alone; a circuit runs gates in company.
  Spectators, simultaneous gates and idle windows add **context-dependent** errors that a component model cannot
  predict. Finding and naming them is the subject of *Context noise*, *Device versus model* and *Running on IBM
  hardware*.
