"""'Why this happens?' texts: the physics and mathematics behind every figure, in TeX."""

WHY = {
    "composer": r"""
The **logical circuit** is what you drew. The **implemented circuit** is what a device runs: input-state
preparation and measurement-basis changes are added, and with *ISA compilation* every gate is rewritten
into the device's native set $\{R_Z,\sqrt X, X, \mathrm{CX}\}$ and routed onto the coupling map (SWAPs cost three
CX each). Mitigation changes the circuit again: ZNE repeats gates ($U\to U(U^\dagger U)^k$, shaded blocks),
PEC inserts random Paulis (outlined amber), twirling sandwiches two-qubit gates between Paulis, DD fills idle
windows with pulses. Every added gate is itself noisy.
""",
    "time_travel": r"""
Each qubit's reduced state $\rho_q=\tfrac12(I+\vec r_q\cdot\vec\sigma)$ is a point in the Bloch ball.
A **coherent** error rotates $\vec r$ (a *tilt* against the ideal vector, length unchanged); an **incoherent**
error shrinks it ($|\vec r|<1$). Entanglement also shrinks the one-qubit vectors — the ideal vector shrinks
too, so only the comparison separates the two. In the circuit above, the blue box marks only the operation
executed in this step and the wires it acts on; every other wire is idle during that step. The fidelity
$F=\langle\psi_{\rm ideal}|\rho|\psi_{\rm ideal}\rangle$ falls step by step; on a folded circuit it keeps
falling through the $U^\dagger U$ blocks although the ideal state returns — that is the amplified noise ZNE
measures.
""",
    "density": r"""
$\rho_{ij}=\langle i|\rho|j\rangle$. Squares show $|\rho_{ij}|$ by area and the phase $\arg\rho_{ij}$ by
color. Diagonal entries are populations (what a Z measurement sees); off-diagonal entries are coherences.
Depolarizing and dephasing noise shrink the coherences; amplitude damping moves population toward $|0\rangle$;
a coherent phase error rotates the colors without shrinking the squares.
""",
    "fidelity": r"""
Purity $\mathrm{Tr}\rho^2=1$ for a pure state and $2^{-n}$ for the maximally mixed one. A drop in fidelity
with purity still at 1 is a **coherent** error (the state is pure but wrong); a drop in purity is **incoherent**
noise. Virtual distillation can only remove the second kind.
""",
    "estimates": r"""
Every estimator has a **bias** $b=\mathbb E[\hat O]-O_{\rm ideal}$, which no number of shots removes, and a
**variance**. Every method spends the same total budget $B=N_t+F+cs$ (target, training and calibration shots), so
$\mathrm{RMSE}(B)=\sqrt{b^2+v/N_t+w/s+u}$: mitigation lowers $|b|$ by amplifying $v$. The error bars are $\pm$ sd at
the shot budget; the dashed line is the ideal value, the dotted line the unmitigated one.
""",
    "distribution": r"""
Each violin is the spread of the estimates you would get by repeating the whole experiment (all its circuits,
at the chosen shots). The center shows the bias, the width the variance. PEC's estimates are a difference of
two sampled parts of mass $(\Gamma\pm1)/2$; ZNE's are a weighted sum $\sum_j\eta_jE_j$ with weights
$|\eta_j|$ that grow with the order of the fit; learning methods add the noise of their training data.
""",
    "table": r"""
$\gamma$ is the standard-deviation amplification relative to one unmitigated circuit: PEC's
$\prod_g\gamma_g$, ZNE's $\sum_j|\eta_j|$, learning's $\|\mathbf c\|_1$, readout mitigation's
$\prod_i\frac{1+|\epsilon_i-\eta_i|}{1-\epsilon_i-\eta_i}$, post-selection's $P_{\rm accept}^{-1/2}$.
QPU time counts the target shots plus training and calibration shots at the chosen time per shot.

**sd** is $\sqrt{v/N}$ for estimators that are linear in the measured frequencies (their exact per-shot
variance $v$). A learning method refits its map on noisy training data in every experiment, which the
linearized $v$ leaves out; its **sd** is therefore the spread of the simulated repeats (column *sd from*).
**RMSE** $=\sqrt{b^2+\mathrm{sd}^2}$.
""",
    "zne_fit": r"""
The measured points sit at $\lambda\ge1$; the value at $\lambda=0$ depends on the curve drawn through them.
Depolarizing noise gives $E(\lambda)\approx A f^{\lambda}$, where the exponential rule is exact and a line
under-corrects. Coherent errors make $E(\lambda)$ oscillate, and a high-order polynomial can overshoot.
""",
    "pec_gamma": r"""
Every corrected gate multiplies the total quasi-probability by $\gamma_g$. The positive instances carry
$(\Gamma+1)/2$, the negative ones $(\Gamma-1)/2$, $\Gamma=\prod_g\gamma_g$, and only their difference (1)
is the answer. The shot cost grows as $\Gamma^2=e^{2\sum_g\ln\gamma_g}$ — the *exponential wall*.
""",
    "pec_quasi": r"""
For a depolarizing model $\mathcal D(\rho)=(1-p)\rho+p\,I/d$, the inverse is
$\mathcal D^{-1}=\sum_Pq_P\,P\cdot P$ with $q_I>1$ and $q_{P\neq I}=\frac{1-1/(1-p)}{d^2}<0$ (red). The
compensation construction of Endo et al. writes the same map as two branches, costing
$\gamma=\frac{1+p}{1-p}$, slightly more than the inverse.
""",
    "learning": r"""
Each dot is a training circuit: its noisy value (horizontal) against its exact ideal value (vertical). If the
noise acted the same way on every circuit, the dots would lie on a line, and that line maps the target's noisy
value to its mitigated value. Scatter around the line is noise that depends on the circuit, which the
regression cannot learn.

**Why the training values must vary.** With training pairs $(x_i, y_i)$ the least-squares slope is
$\hat a=\mathrm{Cov}(x,y)/\mathrm{Var}(x)$ and the intercept $\hat b=\bar y-\hat a\bar x$. If every training
circuit has the same ideal value, $y_i\equiv y_0$, then $\mathrm{Cov}(x,y)=0$, so $\hat a=0$ and
$\hat O=\hat b=y_0$ for *any* target: the "mitigated" value is a constant that says nothing about the target
(for instance $0.0000$ when all $y_i=0$). The standard error of the slope,
$\mathrm{se}(\hat a)=\sigma_\varepsilon/\sqrt{\sum_i(x_i-\bar x)^2}$, also blows up when the $x_i$ barely vary.

The **adaptive** training set therefore checks the spread $\mathrm{sd}(y)$ with classical simulations *before*
any noisy circuit runs and escalates until $\mathrm{sd}(y)\ge$ (min. spread) $\times(\max o-\min o)$:
near-Clifford angles → uniformly random Clifford angles → keep one more non-Clifford gate → … → redraw every
rotation → also random basis-state inputs. Keeping $k$ non-Clifford gates makes the training circuits closer to the
target (better noise match) at the price of a classical cost that grows with $k$ (exponentially for stabilizer
simulators). If nothing works — e.g. $\langle Z_i\rangle=0$ for every QAOA circuit by the $X^{\otimes n}$
symmetry — the fit is **not applied**, the unmitigated value is reported and a warning is shown.
""",
    "nil": r"""
**Neighbor-informed learning** (Chen et al., arXiv:2512.12578) uses $k$ *neighbor circuits*: copies of the target
with one Pauli $X$, $Y$ or $Z$ inserted right after one gate (or the target at amplified noise). Their noisy values
$x_j$ are the features, and the estimate is $\hat O=\sum_j c_jx_j$. The weights $\mathbf c$ are learned on $T$
training circuits whose non-Clifford angles are replaced by $\{0,\pi/2,\pi,3\pi/2\}$ (ideal values computable),
each run with the same $k$ insertions: $Tk$ training circuits in all.

**Left:** the fitted value $\sum_jc_jx_j^{(i)}$ of every training circuit against its ideal value $y^{(i)}$; on
the diagonal means the combination reproduces it. The open circle is the target: its estimate against its ideal
value. **Right:** the coefficients. Chen et al. constrain $\gamma=\sum_j|c_j|\le\gamma_{\max}$: with the target
shots split $N_j\propto|c_j|\sigma_j$ the variance is $(\sum_j|c_j|\sigma_j)^2/N\le\gamma^2\max_j\sigma_j^2/N$
($\sigma_j$ the per-shot standard deviation of circuit $j$), so the bound caps the shot cost. Ridge and OLS
generally keep every neighbor ($c_j\ne0$); the ℓ1 rules (ℓ1-constrained, Lasso, elastic net) and NNLS set some
$c_j=0$, and those circuits need no target shots.

Neighbor circuits that differ only by a Pauli which commutes to the measurement of every training circuit give
identical features on the training set (they may still differ on the target), and nearby insertions give nearly
identical ones: the training data then fix only some combinations of the coefficients (the caption counts the
directions resolved above the training shot noise); the rest is decided by the regression rule — its penalty or
constraint — and identical copies share their weight equally.
""",
    "chain": r"""
A **chain** runs an outer method (ZNE, CDR or NIL) whose every circuit — folded copies, training circuits,
neighbor circuits — is first mitigated by an inner method (post-selection, PEC or virtual distillation). The outer
method then combines inner-mitigated values instead of raw ones: ZNE extrapolates $\hat E_{\rm inner}(\lambda_j)$,
CDR and NIL regress on them. Costs multiply: each circuit pays the inner method's overhead (rejected shots, $\gamma^2$
of PEC, $M$ copies of VD), and the outer weights amplify that again, $\mathrm{Var}=(\sum_j|\eta_j|\sigma_j)^2/N$ with
$\sigma_j$ the *inner* per-shot standard deviation. With an exact inner method (PEC told the true noise) every
$\hat E_{\rm inner}(\lambda_j)$ is already the ideal value and the outer step only adds variance; chaining pays off
when the inner method removes one kind of error (e.g. parity-violating errors) and the outer one another.
""",
    "spectrum": r"""
$\rho^M/\mathrm{Tr}\rho^M$ keeps the largest eigenvalue and suppresses the others by
$(\lambda_k/\lambda_1)^M$. The cost is $\mathrm{Tr}(\rho^M)^{-2}$ in shots and $M$ copies of the register.
""",
    "symmetry": r"""
Shots whose parity differs from the ideal one are discarded. The kept fraction $P_{\rm accept}$ sets the
cost ($1/P_{\rm accept}$ more shots); errors that leave the parity unchanged are not caught.
""",
}

WHY.update({
    "rmse_shots": r"""Every estimator has a **bias** $b$ (what infinitely many shots would still get wrong) and a per-shot
**variance** $v$. With $N$ shots

$$\mathrm{RMSE}(N)=\sqrt{b^2+\frac{v}{N}} .$$

On log–log axes each curve falls with slope $-\tfrac12$ while the variance dominates and then flattens at the
**bias floor** $|b|$ (dotted). Mitigation lowers $b$ but raises $v$ (ZNE by $(\sum_j|\eta_j|)^2$, PEC by
$\gamma^2$), so an unmitigated estimate wins at small $N$ and loses once
$N>N^*=\dfrac{v_m-v_r}{b_r^2-b_m^2}$ (open circles). The target line $\varepsilon$ is reachable only by methods whose
floor lies below it: $N_\varepsilon=v/(\varepsilon^2-b^2)$.""",
    "time_needed": r"""The QPU time to reach a precision $\varepsilon$ is $t=N_\varepsilon\,t_{\rm shot}$ with
$N_\varepsilon=v/(\varepsilon^2-b^2)$, plus any fixed calibration or training shots. It diverges when
$\varepsilon\to|b|$: no amount of shots removes a bias. Reading the chart from left (loose) to right (tight):
the cheapest method changes as the target tightens.""",
    "pec_wall": r"""PEC writes the inverse of each gate's noise as a quasi-probability mixture with
$\gamma_g=\sum_i|q_i|>1$. The overheads multiply: $\Gamma=\prod_g\gamma_g$, so on a circuit of $L$ layers with
$n_1$ one-qubit and $n_2$ two-qubit gates per layer

$$\Gamma(L)=\gamma_1^{\,n_1L}\gamma_2^{\,n_2L},\qquad N_\varepsilon=\frac{\Gamma^2\sigma^2}{\varepsilon^2}.$$

For depolarizing noise $\gamma\approx1+2\frac{d^2-1}{d^2}p$, hence $\Gamma^2\approx e^{4\,\bar p\,G}$ with $G$ gates:
**exponential in circuit size**. On a log axis these are straight lines; the budget line shows the depth at
which PEC stops being affordable. This is not a flaw of an implementation — lower bounds of this form hold for
every unbiased mitigation method (Takagi et al. 2022; Quek et al. 2024).""",
    "depth_sweep": r"""Real simulations of deeper and deeper circuits with the same ideal answer. Watch three things:
the unmitigated bias grows with depth; ZNE's bias grows more slowly but its variance factor stays roughly
constant; PEC's bias stays near zero (for a correct model) while $\gamma^2$ climbs exponentially — the left
panel's RMSE turns up once $\gamma^2/N$ dominates.""",
    "bullseye": r"""Each dot is one complete experiment with $N$ shots, placed by its error in two observables
(estimate − ideal). The bullseye is the truth.

* **Bias** is where the cloud is centered: the arrow from the bullseye to the cloud's center.
* **Variance** is how wide the cloud is: the dashed ellipse holds ~86 % of experiments (2σ in 2D).

Unmitigated clouds are tight but off-center; mitigated clouds are centered but wide. Increase $N$: every cloud
shrinks like $1/\sqrt N$ toward its center, but the centers do not move — so with enough shots the unbiased
methods win. The two coordinates come from independent experiments (one per observable).""",
    "mse_bars": r"""The mean squared error splits exactly into $\mathrm{MSE}=b^2+\sigma^2/N$ (summed over the two
observables). Red is what more shots cannot fix; blue shrinks as $1/N$.""",
    "mismatch": r"""PEC inverts the noise model it is **given**, not the noise the device has. Telling it
$p_{\rm assumed}=r\,p_{\rm true}$:

* $r<1$: it under-corrects; the estimate stays between the unmitigated value and the truth;
* $r=1$: unbiased (for purely depolarizing noise);
* $r>1$: it over-corrects and overshoots the truth, while paying more: $\gamma$ grows with the assumed rate.

For a single depolarizing layer with Pauli-fidelity $f=1-\tfrac{d^2}{d^2-1}p$, the corrected value is
$\langle O\rangle\,f_{\rm true}/f_{\rm assumed}$, so the bias is linear in the mismatch for small errors.
Noise that the model does not contain at all (coherent over-rotation, damping, readout) leaves a bias even at
$r=1$ — the **Unmodeled noise** tab shows it.""",
    "mismatch_map": r"""The whole mismatch landscape at once: rows are the true error rate, columns the assumed one.
The white diagonal is a correct model. Red below it (assumed < true): under-correction. Blue above it:
over-correction. The color grows with distance from the diagonal and with depth of the circuit.""",
    "readout_drift": r"""Readout mitigation inverts the assignment matrix measured in a calibration run. If the true
flip rates drift by a fraction $\delta$ after calibration, the inverse is wrong by the same fraction; the bias is
roughly linear in $\delta$. Recalibrating often (or twirling the readout, TREX) keeps $\delta$ small.""",
    "unmodeled": r"""PEC here is told the **correct** depolarizing rates, and a coherent over-rotation that the model
does not include is added. The residual bias is what a depolarizing model cannot see. Twirling (RC) converts
coherent errors into Pauli noise that a Pauli model can learn — compare ‘PEC · learned model + RC’.""",
    "sweep_lines": r"""One simulation per setting gives bias $b$ and per-shot variance $v$ for every method; the curves
are $\mathrm{RMSE}=\sqrt{b^2+v/N_t+w/s+u}$ at the chosen total budget $N$ (or the chosen metric). Crossings are where
the best method changes.""",
    "winner_map": r"""For every noise setting (x) and shot budget (y) the color is the method with the lowest
RMSE. At small budgets variance rules and cheap methods win; at large budgets bias rules and unbiased methods
win. The border between two regions is exactly $N^*=(v_a-v_b)/(b_b^2-b_a^2)$.""",
    "duel": r"""The ratio $\mathrm{RMSE}_a/\mathrm{RMSE}_b$ on a log scale: blue where $a$ is better, red where $b$
is better, white where they tie.""",
    "hardware": r"""The app records which circuits each estimator needs (folded circuits, PEC instances, training and
calibration circuits), runs them with the Sampler primitive, and replays the same estimator code on the measured
frequencies. Error bars come from resampling the counts. Gate twirling and dynamical decoupling use the
primitive's own options on real hardware; TREX randomizes measurements explicitly (random X before measurement,
bits flipped back).""",
})


WHY.update({
    "compile_diff": r"""
Compilation is three rewrites, each exact on its own:

1. **Layout** picks a physical qubit $\pi(q)$ for every logical qubit $q$.
2. **Routing**: a two-qubit gate on $(q_a,q_b)$ needs a coupler between $\pi(q_a)$ and $\pi(q_b)$. When
   there is none, SWAPs move the states along the coupling graph; each SWAP costs three two-qubit gates,
   $\mathrm{SWAP}=\mathrm{CX}_{ab}\,\mathrm{CX}_{ba}\,\mathrm{CX}_{ab}$, and it changes where later gates act (the
   final layout, used to read out the right classical bits, differs from the initial one).
3. **Translation** writes each gate in the native set. Every one-qubit gate is $R_Z(\alpha)\sqrt X R_Z(\beta)\sqrt X R_Z(\gamma)$ up to a phase; for example
   $H = e^{i\pi/4}\,R_Z(\tfrac\pi2)\,\sqrt X\,R_Z(\tfrac\pi2)$, and $\mathrm{CX} = (I\otimes H)\,\mathrm{CZ}\,(I\otimes H)$.
   $R_Z$ is **virtual** on IBM hardware (a phase-frame update, no pulse, no error), so only $\sqrt X$, $X$ and
   the two-qubit gates carry noise.

The frames in the drawing group the native gates that implement one logical gate (amber: a routing SWAP).
The executed circuit is then optimized, which can merge neighboring $R_Z$ and cancel pairs across frames.
The noise a circuit collects grows with its native two-qubit gate count $N_2$: with error $e_2$ per gate,
roughly $F\approx(1-e_2)^{N_2}$, so every SWAP multiplies the fidelity by about $(1-e_2)^3$.
""",
    "topology": r"""
The **coupling graph** has a node per physical qubit and an edge per coupler (a pair that can run a two-qubit
gate). IBM's **heavy-hex** lattice gives every qubit at most three neighbors: fewer couplers means less
crosstalk and frequency crowding, at the price of more SWAPs. A 2D grid has four neighbors, a line two.

A logical gate between qubits at graph distance $d$ needs $d-1$ SWAPs (or a chain of them) before it can run.
Rings on the graph show where the circuit's qubits start (the **initial layout**); amber couplers are where
routing swapped. With $N_{\rm swap}$ SWAPs the circuit gains $3N_{\rm swap}$ two-qubit gates, and its expected
success drops by about $(1-e_2)^{3N_{\rm swap}}$.
""",
    "layout_search": r"""
Placing a compiled circuit on other physical qubits with the **same connections** (a subgraph isomorphism of
the circuit's interaction graph into the device graph, found with VF2) gives an equivalent circuit — same gates,
same SWAPs — on hardware with different calibration. Each placement is scored by its **expected success**

$$F=\prod_{\text{2q gates}}(1-e_{ab})\;\prod_{\sqrt X,\,X}(1-e_a)\;\prod_{\text{measured}}(1-r_a),$$

the probability that no gate or readout fails (mapomatic's default cost is $1-F$). Idle decoherence ($T_1$, $T_2$)
is not in the score; the device simulation includes it. The score is a proxy: the true error of an observable
depends on which errors the observable sees, so check the simulated value after switching layouts.
""",
    "hamiltonian": r"""
A Hamiltonian $H=\sum_k c_k P_k$ (Pauli strings $P_k$) is measured term by term, but **qubit-wise commuting**
terms (on every qubit the same Pauli or the identity) share one measurement setting: rotate each qubit into the
needed basis ($X$: $H$, $Y$: $S^\dagger H$), then measure $Z$. Each setting $g$ gives an unbiased estimate
$\hat E_g$ of its part of $H$; the total is $\hat E=c_0+\sum_g \hat E_g$ with variance $\sum_g v_g/N_g$.
Splitting $N$ shots as $N_g\propto\sqrt{v_g}$ minimizes it to $\big(\sum_g\sqrt{v_g}\big)^2/N$.
Every setting is a separate circuit, so a mitigation method is applied to each and its cost adds up.
""",
    "crosstalk": r"""
A drive on qubit $a$ leaks onto its idle neighbor $b$ (spectator error). Two coherent pieces are modeled for
every driven gate and every idle neighbor: a phase kick $R_Z(\delta)$ on $b$ (AC-Stark shift) and a
conditional phase $e^{-i\zeta Z_aZ_b/2}$ (residual $ZZ$). Both are **coherent** and **quasi-static**: the same
sign every time, so they add up, $\phi_{\rm total}=\sum\delta$. Dynamical decoupling flips the spectator,
$X\,R_Z(\delta)\,X=R_Z(-\delta)$, so a pulse between two equal kicks cancels them:
$R_Z(\delta)\,X\,R_Z(\delta)\,X=I$. That is why DD helps against crosstalk and not against depolarizing noise.
""",
    "variational": r"""
A variational circuit $U(\vec\theta)$ prepares $|\psi(\vec\theta)\rangle$; the cost is
$E(\vec\theta)=\langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle$. Noise pulls $E$ toward $\mathrm{Tr}(H)/2^n$
(for depolarizing noise, $E_{\rm noisy}\approx f\,E_{\rm ideal}+(1-f)\mathrm{Tr}H/2^n$): the landscape keeps
its shape but flattens, so gradients shrink by $f$ and the minimum is **higher** but often at nearly the same
$\vec\theta$. Coherent errors shift the minimum instead. Mitigation restores the depth of the landscape at the
cost of more shots per point.
""",
    "batch": r"""
Every cell of the table is one simulated experiment: algorithm × noise × simulator × method. The **bias**
$b$ and per-shot variance $v$ come from the exact simulation; $\mathrm{RMSE}=\sqrt{b^2+v/N_t+w/s+u}$ at the chosen
total shot budget $N$ (training and calibration shots included). Comparing methods across scenarios shows which ones are robust: a method that wins on depolarizing
noise can lose on coherent or crosstalk noise (its model is wrong) or on deep circuits (its overhead $\gamma$
explodes).
""",
    "batch_sweep": r"""
A sweep changes one quantity and holds everything else fixed, so each curve isolates one cause. Three common uses:

* **Inputs.** The phase ramp $\lvert{\rm ramp}{:}k\rangle=F^\dagger\lvert k\rangle=\tfrac1{\sqrt N}\sum_j e^{-2\pi ijk/N}\lvert j\rangle$
is a product state (qubit $q$ carries the phase $-2\pi k2^q/N$), so it costs no
entangling gate, and the exact QFT maps it to $\lvert k\rangle$. The input $\lvert{+}\cdots{+}\rangle$
(``ramp:0``) is special: after each Hadamard of the QFT its qubit is already $\lvert0\rangle$, so every controlled
phase that follows has a qubit in $\lvert0\rangle$ and does nothing. An approximate QFT that drops phases therefore
still returns $\lvert0\cdots0\rangle$ exactly — the benchmark is *blind* to the truncation. ``ramp:1`` exposes it:
$\lvert\langle0001\rvert\,\mathrm{AQFT}_{m=2}\lvert{\rm ramp}{:}1\rangle\rvert^2=0.961940$ at $n=4$.
* **Parameters.** Sweeping a circuit parameter (the AQFT degree $m$, a code's error rate $p$, an angle) changes the
circuit and therefore the noise it accumulates; a mitigation method whose error stays flat is robust to that change.
* **Noise rates.** Sweeping $p_2$ at a fixed circuit shows how each estimator degrades as the error grows: the
unmitigated bias grows roughly linearly at first, ZNE's residual bias grows with the curvature its fit cannot follow,
and PEC stays unbiased while its sampling cost $\gamma^2$ grows exponentially in $p_2$ times the number of two-qubit
gates.

The metric is $\mathrm{RMSE}=\sqrt{b^2+v/N_t+w/s+u}$ (or $|b|$) at the chosen total shot budget $N$.
""",
})


WHY.update({
    "distribution_view": r"""
When the observable is a **probability** — $\langle O\rangle=\sum_{x\in S}p(x)$ for a set $S$ of outcomes — the
whole distribution $p(x)$ is the result, and noise shows up as probability leaking out of the ideal peaks into
every other bit string. Two numbers summarize the damage: the **total variation distance**
$\mathrm{TVD}(p,q)=\tfrac12\sum_x|p(x)-q(x)|$ (the largest difference in probability any test can see) and the
**Hellinger fidelity** $F_H=\bigl(\sum_x\sqrt{p(x)q(x)}\bigr)^2$.

Linear mitigations act outcome by outcome. Readout mitigation applies the inverse assignment matrix,
$\hat p=A^{-1}p_{\rm meas}$ with $A=\bigotimes_c A_c$, and returns a *quasi*-probability (entries can be slightly
negative). ZNE with linear weights applies the same combination to every outcome,
$\hat p_0(x)=\sum_j\eta_j\,p_{\lambda_j}(x)$, so summing the mitigated distribution over $S$ reproduces the ZNE
estimate exactly. When the observable reads only some bits (Shor's counting register, Bernstein–Vazirani's inputs),
the histogram shows the **marginal** over those bits.
""",
    "qec_view": r"""
For a code with logical observable $Z_L$ and a logical flip probability $p_L$,
$\langle Z_L\rangle=(1-2p_L)\,\langle Z_L\rangle_{\rm ideal}$, so every method's estimate converts into an
**effective logical error** $p_L=\tfrac12\bigl(1-\langle Z_L\rangle/\langle Z_L\rangle_{\rm ideal}\bigr)$. Correction
lowers it by undoing errors; mitigation lowers it by post-processing; below the code's threshold a larger code
suppresses it as $p_L\sim(p/p_{\rm th})^{\lceil d/2\rceil}$.

The **syndrome** is the vector of stabilizer outcomes. Without errors it is all zeros; each error pattern lights up a
characteristic set of checks (the decoder's input). The histogram shows how often each syndrome occurs in the noisy
run: $P(\text{no check fired})$ is the fraction a detection code keeps after post-selection — the acceptance of
symmetry verification when the check is the verified symmetry.
""",
    "diagnostics_view": r"""
Each method has one curve that shows whether it is working as intended:

* **ZNE** — $E(\lambda)$ at the measured noise scales and the extrapolation to $\lambda=0$. A strongly curved or
  oscillating $E(\lambda)$ (coherent errors) warns that the fit rule matters.
* **PEC** — the quasi-probabilities $q_i$ and the growth of $\gamma=\prod_g\gamma_g$ along the circuit: the shot
  cost is $\gamma^2$.
* **Learning (CDR / FR / NIL)** — ideal against noisy (CDR, FR) or fitted (NIL) values of the training circuits; a flat cloud means
  the training data cannot constrain the fit.
* **Virtual distillation** — the eigenvalues of $\rho$; the dominant one approaches the ideal state as $M$ grows.
* **Symmetry verification** — the accepted fraction $P_{\rm accept}$: shots cost $\times1/P_{\rm accept}$.
* **Readout** — the calibrated assignment errors $P(1|0)$, $P(0|1)$ per bit; the inversion overhead grows with
  $\prod_c(1-P(1|0)_c-P(0|1)_c)^{-1}$.
""",
})
