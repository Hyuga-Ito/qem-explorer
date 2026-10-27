# Theory and sources — a research handbook

This handbook collects the physics and mathematics behind QEM Explorer and places it in the research landscape. It is
organized in six parts:

* **Foundations** (§1–§3): notation, noise as quantum channels, and what mitigation can and cannot do.
* **The mitigation methods the app simulates** (§4–§11): ZNE, PEC, learning-based mitigation, readout mitigation,
  twirling, dynamical decoupling, symmetry verification and virtual distillation.
* **The machinery** (§12–§15): the test circuits, the compiler, the simulators and hardware execution.
* **Advanced mitigation** (§16): symmetry expansion, pulse-level ZNE and error amplification, quantum subspace
  expansion, classical shadows and M3 in depth, and how all methods relate.
* **Fault tolerance** (§17): the stabilizer formalism, a taxonomy of codes, decoding, the threshold theorem, and
  *logical* error mitigation — where the two roads meet.
* **Research methods** (§18–§21): equal-budget phase diagrams and the light-cone fault rate; how a device departs
  from its component-level model; the designed experiments for context-dependent noise on heavy-hex devices (layout
  variants, spectator probes, the plurality vote); and the diagnostics that name the mechanism (χ² ladder,
  divergence budget, folding fingerprint, diagnostic matrix).

Formulas use the same symbols as the app. References in square brackets, e.g. [Temme 2017], are listed with DOIs and
arXiv numbers at the end.

## 1. Notation and conventions

### Qubit order

Qubit 0 is the least significant bit, as in Qiskit: a computational basis state
$|b_{n-1}\cdots b_1b_0\rangle$ has index $x=\sum_i b_i2^i$. A $k$-qubit gate written `cx 0 1` has control 0 and
target 1, and its matrix is Qiskit's `Gate.to_matrix()` for that qubit list.

### States

An $n$-qubit state is a density matrix $\rho\ge0$, $\mathrm{Tr}\rho=1$, of size $2^n\times2^n$. A
single qubit is fully described by its **Bloch vector** $\vec r=(x,y,z)$,

$$\rho=\tfrac12\left(I+xX+yY+zZ\right),\qquad x=\mathrm{Tr}(\rho X),\ y=\mathrm{Tr}(\rho Y),\ z=\mathrm{Tr}(\rho Z).$$

Pure states have $|\vec r|=1$; mixed states lie inside the ball. For one qubit of a larger register, $\vec r$ is
computed from the reduced density matrix $\rho_q=\mathrm{Tr}_{\bar q}\,\rho$, so a qubit can have $|\vec r|<1$
because it is **entangled** with the others even if the global state is pure. The time-travel tab therefore
always shows the ideal vector next to the noisy one: if the ideal vector is short too, the qubit is entangled;
if only the noisy one is short, noise mixed it.

### Angles on the sphere

$\theta=\arccos(z/|\vec r|)$ is the polar angle from $|0\rangle$ and
$\varphi=\operatorname{atan2}(y,x)$ the azimuth from $|+\rangle$ toward $|{+i}\rangle$. The **tilt** between the
noisy and ideal vectors is $\arccos\bigl(\hat r_{\rm noisy}\cdot\hat r_{\rm ideal}\bigr)$: a coherent error rotates
the vector (tilt grows) without shortening it; an incoherent error shortens it.

### Figures of merit

Purity $\mathrm{Tr}\rho^2=\sum_{ij}|\rho_{ij}|^2$ (1 for pure states, $2^{-n}$ for the maximally
mixed state). Fidelity with a pure ideal state $|\psi\rangle$: $F=\langle\psi|\rho|\psi\rangle$; in general the
Uhlmann fidelity $F(\rho,\sigma)=\bigl(\mathrm{Tr}\sqrt{\sqrt\sigma\rho\sqrt\sigma}\bigr)^2$.

### Observables

Estimators measure diagonal observables $O=\sum_x o(x)|x\rangle\langle x|$ in the measured basis;
X or Y measurements are made by adding basis-change gates before measurement ($H$ for X, $S^\dagger H$ for Y).
The expectation value is $\langle O\rangle=\sum_x o(x)\,p(x)$ and one shot has variance
$\sigma^2=\sum_x o(x)^2p(x)-\langle O\rangle^2$.

## 2. Noise

### 2.1 Quantum channels

A noise process is a completely positive, trace-preserving (CPTP) map written with Kraus operators,
$\mathcal N(\rho)=\sum_k K_k\rho K_k^\dagger$ with $\sum_kK_k^\dagger K_k=I$. In the **Pauli transfer matrix**
(PTM) representation, $R_{PQ}=\tfrac1d\mathrm{Tr}\bigl(P\,\mathcal N(Q)\bigr)$ for Paulis $P,Q$ and $d=2^k$. A
#### Pauli channel

$\mathcal N(\rho)=\sum_Q q_Q\,Q\rho Q$ has a diagonal PTM whose entries, the **Pauli
fidelities**, are

$$f_P=\sum_Q q_Q\,\omega(P,Q),\qquad \omega(P,Q)=\begin{cases}+1 & PQ=QP\\-1 & PQ=-QP\end{cases},\qquad q_Q=\frac1{d^2}\sum_P f_P\,\omega(P,Q).$$

This pair of formulas (a Walsh–Hadamard transform) is used everywhere in the app: for PEC's inverses, for Pauli
twirling and for the device model's twirled errors.

### 2.2 The injector's channels

| Symbol | Channel | Definition |
|---|---|---|
| $p_1$, $p_2$ | depolarizing after one-/two-qubit gates | $\rho\mapsto(1-p)\rho+p\,I/d$ |
| $\gamma$ | amplitude damping ($T_1$-like) | $K_0=\begin{pmatrix}1&0\\0&\sqrt{1-\gamma}\end{pmatrix}$, $K_1=\begin{pmatrix}0&\sqrt\gamma\\0&0\end{pmatrix}$ |
| $\lambda$ | phase damping ($T_\phi$-like) | coherences multiplied by $\sqrt{1-\lambda}$ |
| $\kappa$ | random-axis kick | a rotation by $\kappa$ about a uniformly random axis; averaged, a depolarizing channel with $p=\tfrac23(1-\cos\kappa)$ |
| $\varepsilon$ | over-rotation | every physical pulse (X, Y, √X, H, RX, RY) rotates by $(1+\varepsilon)\theta$ about its own axis; virtual Z gates are exact |
| $\theta_{ZZ}$ | ZZ phase | $e^{-i\theta_{ZZ}ZZ/2}$ after each two-qubit gate (residual coupling) |
| $\varphi$ | Z phase | $R_Z(\varphi)$ on every qubit a gate touches |
| $\sigma$ | quasi-static field | a random detuning $\delta_q\sim\mathcal N(0,\sigma^2)$, fixed during one shot, rotating an idle qubit by $R_Z(\delta_q)$ per idle layer |
| $\lambda_{\rm idle}$, $\gamma_{\rm idle}$ | Markovian idle noise | dephasing and damping per idle layer; a fraction $t$ of a layer gets $1-(1-\lambda)^t$ |
| $\epsilon_{01}$, $\epsilon_{10}$ | readout | $P(1\mid0)$ and $P(0\mid1)$ for every bit |

#### From datasheets

Device datasheets quote the average gate infidelity $r=1-\bar F$. For a depolarizing channel
$\bar F=1-p\,(d-1)/d$ [Nielsen 2002], so

$$r=\frac{d-1}{d}\,p:\qquad r=\tfrac p2\ \text{(one qubit)},\qquad r=\tfrac{3p}4\ \text{(two qubits)}.$$

The sidebar shows $r$ next to the injector's $p$ values; the device model converts back with $p=r\,d/(d-1)$.

#### $T_1$ and $T_2$

Over a time $t$ amplitude damping has $\gamma=1-e^{-t/T_1}$ and coherences decay as
$e^{-t/T_2}$, with $1/T_2=1/(2T_1)+1/T_\phi$ and $T_2\le2T_1$.

### 2.3 Coherent versus incoherent errors

A stochastic (Pauli) error with probability $p$ per gate lowers the fidelity by about $p$ per gate: errors add
**in probability**, $1-F\approx Gp$ after $G$ gates. A coherent rotation error $\varepsilon$ adds **in amplitude**:
$G$ aligned errors rotate by $G\varepsilon$, so $1-F\approx(G\varepsilon)^2/4$ can grow quadratically, and it can
also cancel (folding $U U^\dagger U$ undoes an over-rotation half of the time, which is why ZNE can misbehave
under coherent noise). Pauli twirling (§8) converts coherent errors into stochastic ones.

The **quasi-static field** is simulated by Monte Carlo over the detuning: $m$ samples of $\delta_q$ taken at the
Latin-hypercube midpoints $\Phi^{-1}\bigl((k+\tfrac12)/m\bigr)\sigma$, independently permuted per qubit, and the
resulting density matrices averaged. Low-frequency noise of this kind is exactly what dynamical decoupling refocuses.

### 2.4 Idle noise and scheduling

Circuits are laid out in **as-soon-as-possible layers**. A qubit that waits while others are busy is *idle* for
that layer and receives the idle channels (dephasing, damping, the quasi-static rotation). Noise operations you
place yourself take no time: they act where you put them. Barriers align their qubits and therefore shape the idle
windows.

### 2.5 Readout errors

Readout of bit $i$ is described by its **assignment matrix**

$$A_i=\begin{pmatrix}1-\epsilon_{01}&\epsilon_{10}\\ \epsilon_{01}&1-\epsilon_{10}\end{pmatrix},\qquad \vec p_{\rm meas}=\Bigl(\bigotimes_iA_i\Bigr)\vec p_{\rm true},$$

with $\epsilon_{01}=P(\text{read }1\,|\,0)$. Relaxation during readout makes $\epsilon_{10}>\epsilon_{01}$ on most devices.

### 2.6 Fake IBM devices

Choosing a fake backend (FakeFez, FakeTorino, FakeBrisbane, …) replaces the injector with a calibration snapshot
of a real processor [Javadi-Abhari 2024]:

* the circuit is **compiled for the device**: its coupling map, its native gates (CZ on Heron, ECR on Eagle) and
  its gate durations; couplers or qubits the calibration marks as broken (error ≥ 0.5) are excluded;
* every native gate is followed by the error Qiskit Aer derives from the calibration: a depolarizing error followed
  by thermal relaxation over the gate's duration, with the depolarizing strength chosen so that together they
  reproduce the reported gate error;
* the circuit is **scheduled as late as possible** with the real durations, and every idle window becomes thermal
  relaxation with that qubit's $T_1$ and $T_2$;
* readout uses each qubit's own $P(1|0)$ and $P(0|1)$;
* the density matrix is kept only for the physical qubits the circuit uses, so a 156-qubit device costs no more
  than an $n$-qubit simulation.

A fake device is therefore a **component-level model**: every number in it was measured on isolated gates. What it
cannot contain — spectators, simultaneous gates, defects, drift — is the subject of §19–§21.

This model is **Markovian**: idle dephasing is exponential in time. Dynamical decoupling cannot refocus Markovian
noise, so on a fake device DD shows only its cost (extra pulses). On real hardware DD helps because real qubits also
suffer low-frequency noise and ZZ crosstalk; the injector's quasi-static field and ZZ phase let you study that case.

## 3. What error mitigation can and cannot do

### 3.1 Estimators, bias and variance

Error mitigation runs noisy circuits (often several) and combines their measured statistics into an **estimator**
$\hat O$ of the ideal $\langle O\rangle$. Every estimator in the app reports

* an exact mean $\mathbb E[\hat O]$, hence the **bias** $b=\mathbb E[\hat O]-\langle O\rangle_{\rm ideal}$, and
* a per-shot variance $v$, so that with $N_t$ target shots the target circuits contribute $v/N_t$ to the variance.

**Equal budget.** Every method is compared at the same **total** number of shots $B$ per estimate (the sidebar's shot
budget), split between three kinds of circuits:

$$B=N_t+F+c\,s,$$

with $N_t$ shots on the *target* circuits (the folded copies of ZNE, the sampled circuits of PEC, the post-selected
circuit of symmetry verification, …), $F$ *training* shots (CDR/NIL: training circuits × shots each) and $c$ readout
*calibration* circuits with $s$ shots each. A finitely sampled calibration estimates flip probabilities $\theta_k$ with
binomial error; to first order (delta method) it adds $w/s$ with $w=\sum_k(\partial\hat O/\partial\theta_k)^2\theta_k(1-\theta_k)$. The training circuits' own shot noise propagates through the fit to a constant $u$ that more
target shots cannot remove. The variance and the mean squared error at budget $B$ are

$$\mathrm{Var}(B)=\frac{v}{N_t}+\frac{w}{s}+u,\qquad \mathrm{MSE}(B)=b^2+\mathrm{Var}(B),\qquad \mathrm{RMSE}(B)=\sqrt{\mathrm{MSE}(B)}.$$

With $R=B-F$ the calibration shots that minimize the variance are $s^*=R\kappa/(1+c\kappa)$, $\kappa=\sqrt{w/(cv)}$
(at least one shot per calibration circuit); a method whose training cost exceeds $B$ gives no estimate at that budget.
Unmitigated estimation has a large bias and a small variance ($v=\sigma^2$). Mitigation reduces $b$ and increases
$v$: the **sampling overhead** $\gamma$ is the factor by which the standard deviation grows, $v\approx\gamma^2\sigma^2$.
Without training or calibration a mitigated method beats the unmitigated one once

$$N>N^*=\frac{v_m-v_r}{b_r^2-b_m^2},$$

and reaching a target RMSE $\varepsilon$ needs $N_\varepsilon=v/(\varepsilon^2-b^2)$ shots, infinite when
$\varepsilon\le|b|$. With training and calibration the app solves the same two questions numerically on
$\mathrm{RMSE}(B)$: the crossover budget $B^*$ and the budget for $\varepsilon$, which is infinite when
$\varepsilon^2\le b^2+u$. These are the contents of the **Cost vs precision** page (implementation:
`qemx/budget.py`).

### 3.2 The exponential cost is fundamental

For a circuit of $G$ gates each with error rate $p$, the unmitigated signal decays roughly like $(1-p)^{G}$. Any
method that removes the bias must amplify the remaining signal by the inverse factor, so its variance grows like
$e^{2cpG}$ for some constant $c$. This is not a limitation of one method: lower bounds show that, for broad classes
of noise, every unbiased mitigation protocol needs a number of samples exponential in circuit depth or size
[Takagi 2022; Quek 2024]. What differs between methods is the constant $c$, the assumptions they need, and how
their bias behaves when the assumptions fail [Cai 2023]. §18.4 measures the constants: the cost exponent $\beta$ of
each method on fake_fez, in $\mathrm{RMSE}\approx\sqrt{b^2+Ae^{\beta\lambda_{LC}}/B}$.

### 3.3 QPU time

The app converts shots into QPU time with the sidebar's time per shot $t_{\rm shot}$ (repetition period including
reset): $t=B\,t_{\rm shot}$ — the same for every method, because target, training and calibration shots all come
out of the same budget $B$.

**Circuits.** The number of distinct circuits a method sends is part of its cost too (compilation, job size): ZNE
sends one circuit per noise scale, PEC one randomly sampled circuit per $s_{\rm PEC}$ shots ($N_t/s_{\rm PEC}$
circuits; the textbook protocol has $s_{\rm PEC}=1$, one shot per sampled circuit), twirling and randomized compiling
one random circuit per shot, CDR/NIL one per training circuit, and readout mitigation two calibration circuits
($|0\cdots0\rangle$, $|1\cdots1\rangle$; one twirled $|0\cdots0\rangle$ for TREX). TREX sends every fixed circuit with 8
random measurement masks (X gates before measurement, undone classically), as on hardware.

## 4. Zero-noise extrapolation (ZNE)

### 4.1 Idea

If the noise can be **amplified** in a controlled way by a factor $\lambda\ge1$, one can measure
$E(\lambda)=\langle O\rangle_\lambda$ at several $\lambda_j$ and extrapolate to the unreachable $\lambda=0$
[Temme 2017; Li 2017]. ZNE needs no noise model — only the assumption that the measured points follow a known
curve.

### 4.2 Noise amplification by folding

On gate-based hardware the noise is amplified digitally by **unitary folding** [Giurgica-Tiron 2020]:

* **global folding** $U\to U\,(U^\dagger U)^k$ runs every gate $\lambda=2k+1$ times while the ideal circuit is
  unchanged. For a scale between odd integers the app folds a tail of the circuit (the number of gates closest
  to the request), so $\lambda$ is realized in steps of $2/G$;
* **local folding** replaces each chosen gate $G\to G(G^\dagger G)^k$; the app folds two-qubit gates (all gates
  when there are none) and, for a fractional $\lambda$, a random subset once more.

The achieved $\lambda$ is counted in two-qubit gates (all gates when there are none) and shown next to the
requested one. Folding amplifies gate errors in proportion; it does **not** amplify errors that do not scale with
gate count (state preparation, readout, idle noise is amplified only through the longer schedule), and coherent
errors can partly cancel in $U^\dagger U$.

<!-- fig:zne_fold -->

### 4.3 Extrapolation rules

With measured points $(\lambda_j,E_j)$, $j=1..m$:

* **Richardson** — the degree-$(m-1)$ polynomial through all points evaluated at 0, i.e. Lagrange interpolation:

  $$\hat E(0)=\sum_j\eta_jE_j,\qquad \eta_j=\prod_{l\ne j}\frac{\lambda_l}{\lambda_l-\lambda_j},\qquad \sum_j\eta_j=1.$$

* **linear** — the least-squares line; $\eta$ is the first row of the pseudo-inverse of the Vandermonde matrix
  $[1,\lambda_j]$.
* **exponential** — $E(\lambda)=Ae^{-b\lambda}$, fitted as a line to $\log|E_j|$ (all $E_j$ of one sign); natural
  for depolarizing-like decay $E(\lambda)\approx(1-p)^{\lambda G}E_{\rm ideal}$. It fails (no finite value) when
  the $E_j$ change sign; the app counts such failures in repeated experiments instead of averaging them.

### 4.4 Variance and optimal shot allocation

A linear estimator $\hat E(0)=\sum_j\eta_j\hat E_j$ with $N_j$ shots on circuit $j$ has
$\mathrm{Var}=\sum_j\eta_j^2\sigma_j^2/N_j$. Minimizing under $\sum_jN_j=N$ (Lagrange multiplier) gives

$$N_j\propto|\eta_j|\sigma_j,\qquad \mathrm{Var}\bigl[\hat E(0)\bigr]=\frac{\bigl(\sum_j|\eta_j|\sigma_j\bigr)^2}{N},$$

so ZNE's overhead is $\gamma=\sum_j|\eta_j|$ (with all $\sigma_j$ equal). Richardson with scales 1, 3, 5 has
$\eta=(15/8,-10/8,3/8)$ and $\gamma=3.5$: small bias, large variance. The exponential rule is linearized around the
measured values to obtain its $\eta$.

### 4.5 When ZNE fails

The bias is the error of the extrapolation model: a curved $E(\lambda)$ fitted linearly under-corrects, a
high-order polynomial through noisy points over-shoots, and coherent errors make $E(\lambda)$ oscillate. IBM's
large-scale experiments combine ZNE with learned noise models to amplify noise more faithfully
(probabilistic error amplification) [Kim 2023a; Kim 2023b].

### 4.6 Nominal and effective noise scale; global against local folding (v1.4.1)

Both foldings reach the same two-qubit gate count $\lambda n_2$ (the *nominal* scale), but not the same noise.
Global folding also repeats every one-qubit layer and idle window; local folding repeats only the two-qubit gates, so
its circuits are shallower and its noise grows more slowly. The pipeline measures the *effective* scale of every
folded circuit, the ratio of light-cone fault rates $\lambda_{\rm eff}=\lambda_{LC}(\text{folded})/\lambda_{LC}$.
Richardson extrapolation in the nominal $\lambda$ lands at the residual scale

$$\rho=\sum_j\eta_j(\lambda_{\rm nominal})\,\lambda_{{\rm eff},j},\qquad b\approx\rho\,b_{\rm RAW}\ \text{(first order)}.$$

On fake_fez local folding reaches $\lambda_{\rm eff}\approx1,\,2.62,\,4.23$ for nominal $1,3,5$, so $\rho\approx0.19$:
it leaves about a fifth of the raw bias. Its folded circuits are less noisy, so its per-shot variance is smaller.
With $\mathrm{RMSE}^2=b^2+v/N$ the two foldings cross at

$$N_\times=\frac{v_G-v_L}{b_L^2-b_G^2}\propto\lambda_{LC}^{-1},$$

because the bias difference grows like $\lambda_{LC}$ (its square like $\lambda_{LC}^2$) and the variance difference
like $\lambda_{LC}$. Local folding therefore breaks down at a budget that falls as noise × depth grows; it is a bias
effect, not a depth effect. Two limits of $\lambda_{LC}$ as a calibration:

* for global folding the folded circuit's light cone is wider than the original's, so $\lambda_{LC}$ counts faults
  that rarely reach a local observable and overstates the amplification;
* coherent errors add in amplitude when a gate is repeated ($CZ\cdot CZ\cdot CZ$), which a fault count cannot see.

Fitting against $\lambda_{\rm eff}$ (``lam_est: fault-rate``) therefore helps local folding under stochastic noise and
over-corrects under coherent noise (``scripts/dirA_zne_folding.py``, release notes 1.4.1).

### 4.7 Fractional ZNE: which gates, and when it pays (v1.4.1)

A local fold at $\lambda=1+2(k+m/n)$ repeats every two-qubit gate $k$ times and $m$ of the $n$ gates once more. The
estimate depends on *which* $m$: the gates differ in how strongly the observable feels them, so two subsets of the same
size give different $E(\lambda)$ — a deterministic, subset-dependent bias that a high-weight fit multiplies (the
least-squares parabola through $\lambda=1,\tfrac43,\tfrac53,2$ has $\sum_j|\eta_j|=19$). ``subset`` chooses a
*nested* order (the gates folded at a smaller $\lambda$ are folded at every larger one): a seeded permutation
(``nested``), the bit-reversal order of the gate positions (``stratified``: every prefix evenly spread), or the
greedy order whose every prefix has the circuit's mean calibrated error (``balanced``: $\lambda$ in gates
$\approx\lambda$ in noise). ``fold_draws`` $K$ averages $K$ independent nested subsets at the same total shots,

$$\hat E(0)=\sum_j\eta_j\,\frac1K\sum_{k=1}^{K}E_{jk},\qquad N_{jk}\propto|\eta_j|\,\sigma_{jk}/K,$$

which leaves the per-shot variance unchanged and shrinks the subset noise as $1/\sqrt K$ (on fake_fez: $|b|/|b_{\rm
RAW}|$ 0.31 for one subset, 0.135 for $K=4$). At equal budget fractional scales lose to Richardson on $1,3,5$ unless
the circuit's light-cone fault count exceeds $\lambda_{LC}\approx0.6$; above 0.8 they win in 94 % of the cells.

### 4.8 Folding as a measurement of the noise

Because global folding stretches idle windows and local folding repeats the same gate, the two respond differently
to unmodeled noise that is attached to *time* and to noise that builds up *coherently* under repetition. Comparing
how fast the observed and the predicted fidelity decay under each folding gives the **folding fingerprint** (§21.4):
ZNE's own circuits become a measurement of where the noise lives, at no extra cost.

## 5. Probabilistic error cancellation (PEC)

### 5.1 Quasi-probability decomposition

Write each noisy gate as $\mathcal O=\mathcal N\circ\mathcal U$. If the noise $\mathcal N$ is known, its inverse
can be written as a linear combination of implementable operations $\mathcal B_i$,

$$\mathcal N^{-1}=\sum_iq_i\mathcal B_i,\qquad \sum_iq_i=1,\qquad \gamma=\sum_i|q_i|\ge1,$$

where some $q_i<0$ because $\mathcal N^{-1}$ is not a physical map [Temme 2017; Endo 2018]. The ideal expectation is
$\langle O\rangle=\sum_iq_i\langle O\rangle_{\mathcal B_i\circ\mathcal O}$. **Sampling**: pick $\mathcal B_i$ with
probability $|q_i|/\gamma$, run it, and weight the outcome by $\gamma\,\mathrm{sgn}(q_i)$. The estimator is
unbiased for the assumed model; per gate the standard deviation grows by $\gamma_g$, and for the circuit by

$$\Gamma=\prod_g\gamma_g,\qquad N_\varepsilon\approx\frac{\Gamma^2\sigma^2}{\varepsilon^2}.$$

### 5.2 Depolarizing inverse

For $\mathcal D(\rho)=(1-p)\rho+p\,I/d$ every non-identity Pauli fidelity is $f=1-p$. The inverse has Pauli fidelities
$1/f$; by the transform of §2.1 it is the Pauli quasi-mixture

$$q_I=\frac{1+(d^2-1)/f}{d^2},\qquad q_{P\ne I}=\frac{1-1/f}{d^2}<0,\qquad \gamma=\frac{2(d^2-1)/f-(d^2-2)}{d^2},$$

so $\gamma=(3/f-1)/2$ for one qubit and $\gamma=(15/f-7)/8$ for two qubits, $\gamma\approx1+2\tfrac{d^2-1}{d^2}p$ for
small $p$. The **compensation** construction of [Endo 2018] instead writes
$\mathcal U=\tfrac1{1-p}\mathcal O-\tfrac p{1-p}\mathcal T\circ\mathcal O$ with $\mathcal T$ the completely
depolarizing map (a uniformly random Pauli); it is simpler to sample but costs $\gamma=(1+p)/(1-p)$, larger than the
inverse's.

### 5.3 Learned models

Real noise is not depolarizing. Pauli twirling (§8) makes each gate's noise a Pauli channel, whose fidelities can be
learned; IBM learns a **sparse Pauli–Lindblad** model per layer and inverts it [van den Berg 2023]. The app's
*learned* PEC is the idealized version: in simulation it inverts the exact Pauli twirl of each gate's noise; on a fake
device it uses a per-gate depolarizing model with that gate's calibrated error. Without twirling, coherent parts of the
noise are not Pauli and remain uncorrected.

### 5.4 Model mismatch

If PEC is told $p_{\rm assumed}$ while the device has $p_{\rm true}$, each corrected gate leaves a residual Pauli
fidelity $f_{\rm true}/f_{\rm assumed}$ on every Pauli it propagates. Under-estimating the noise under-corrects,
over-estimating it over-shoots the ideal value and costs more ($\gamma$ grows with the assumed rate). Any noise
absent from the model — damping, coherent rotations, readout — is left in place. The **Model mismatch** page maps
these regimes.

### 5.5 How the app computes PEC

On the NumPy executor the correction maps are applied exactly as signed mixtures, giving the exact PEC mean; the
variance is computed from the absolute-value mixture ($\mathbb E[X^2]=\Gamma\sum_x p_{\rm abs}(x)\,o_{\rm eff}(x)^2$,
$o_{\rm eff}$ the observable with the readout mitigation folded in). Aer cannot apply negative weights, so it samples
instances (random Pauli insertions with signs), reports their Monte Carlo mean and estimates $\mathbb E[X^2]$ as
$\Gamma^2$ times the instances' average of $\sum_xp_i(x)o_{\rm eff}(x)^2$. On hardware the app does the same with
explicit Pauli gates.

These variances assume one shot per sampled circuit. With $s$ shots per circuit ($N_t/s$ circuits) the shots of one
circuit share its sign and Pauli insertions, and the per-shot variance becomes
$v_s=v_1+(s-1)\,\mathrm{Var}_{\rm inst}$, where $\mathrm{Var}_{\rm inst}$ is the variance over sampled circuits of
$\Gamma\,\mathrm{sign}_i\langle o\rangle_i$ (a Monte Carlo estimate from the sampled instances). The circuit count in the
cost table is $\lceil N_t/s\rceil$.

## 6. Learning-based mitigation: CDR, FR and NIL

### 6.1 Clifford data regression

Circuits made only of Clifford gates can be simulated classically. **Clifford data regression** [Czarnik 2021] builds
training circuits by replacing the target's non-Clifford rotations with nearby Clifford angles (the app picks
$\theta'\in\{0,\tfrac\pi2,\pi,\tfrac{3\pi}2\}$ with probability $\propto e^{-(\Delta\theta/\sigma)^2}$, $\sigma=1$, and
optionally keeps a few rotations non-Clifford), runs them on the noisy device, computes their ideal values classically
and fits

$$y_{\rm ideal}\approx b+c\,x_{\rm noisy}.$$

The fit is then applied to the target's noisy value. Training circuits share the target's structure, so they feel
the same noise; the method fails when the noise depends on the angles that the training circuits changed, and when the
training set does not span the target (the app reports the rank of the training features and warns when all training
circuits have the same ideal value; §6.3 explains how the training set is chosen so that this does not happen).

### 6.2 Feature regression (FR)

CDR's regression can use more features of the target's own shots than the noisy $\langle O\rangle$. The app's
**feature regression** (FR; parameter value ``features="fr"`` since 1.5 — the legacy value ``features="nil"`` of stored
studies is migrated with a note, see docs/MIGRATION.md; method keys FR-L / FR-R; this project's own heuristic,
not neighbor-informed learning) uses $\mathbf x=(\langle O\rangle,\langle Z_i\rangle,\langle Z_iZ_j\rangle)$ of the noisy distribution and
fits $\hat O=b+\mathbf c\cdot\mathbf x$ by

* ordinary least squares $\min\sum_k(y_k-b-\mathbf c\cdot\mathbf x_k)^2$,
* Ridge, adding $\alpha\lVert\mathbf c\rVert_2^2$, or
* Lasso, adding $\alpha\lVert\mathbf c\rVert_1$, which keeps few features.

FR is a heuristic of this project, not a published protocol. The closest published work is ML-QEM [Liao 2024], which
feeds noisy expectation values and circuit descriptors into regression models; data-driven unifications of ZNE and
CDR are vnCDR and UNITED [Lowe 2021; Bultrini 2023]. FR is **not** the learning-based method of [Strikis 2021], whose
features are noisy values of *Pauli-inserted variants* of the circuit — that construction is NIL (§6.3).

Because $\hat O$ is linear in the measured frequencies, its per-shot variance is that of $\sum_xg(x)$ with
$g=F\mathbf c$; the overhead is at most $\gamma=\lVert\mathbf c\rVert_1$. Training shots are a fixed part of the
budget, and their shot noise makes the fit itself noisy — a floor $u$ that more target shots cannot lower. The fit is
non-linear in the data (and with collinear features a first-order value is not even defined), so $u$ is the variance
of the estimate over refits on resampled training data (Monte Carlo). The readout calibration acts on every training
feature and on the target at once; its coefficient $w$ ($\mathrm{Var}=w/s$) is measured the same way, by paired refits
with and without a sampled calibration. The training set counts only feature directions whose spread exceeds the shot
noise of the training circuits (singular values of the centered features above
$\sigma(\sqrt{n_{\rm train}}+\sqrt{n_{\rm features}})$, decided at every number of training shots); with none,
learning is not applied and the unmitigated value is reported.

### 6.3 Neighbor-informed learning (NIL)

**Neighbor-informed learning** [Chen 2025] scales up the idea of [Strikis 2021]. A *neighbor* of the target circuit
$U=U_L\cdots U_1$ is a copy with one Pauli inserted right after one gate, $U_j\to PU_j$, $P\in\{X,Y,Z\}$ (a weight-1
neighbor), or the target at amplified noise (ZNE neighbors). With $s$ neighbors drawn at random, the estimate is a
linear combination of their noisy values,

$$\hat O=\sum_{j=1}^{s}c_j\,\langle\widetilde C_j\rangle,\qquad \mathbf c=\arg\min_{\lVert\mathbf c\rVert_1\le\gamma}\ \sum_{i=1}^{T}\Bigl(\sum_jc_j\,x^{(i)}_j-y^{(i)}\Bigr)^2,$$

learned on $T$ training circuits in which every non-Clifford rotation angle is replaced by a uniform draw from
$\{0,\pi/2,\pi,3\pi/2\}$ (a rotational 2-design; Chen et al.'s Algorithm 2) and which carry the same insertions:
$x^{(i)}_j$ is the noisy value of neighbor $j$ of training circuit $i$, $y^{(i)}$ its ideal value. Chen et al. fit
this ℓ1-constrained least squares without an intercept ($\gamma=2$ for Pauli neighbors, $\gamma=5$ for ZNE
neighbors), with about $10^3$ training circuits and the same number of shots ($10^4$) for every neighbor circuit.

The app implements the protocol (``nil``) with the regression rule as a choice — Chen et al.'s ℓ1 constraint, Lasso,
Ridge, OLS, elastic net, non-negative least squares, with or without an intercept — and the training ensemble as a
choice (Chen et al.'s 2-design, uniform angles, CDR's near-Clifford angles). OLS (minimum norm) and Ridge are solved
exactly; the ℓ1 rules are solved to a certificate: a duality or Frank–Wolfe gap bounds the excess objective by
$10^{-10}\lVert y\rVert^2$. For them a small stabilization $\varepsilon\lVert\mathbf c\rVert_2^2$,
$\varepsilon=(10^{-6}\lVert X\rVert_2)^2$, makes the solution unique. Neighbors whose values coincide on every training
circuit (a Pauli that commutes to the measurement of all of them) share their weight equally under every rule instead
of by a solver's tie-break, while they may differ on the target.

Cost. Training needs $T\cdot s$ circuits. Given $\mathbf c$, the target shots are split $N_j\propto|c_j|\sigma_j$
(``target_split="optimal"``; $\sigma_j$ the per-shot standard deviation of neighbor $j$), so
$\mathrm{Var}=(\sum_j|c_j|\sigma_j)^2/N\le\gamma^2\max_j\sigma_j^2/N$ — the ℓ1 bound caps the shot cost — and
circuits with $c_j=0$ are not run; or equally, $N/s$ shots each (Chen et al.'s plan;
$\mathrm{Var}=s\sum_jc_j^2\sigma_j^2/N$). The optimal split needs $\mathbf c$ before the target circuits are sent, so on
hardware, where a plan is sent as one batch, only the equal split is planned. As for FR, the training-shot variance
$u$ and the calibration coefficient $w$ are measured by Monte Carlo refits; in the equal-budget pipeline they are
measured at every training level, with the target variance averaged over the refitted coefficients.

### 6.4 Designing the training set: why CDR can collapse, and the adaptive strategy

For training pairs $(x_k,y_k)$ ($x$ noisy, $y$ ideal) the least-squares slope and intercept are

$$\hat c=\frac{\sum_k(x_k-\bar x)(y_k-\bar y)}{\sum_k(x_k-\bar x)^2},\qquad \hat b=\bar y-\hat c\,\bar x .$$

If every training circuit has the **same ideal value**, $y_k\equiv y_0$, the covariance vanishes, $\hat c=0$ and
$\hat O=\hat b=y_0$ for *every* target: the "mitigated" result is a constant that carries no information about the
target (often exactly $0.0000$ or $1.0000$), and a shot-noise simulation of it has zero variance. This is not
overfitting in the usual sense but an **unidentifiable** regression. It happens when the observable is blind to the
Clifford replacements — e.g. a single $\langle Z_q\rangle$ at the end of phase estimation, where every near-Clifford
training circuit returns the same basis state. With several features (FR, NIL) the analogous failure is a rank-deficient
design matrix: directions the training set never explores are extrapolated blindly.

The app therefore chooses the training set **before** any noisy circuit is run, using only classical (ideal)
simulations, and escalates until the spread of the training values reaches a threshold,
$\mathrm{sd}(y)\ge s_{\min}\,(\max_x o(x)-\min_x o(x))$ (default $s_{\min}=0.05$):

1. $k$ non-Clifford gates kept (the user's value, default 0), near-Clifford angles ($\sigma=1$);
2. the same $k$ with Clifford angles drawn uniformly (more diverse circuits);
3. $k+1$, $k+2$, … kept non-Clifford gates, each with both samplings (training circuits closer to the target, at a
   classical cost that grows exponentially with $k$ for stabilizer simulators [Czarnik 2021]);
4. every rotation redrawn as a random Clifford angle;
5. the same with a random computational basis state as input (X gates on a random subset of qubits).

The first stage that passes is used and the stages tried are reported. Because the choice depends only on the
circuit, the observable and the seed, a hardware run records and replays exactly the same training circuits. If no
stage gives varying training values — for instance $\langle Z_i\rangle=0$ for every QAOA circuit, by the
$X^{\otimes n}$ symmetry of $|+\rangle^{\otimes n}$ and of both QAOA unitaries — the regression is **not applied**:
the unmitigated value is returned, the table's *status* column says so, and a warning replaces the regression plot.
The *fixed* strategy reproduces plain CDR with exactly the chosen $k$ and $\sigma=1$, for comparison.

## 7. Readout mitigation

### 7.1 Tensored inverse

Calibrate each bit with two circuits (all qubits in $|0\rangle$; all in $|1\rangle$) to obtain $A_i$ (§2.5), then apply
$\bigotimes_iA_i^{-1}$ to the measured distribution [Bravyi 2021]. The result is unbiased for uncorrelated readout
errors but is a quasi-distribution (it can have negative entries) and amplifies the variance by
$\prod_i\frac{1+|\epsilon_{01}-\epsilon_{10}|}{1-\epsilon_{01}-\epsilon_{10}}$.

### 7.2 M3

The full inverse is exponentially large. **Matrix-free measurement mitigation** (M3) solves the assignment equations
only on the bitstrings actually observed (optionally within a Hamming distance), with the reduced matrix's columns
renormalized, so the cost scales with the number of distinct outcomes rather than $2^n$ [Nation 2021].

### 7.3 TREX

**Twirled readout error extinction** randomly flips each qubit with an X gate before measurement and flips the recorded
bit back [van den Berg 2022]. Averaged over the flips, the assignment matrix becomes symmetric with flip probability
$\bar p_i=\tfrac12(\epsilon_{01}+\epsilon_{10})$, so every Z-string expectation is scaled by a known factor:

$$\langle Z_S\rangle_{\rm meas}=\prod_{i\in S}(1-2\bar p_i)\,\langle Z_S\rangle,$$

which a single calibration circuit ($|0\cdots0\rangle$ with the same twirl) measures. Dividing removes readout bias
without modeling asymmetric or correlated structure.

## 8. Pauli twirling and randomized compiling

Surrounding a Clifford gate $G$ with random Paulis $P$ before and $P'=GPG^\dagger$ after leaves the ideal gate unchanged.
Averaged over $P$, the gate's noise channel becomes its **Pauli twirl**: only the diagonal of its PTM survives
[Knill 2005; Wallman 2016]. Coherent errors become stochastic Pauli errors, which accumulate in probability rather than
amplitude and are exactly what Pauli-model PEC and learned models assume. **Randomized compiling** twirls all gate
layers and merges the Paulis into neighboring one-qubit gates [Hashim 2021]. The app applies the twirl exactly (the
average over all twirl instances) by replacing each Clifford gate's noise with its PTM diagonal; the Circuits tab shows
one random instance.

## 9. Dynamical decoupling

An idle qubit accumulates phase from slow frequency fluctuations. Inserting $\pi$ pulses reverses the sign of that
phase halfway, so a static detuning cancels [Viola 1998]. The app places

* **X2** (CPMG-like): X at $T/4$ and $3T/4$ of an idle window of length $T$;
* **XY4**: X, Y, X, Y at $T/8,3T/8,5T/8,7T/8$, which also compensates pulse errors to first order.

In the toggling frame the pulses multiply the noise by a switching function $y(t)=\pm1$; the accumulated phase
$\int\delta(t)y(t)\,dt$ vanishes for constant $\delta$. DD filters low-frequency noise; it cannot remove Markovian
(white) noise and adds the pulses' own errors [Ezzell 2023]. The Time-travel tab shows a static field being refocused.

## 10. Symmetry verification

If the ideal state is an eigenstate of a symmetry — here a Z-parity $Z_S$ with eigenvalue $s=\pm1$ — shots violating
it must be errors and are discarded [Bonet-Monroig 2018; McArdle 2019]:

$$\hat O=\frac{\sum_{x:\,\mathrm{par}_S(x)=s}o(x)\,\hat p(x)}{\sum_{x:\,\mathrm{par}_S(x)=s}\hat p(x)}.$$

The cost is the discarded fraction: the effective shots are $NP_{\rm accept}$, so $\gamma=P_{\rm accept}^{-1/2}$. Errors
that preserve the symmetry are not detected. The app warns when the ideal state is not an eigenstate of the chosen
parity (post-selection then biases the answer). In HHL the same machinery post-selects the ancilla.

## 11. Virtual distillation

Given $M$ copies of $\rho=\sum_k\lambda_k|\psi_k\rangle\langle\psi_k|$, an entangling **derangement** (cyclic shift)
circuit measures $\mathrm{Tr}(\rho^MO)$ and $\mathrm{Tr}(\rho^M)$ [Huggins 2021; Koczor 2021]:

$$\langle O\rangle_M=\frac{\mathrm{Tr}(\rho^MO)}{\mathrm{Tr}(\rho^M)}\xrightarrow[M\to\infty]{}\langle\psi_1|O|\psi_1\rangle.$$

Incoherent errors are suppressed exponentially in $M$; a coherent error that rotates the dominant eigenvector
$|\psi_1\rangle$ is not (the "coherent mismatch" floor). The cost is $M$ times the qubits and a variance amplified by
about $\mathrm{Tr}(\rho^M)^{-2}$. The app evaluates the formula from the exact density matrix with ideal copies and
derangement and shows the spectrum of $\rho$.

## 12. Test circuits

The templates are the benchmarks every method is tested on: they range from two-qubit states to circuits with hundreds
of two-qubit gates, and every one has a known ideal answer.

### 12.1 Entanglement and phase kickback

#### Bell and GHZ states

$|\Phi^+\rangle=(|00\rangle+|11\rangle)/\sqrt2$ and its $n$-qubit version; the parity $Z_0Z_{n-1}=+1$
is a symmetry.

<!-- fig:ghz -->

#### Bernstein–Vazirani

One query of $f(x)=s\cdot x \bmod 2$, with the ancilla in $|-\rangle$ (phase kickback), gives
$H^{\otimes n}U_fH^{\otimes n}|0\rangle^{\otimes n}=|s\rangle$ [Bernstein 1997]; the observable is $P(s)$.

<!-- fig:bv -->

### 12.2 Fourier-based algorithms

#### Quantum Fourier transform

$F|j\rangle=2^{-n/2}\sum_ke^{2\pi ijk/2^n}|k\rangle$. The textbook circuit is Hadamards and controlled phases
$\mathrm{CP}(\pi/2^d)$ followed by a SWAP network that reverses the bit order [Nielsen 2010]. The template runs
$F\,F^\dagger|s\rangle=|s\rangle$ (the input $F^\dagger|s\rangle$ is a product state) in three ways: with the SWAP network;
without SWAPs, loading the Fourier state of the bit-reversed $s$; or without SWAPs, relabeling the output bits
classically. The last two save $3\lfloor n/2\rfloor$ CX gates.

<!-- fig:qft -->

#### Approximate QFT

Controlled phases $\pi/2^d$ with $d>m$ are dropped [Coppersmith 1994]; the error is
$O(n2^{-m})$ in operator norm while the gate count falls from $O(n^2)$ to $O(nm)$, and under noise the shorter circuit
can be more accurate than the exact one [Barenco 1996]. Here $m$ is the largest qubit distance $d$ that is kept; in
terms of $R_k=\mathrm{CP}(2\pi/2^k)$ the circuit keeps $k\le m+1$ (conventions for the "degree" of the AQFT differ by
one between papers). The approximation error depends on the input state, so the bit-reversed-input variant (which
feeds the truncated circuit the Fourier state of $\mathrm{rev}(s)$) has a different success probability from the SWAP
and relabel variants.

#### Phase estimation

$t$ counting qubits, controlled $U^{2^k}$, inverse QFT [Kitaev 1995]; with $U=P(2\pi\varphi)$ and
$\varphi=j/2^t$ the output is exactly $j$. **QPE of an Ising Hamiltonian** uses $U=e^{-iJZ_aZ_b\tau}$ on the eigenstate
$|ab\rangle$; controlled-$U^{2^k}$ is $\mathrm{CX}\cdot\mathrm{CRZ}(2J\tau2^k)\cdot\mathrm{CX}$.

<!-- fig:qpe -->

#### Trotterized Ising dynamics

$H=-J\sum_iZ_iZ_{i+1}-h\sum_iX_i$, first-order Trotter steps
$\prod_iR_{ZZ}(-2J\Delta t)\prod_iR_X(-2h\Delta t)$; the observable is the magnetization $\langle\sum_iZ_i\rangle/n$.

<!-- fig:ising -->

### 12.3 NISQ: variational circuits

#### VQE with a hardware-efficient ansatz

The variational quantum eigensolver minimizes
$E(\boldsymbol\theta)=\langle\psi(\boldsymbol\theta)|H|\psi(\boldsymbol\theta)\rangle$ over a parametrized state
[Peruzzo 2014]. The hardware-efficient ansatz uses only native operations [Kandala 2017]:
$|\psi\rangle=\prod_{\ell=1}^{L}\bigl[\bigotimes_qR_Z(\phi_{\ell q})R_Y(\theta_{\ell q})\,U_{\rm ent}\bigr]
\bigotimes_qR_Z(\phi_{0q})R_Y(\theta_{0q})|0\rangle^{\otimes n}$ with a linear, circular or all-to-all CX entangler
$U_{\rm ent}$ ($2n(L+1)$ angles). The template's Hamiltonian is the antiferromagnetic chain
$H=\sum_iZ_iZ_{i+1}$, whose ground energy is $-(n-1)$ (Néel states); the observable is $\langle H\rangle/(n-1)\in[-1,1]$.
With *VQE-optimized* angles the app minimizes $E$ on a noiseless statevector with L-BFGS-B when the template loads,
so the ideal value is the optimum ($-1$) and every deviation in the comparison is noise — the setting in which
Kandala et al. applied ZNE to VQE [Kandala 2019]. *Random* angles give a generic HEA state instead.

<!-- fig:vqe -->

#### QAOA for MaxCut

For a graph $G=(V,E)$ the cut size is $C(x)=\sum_{(i,j)\in E}[x_i\neq x_j]
=\sum_{(i,j)}\tfrac12(1-Z_iZ_j)$. QAOA prepares $|\boldsymbol\gamma,\boldsymbol\beta\rangle=\prod_{\ell=1}^{p}
e^{-i\beta_\ell B}e^{-i\gamma_\ell C}|+\rangle^{\otimes n}$ with the mixer $B=\sum_iX_i$ [Farhi 2014]. Up to a global phase
$e^{-i\gamma C}=\prod_{(i,j)}R_{ZZ}(-\gamma)$ and $e^{-i\beta B}=\prod_iR_X(2\beta)$, which is how the template is built.
The observable is the approximation ratio $\langle C\rangle/C_{\max}$. Graphs: ring, path, complete, the 3-regular
Möbius ladder (ring plus diameters, $n$ even) and a seeded random $G(n,\tfrac12)$. For a ring with $n\ge4$ at $p=1$
the optimum is $\langle C\rangle/|E|=3/4$ [Farhi 2014], which equals the ratio $\langle C\rangle/C_{\max}$ for even $n$
(an odd ring has $C_{\max}=|E|-1$: 0.9375 for $n=5$); the optimizer reproduces these values.

<!-- fig:qaoa -->

#### Quantum machine learning

The *ZZ feature map* encodes $x\in\mathbb R^n$ as
$U_\Phi(x)=\bigl[e^{i\sum_{i}\phi_{i,i+1}(x)Z_iZ_{i+1}}e^{i\sum_i\phi_i(x)Z_i}H^{\otimes n}\bigr]^{r}$ with
$\phi_i=x_i$, $\phi_{ij}=(\pi-x_i)(\pi-x_j)$ [Havlíček 2019]. The template follows Qiskit's `ZZFeatureMap` with
`entanglement="linear"` gate by gate — $P(2\phi_i)$ on each qubit and $\mathrm{CX}\,P(2\phi_{ij})\,\mathrm{CX}$ for
neighbors, which implements the exponentials above with the opposite sign, $e^{-i\phi Z}$ and $e^{-i\phi ZZ}$, up to a
global phase (the template's formula shows that sign); a RealAmplitudes-type
layer $W(\boldsymbol\theta)$ follows and the classifier output is $\langle Z^{\otimes n}\rangle$. The *data re-uploading*
classifier interleaves data and weights in every layer, $R_Z(w'_{\ell q}x_q+b'_{\ell q})R_Y(w_{\ell q}x_q+b_{\ell q})$ followed by
a CZ chain, and reads $\langle Z_0\rangle$ [Pérez-Salinas 2020]. Data and weights come from seeds: for error mitigation the
circuit structure — depth, entangling pattern, non-Clifford content — is what matters, not whether the model is trained.

<!-- fig:qml -->

### 12.4 Fault tolerance: error-correction primitives

#### Three-qubit bit- and phase-flip codes

[Shor 1995; Nielsen 2010, §10.1]. $|\psi\rangle=\cos\tfrac\theta2|0\rangle+
\sin\tfrac\theta2|1\rangle$ is encoded as $\cos\tfrac\theta2|000\rangle+\sin\tfrac\theta2|111\rangle$ by two CX gates (the phase-flip
code adds $H^{\otimes3}$, so $Z$ errors act as $X$ errors in that basis). Two ancillas measure the stabilizers $Z_0Z_1$
and $Z_1Z_2$; the syndrome $(s_1,s_2)=(1,0),(1,1),(0,1)$ points to a flip of $q_0,q_1,q_2$. In *syndrome + correction*
mode three Toffoli gates apply that correction coherently (no mid-circuit measurement), the code is decoded and
$\langle Z_0\rangle=\cos\theta$ is restored after any single error. *Syndrome only* extracts but ignores the syndrome;
*majority vote readout* measures the three data qubits and decodes classically. An error can be placed on one qubit
or as a bit-/phase-flip channel of probability $p$ on every data qubit; the logical error rate is then
$p_L=3p^2-2p^3$ instead of $p$.

<!-- fig:qec3 -->

#### Repetition-code memory

A distance-$d$ bit-flip code with $r$ rounds of $Z_iZ_{i+1}$ checks into fresh ancillas and
a bit-flip channel with probability $p$ before every round and once more before readout ($r+1$ exposures, so each data
qubit ends flipped with $p_{\rm eff}=\tfrac12[1-(1-2p)^{r+1}]$); the final data bits are decoded by majority vote,
which fails with probability $\sum_{j>d/2}\binom dj p_{\rm eff}^j(1-p_{\rm eff})^{d-j}$ — exponentially small in $d$
below the threshold $p_{\rm eff}<1/2$. The circuit
noise of the chosen model (the CX gates of the checks) competes with this suppression, which is the essence of the
fault-tolerance threshold.

<!-- fig:repetition -->

#### Steane [[7,1,3]] code

[Steane 1996]. The stabilizers are $X$ and $Z$ on the supports of the rows of the Hamming
check matrix, $\{3,4,5,6\},\{1,2,5,6\},\{0,2,4,6\}$ (0-indexed qubits; column $q$ is the binary number $q+1$).
$|0_L\rangle=2^{-3/2}\sum_{c\in\mathcal C^\perp}|c\rangle$ is the uniform superposition of the 8 codewords of the dual
(simplex) code. The encoder places $|\psi\rangle$ on $q_2$, applies the logical-$X$ representative $X_2X_4X_5$ controlled by
it, then Hadamards on the pivots $q_0,q_1,q_3$ and CX gates from each pivot to the rest of its row. The logical $Z$ is
read transversally as the parity of all 7 bits; the *decoded* readout first computes the Hamming syndrome of the
measured bits, flips the indicated bit, and then takes the parity — so any single $X$ error leaves
$\langle Z_L\rangle=\cos\theta$ unchanged, whereas the raw parity flips sign. The stabilizer $Z_3Z_4Z_5Z_6=+1$ is offered
as the symmetry for verification (post-selection on a stabilizer is error *detection*).

<!-- fig:steane -->

#### Five-qubit code [[5,1,3]]

[Laflamme 1996; Bennett 1996]. Stabilizers $XZZXI$ and its cyclic shifts; $Z_L=Z^{\otimes5}$,
$X_L=X^{\otimes5}$. The encoder is a Clifford circuit $U$ (11 CX) synthesized from the tableau $U Z_0U^\dagger=Z_L$,
$UX_0U^\dagger=X_L$, $UZ_jU^\dagger=S_j$. After an error $E$, the inverse encoder gives
$U^\dagger EU=P_0\otimes Q$: a logical Pauli $P_0$ on $q_0$ and, on $q_1\dots q_4$ (prepared in $|0\rangle$), bit flips
exactly where $E$ anticommutes with $S_j$ — the syndrome. Measuring all five qubits therefore yields the syndrome and
the logical bit; the *decoded* observable flips the logical bit when the syndrome's error acts as $X$ or $Y$ on it
(a 16-entry lookup table: the code is perfect, $2^4=1+3\cdot5$). Every single-qubit $X$, $Y$ or $Z$ error leaves
$\langle Z_L\rangle=\cos\theta$.

<!-- fig:five -->

#### Surface code, one plaquette ([[4,1,2]])

Data $q_0q_1/q_2q_3$ on a $2\times2$ patch; stabilizers $X_0X_1X_2X_3$ (the
plaquette) and the boundary checks $Z_0Z_1$, $Z_2Z_3$; $Z_L=Z_0Z_2$, $X_L=X_0X_1$;
$|0_L\rangle=(|0000\rangle+|1111\rangle)/\sqrt2$, prepared by CX, H and CX gates from $RY(\theta)|0\rangle$ on $q_0$. With
distance 2 any single-qubit error is detected but none is correctable. *Transversal* readout computes the $Z$ checks from
the data bits; *ancilla extraction* measures the plaquette on $q_4$ (it flags $Z$ errors) and the $Z$ checks on
$q_5,q_6$. Symmetry verification post-selects on both $Z$ checks, which removes every single $X$ or $Y$ error — the
errors that can flip $Z_L$ — while weight-2 errors such as $X_0X_1=X_L$ remain undetectable, as distance 2 implies.

<!-- fig:surface -->

### 12.5 Fault tolerance: deep algorithms

#### Shor's order finding for N = 15

With $t$ counting qubits and a 4-qubit work register in $|1\rangle$,
controlled multiplications by $a^{2^k}\bmod15$ and an inverse QFT give outcomes $y\approx s2^t/r$ from which continued
fractions recover the order $r$ of $a$ [Shor 1997]. The multiplications are "compiled" (the powers are computed
classically), as in the first experimental demonstrations [Vandersypen 2001]. The observable is the probability of an
outcome that yields $r$.

<!-- fig:shor -->

**Deep variants.** Besides the compiled circuit, controlled-$U^{2^k}$ can be implemented as $2^k$ repeated
controlled multiplications by $a$ ($2^t-1$ in total, as a circuit that does not know the order would do), and each
controlled multiplication can be the textbook SWAP/NOT network or a generic unitary synthesis of the controlled
$16\times16$ permutation $|y\rangle\mapsto|ay\bmod15\rangle$ (hundreds of CX each). The ideal output distribution is the
same; the depth grows by one to two orders of magnitude, which is the regime where error *mitigation* stops working and
error *correction* becomes necessary [Takagi 2022; Quek 2024].

#### Grover's search

Grover's algorithm finds one marked item $m$ among $N=2^n$ with $O(\sqrt N)$ oracle calls, quadratically fewer than
any classical search and optimal for unstructured search [Grover 1996; Bennett 1997b]. In the plane spanned by $|m\rangle$
and the uniform superposition $|s\rangle=\sin\theta|m\rangle+\cos\theta|m^\perp\rangle$, $\sin\theta=1/\sqrt N$, the oracle
$O_m=I-2|m\rangle\langle m|$ and the diffuser $D=2|s\rangle\langle s|-I$ are reflections whose product rotates by $2\theta$:

$$P(m)=\sin^2\big((2k+1)\theta\big),\qquad k^\ast=\lfloor\pi/(4\theta)\rfloor .$$

The template builds $O_m$ from $X$ gates around a multi-controlled $Z$ and $D=H^{\otimes n}X^{\otimes n}C^{n-1}Z\,X^{\otimes n}
H^{\otimes n}$ (up to a global phase); the $(n-1)$-controlled $Z$ is a CZ for $n=2$, the 6-CX Clifford+T circuit for $n=3$
and Qiskit's synthesis beyond. Iterating past $k^\ast$ rotates beyond the target [Boyer 1998]. The observable is $P(m)$.
The two-qubit *Grover* template (Basics) is the case $N=4$, $k^\ast=1$, which finds the marked item $|11\rangle$ with certainty.

<!-- fig:grover -->

#### The hidden subgroup problem

A function $f$ on a group $G$ hides a subgroup $H$ if it is constant on the cosets of $H$ and distinct on different cosets.
For abelian $G$ the same Fourier-sampling circuit solves every instance: prepare $\sum_x|x\rangle|f(x)\rangle$, apply the
quantum Fourier transform of $G$ to the input register, and measure; the outcome is a uniformly random character trivial
on $H$, i.e. an element of the annihilator $H^\perp$, and $O(\log|G|)$ samples determine $H$ [Jozsa 2001; Childs 2010].
*Simon's algorithm* (the instance $G=\mathbb Z_2^n$, $H=\{0,s\}$; its own template under Algorithms) finds $s$ with $f(x)=f(x\oplus s)$ [Simon 1997].
Every measured $y$ satisfies $y\cdot s=0$, i.e. the parity of the input bits selected by $s$ is $+1$ — an exact symmetry. The template *Hidden subgroup problem* generalizes it in two directions:
$G=\mathbb Z_2^n$ with a hidden subgroup of any dimension, $H=\mathrm{span}\{h_1,h_2\}$, whose oracle $f(x)=Ax$ uses a basis
of $H^\perp$ as the rows of $A$ (so $\ker f=H$), and the cyclic group $G=\mathbb Z_N$, $N=2^n$, with $H=\langle2^j\rangle$,
$f(x)=x\bmod 2^j$ and the QFT over $\mathbb Z_N$ — period finding, the subroutine behind Shor's algorithm. The
observables are $P(y\in H^\perp)$ and $P(y\equiv0\bmod N/2^j)$, both ideally 1; for $\mathbb Z_2^n$ every generator of
$H$ is a $Z$-parity that symmetry verification can check.

<!-- fig:hsp -->

#### HHL

For $A=\begin{pmatrix}1&-1/3\\-1/3&1\end{pmatrix}$ and $|b\rangle=|0\rangle$, phase estimation with
$e^{iAt_0}$, $t_0=3\pi/4$, writes the eigenvalues $2/3,4/3$ as 1 and 2 on two clock qubits; controlled $R_Y$ rotations
put $C/\lambda$ into an ancilla amplitude; uncomputation returns the clock [Harrow 2009; Cao 2012]. Given ancilla $=1$ the
system qubit holds $|x\rangle\propto A^{-1}|b\rangle=(3|0\rangle+|1\rangle)/\sqrt{10}$, so $\langle Z\rangle=0.8$.

<!-- fig:hhl -->

## 13. Compilation

The **logical** circuit is what you draw plus input-state preparation and measurement-basis changes. The
**implemented** circuit is what a device runs:

* **abstract** mode keeps the circuit as drawn;
* **ISA** mode transpiles with Qiskit to $\{R_Z,\sqrt X,X\}$ plus CX or CZ on a chosen coupling map (all-to-all, line,
  ring), with SABRE layout and routing [Li 2019] or a trivial layout, and optimization levels 0–3;
* **device** mode (a fake backend) transpiles for that processor's target (§2.6).

Routing inserts SWAPs (3 two-qubit gates each) and permutes which qubit ends up measured into which classical bit; the
app tracks this **measurement map** so that observables refer to the right bits. $R_Z$ is virtual (a frame change)
and error-free; $\sqrt X$ and $X$ are physical pulses.

## 14. The simulators

### One model, two executors

`resolve()` turns a circuit and a noise specification into one instruction list
(unitaries, Kraus maps, and mixtures that may carry negative weights). The NumPy executor applies it to a density matrix
held as a tensor with $2n$ axes; Qiskit Aer runs the same list in `density_matrix` mode. They agree to machine precision.
Both return exact outcome distributions, from which the app computes exact means and variances; repeated experiments are
simulated by multinomial sampling.

### Time travel

stores the state after every step. When $2\times(\text{steps}+1)$ density matrices would take a large
share of memory, it switches to a compact mode that keeps per-step summaries (Bloch vectors, fidelity with the ideal
state vector, purity).

### Resources

A density matrix of $n$ qubits takes $16\cdot4^n$ bytes (16 MB at $n=10$, 4.3 GB at $n=14$). The app has
no limit on qubits or depth; it estimates memory before heavy runs, catches out-of-memory errors, and stops cleanly at a
time limit you set (0 means none).

## 15. Running on hardware

The IBM Quantum page runs the same estimators on a real device — or rehearses locally with the Sampler on a fake
backend — by *record → allocate → execute → replay*:

1. **Record.** Each estimator runs once against a recorder that notes every circuit it asks for, with its role
   (target, training or calibration), and answers with the circuit simulated under the device's calibrated median
   error rates.
2. **Allocate.** The recorded result gives each contender's cost model, and its budget $B$ is split exactly as in
   simulation: $s^*$ shots per calibration circuit, the method's training shots, and the target shots as the
   estimator's variance assumes — ZNE's $N_j\propto|\eta_j|\sigma_j$, a Pauli sum's $N_g\propto\sqrt{v_g}$, PEC's $K$
   sampled circuits with $N_t/K$ shots each. Contenders never share counts: each is measured on its own budget.
3. **Execute.** The circuits are compiled for the backend's physical qubits and sent with the Sampler primitive, one
   job per option group and shot count (gate twirling and dynamical decoupling as primitive options; TREX with
   explicit random X gates before measurement). The local rehearsal runs the same Sampler calls on Qiskit Aer with
   the device's gate and readout errors on the qubits the circuits use.
4. **Replay.** The estimators run again on the measured frequencies.

The result has the simulator's structure, so the page shows the Playground's figures, analysis views and cost table,
and its ZIP holds the same files plus the plan and the job IDs. The *sd* column is the standard error of the
allocation actually run, $\sqrt{v/N_t+w/s+u}$ with $v$, $w$, $u$ from the measured data; *sd (repeats)* resamples
the measured counts. The API token is used once to connect and is never stored or exported. The divergence study
runs its own plans on hardware through the same discipline; its protocol is in §21.6.

## 16. Advanced mitigation methods

The methods of §4–§11 are the ones the app simulates end to end. This section surveys the wider toolbox a thesis on
mitigation has to position itself in: how each method works, what it assumes, what it costs, and how it relates to
the others. All of them are estimators of $\langle O\rangle=\mathrm{Tr}(\rho O)$ built from a noisy device; they differ
in which *prior knowledge* they spend (a symmetry, a noise model, a scalable noise knob, extra copies, a subspace)
and in how the sampling overhead $\gamma$ grows.

### 16.1 Symmetry verification and symmetry expansion

#### Setting

Let $\{S\}$ be a group of commuting symmetries of the ideal state, $S|\psi\rangle=s|\psi\rangle$ — a
parity $Z^{\otimes n}$, a particle number $N=\sum_j(1-Z_j)/2$, a total spin, or the stabilizers of a code. The
projector onto the correct symmetry sector is $\Pi_s=\frac1{|G|}\sum_{g\in G}\chi_s(g)\,g$.

#### Post-selection

(§10) prepares $\rho_{\rm SV}=\Pi_s\rho\Pi_s/\mathrm{Tr}(\Pi_s\rho)$: every error that leaves the
sector is removed. When the symmetry is diagonal in the measured basis it costs nothing but discarded shots,
$\gamma^2=1/P_{\rm acc}$ with $P_{\rm acc}=\mathrm{Tr}(\Pi_s\rho)$. If the symmetry is not diagonal (e.g. particle number
in a Jordan–Wigner encoding measured in a rotated basis), it can be verified *in post-processing* [Bonet-Monroig
2018; McArdle 2019]: because $\Pi_s$ is a sum of Pauli operators,

$$\langle O\rangle_{\rm SV}=\frac{\mathrm{Tr}(\Pi_s\rho\,\Pi_sO)}{\mathrm{Tr}(\Pi_s\rho)} =\frac{\sum_{g,g'}\chi_s(g)\chi_s(g')\,\mathrm{Tr}(g\rho g' O)}{|G|\sum_g\chi_s(g)\mathrm{Tr}(g\rho)},$$

and for commuting $O$ the numerator reduces to $\langle\Pi_sO\rangle$, a sum of Pauli expectation values.

#### Symmetry expansion

[Cai 2021] generalizes the construction: any weighted combination
$\langle O\rangle_{\rm SE}=\sum_g w_g\langle gO\rangle/\sum_g w_g\langle g\rangle$ is an estimator, and choosing the weights
trades bias against variance (post-selection is one choice; others use only a subset of group elements to cut the
sampling cost).

#### What it cannot do

Errors that commute with every symmetry (a $Z$ error for a $Z$-parity, a logical error for a
code) are invisible; the residual bias is the probability of an undetectable error. Symmetry verification is
*error detection*: with code stabilizers as the symmetries it is exactly post-selected QEC (§17.5).

**In the app:** *Symmetry verification* post-selects on a Z-parity; the QEC templates set their stabilizer (Steane
$Z_3Z_4Z_5Z_6$; for the $d=2$ surface code both $Z$ checks together, via the method's *further parities*) so detection
can be compared with correction. Several parities can be verified at once; the accepted shots are those in which every
check reads $+1$.

### 16.2 Zero-noise extrapolation at the pulse level

ZNE needs a knob $\lambda$ that scales the noise while leaving the ideal operation unchanged. Two families exist:

* **Pulse stretching** (the original proposal [Temme 2017] and the first large demonstration [Kandala 2019]). Every
  gate is re-calibrated with its drive stretched in time by $c_j$ and its amplitude reduced by $1/c_j$, so the unitary
  is unchanged. If the noise is time-invariant and weak — Lindbladian $\mathcal L$ acting during the pulse — the
  expectation value is an analytic function of the noise strength,
  $E(\lambda)=E_0+\sum_{k\ge1}a_k\lambda^k$, and Richardson extrapolation from $\lambda\in\{c_1,\dots,c_m\}$ cancels
  the first $m-1$ orders: the residual bias is $O(\lambda^m)$ (for the exact conditions see [Temme 2017]).
  Stretching scales the noise that accrues *in time* during the gates — relaxation, dephasing, always-on $ZZ$
  coupling, detunings — but not a fractional amplitude miscalibration (it keeps the same relative error at any
  stretch), nor readout or state-preparation errors. The price is calibration: every stretch factor needs its own
  pulse calibration.
* **Digital folding** (§4.2) [Giurgica-Tiron 2020]: $U\to U(U^\dagger U)^k$ inserts gates. No recalibration is needed and
  it runs on any gate-based device, but it scales noise only in the Markovian, gate-local picture; coherent errors
  can partly cancel in $U^\dagger U$ and non-Markovian noise does not scale linearly.
* **Probabilistic error amplification** (PEA) [Kim 2023b]: learn a sparse Pauli–Lindblad model of each layer
  [van den Berg 2023] and *add* sampled Pauli errors to reach $\lambda>1$ exactly in that model. It combines the
  model accuracy of PEC with the modest overhead of ZNE and was used for the 127-qubit utility experiment.

The variance factor is $(\sum_j|\eta_j|)^2$ for linear extrapolation weights $\eta_j$ (§4.4); Richardson weights grow
quickly with the number of points and the spread of $\lambda$, which is why two or three scale factors are common.

### 16.3 Quantum subspace expansion (QSE)

QSE [McClean 2017] turns the noisy state into a small *variational space*: choose expansion operators
$\{\sigma_i\}_{i=1}^m$ (e.g. identity, single-qubit Paulis, stabilizers) and look for the best state of the form
$|\tilde\psi\rangle=\sum_ic_i\sigma_i|\psi\rangle$ (or, for mixed states, $\tilde\rho\propto C\rho C^\dagger$ with
$C=\sum_ic_i\sigma_i$). Measuring the matrices

$$\tilde H_{ij}=\mathrm{Tr}(\rho\,\sigma_i^\dagger H\sigma_j),\qquad \tilde S_{ij}=\mathrm{Tr}(\rho\,\sigma_i^\dagger\sigma_j)$$

and solving the generalized eigenvalue problem $\tilde H\vec c=E\,\tilde S\vec c$ gives an energy that is variationally
lower and, when the $\sigma_i$ contain the inverse of the likely errors, corrects them. With the stabilizers of a code
as expansion operators QSE acts as a *decoder* without mid-circuit measurements [McClean 2020]. The **generalized
subspace expansion** [Yoshioka 2022] also allows powers of $\rho$ (as in virtual distillation) and the Hamiltonian
itself, unifying QSE, symmetry verification and virtual distillation as special choices of subspace.

*Costs and pitfalls:* $O(m^2)$ matrix elements, each a measured expectation value; the generalized eigenproblem is
ill-conditioned when $\tilde S$ is nearly singular, so shot noise must be regularized (thresholding small eigenvalues
of $\tilde S$). The estimator is non-linear in the data and slightly biased at finite shots.

### 16.4 Classical shadows

A **classical shadow** [Huang 2020] is a compact classical description of $\rho$ from randomized single-shot
measurements. For random single-qubit Pauli bases, each shot gives a unitary $U=\bigotimes_jU_j$ and outcome
$|b\rangle$; the unbiased snapshot is

$$\hat\rho=\bigotimes_{j=1}^n\bigl(3\,U_j^\dagger|b_j\rangle\langle b_j|U_j-I\bigr),\qquad \mathbb E[\hat\rho]=\rho .$$

Any $k$-local observable is estimated from the same data; $M$ observables of locality $k$ need
$N=O\!\left(\log(M)\,4^k\max_i\lVert O_i\rVert^2/\varepsilon^2\right)$ snapshots with median-of-means estimation
($3^k$ for Pauli strings), independent of $n$. For
mitigation the relevant facts are:

* **Noise-robust shadows.** Measurement noise changes the inverse of the measurement channel. Calibrating that channel
  on a known state ($|0\rangle^{\otimes n}$) and inverting the *noisy* channel removes readout and gate-dependent
  measurement errors that are Pauli-twirled [Chen 2021; Koh 2022] — the shadow analog of readout mitigation.
* **Shadow distillation.** Estimating $\mathrm{Tr}(\rho^2O)/\mathrm{Tr}(\rho^2)$ from shadows gives virtual distillation
  without two copies or entangling derangement circuits, at the price of more samples [Seif 2023].
* **Shadow-based subspace expansion.** The $\tilde H,\tilde S$ matrices of §16.3 can all be read off one shadow data set.

Shadows do not reduce the bias of gate noise by themselves; they are an efficient *measurement* layer on which
mitigation estimators are built.

### 16.5 M3 in depth

For correlated readout errors the full assignment matrix $A\in\mathbb R^{2^n\times2^n}$ is intractable, and even the
tensored inverse (§7.1) produces a dense quasi-distribution. **Matrix-free measurement mitigation** [Nation 2021]
restricts the problem to the set $\mathcal B$ of bit strings actually observed:

1. Build $\tilde A$, the sub-matrix of $A$ on $\mathcal B$ (optionally only entries within Hamming distance $D$), from
   per-qubit (or small-cluster) calibrations; renormalize columns so probability leaking out of $\mathcal B$ is
   accounted for.
2. Solve $\tilde A\,\vec p=\vec f$ for the measured frequencies $\vec f$ — directly for small $|\mathcal B|$, otherwise
   with a preconditioned iterative solver (GMRES/BiCGSTAB with the diagonal as preconditioner). The matrix is never
   formed in the $2^n$ space.
3. Return the quasi-probabilities $\vec p$ and, optionally, the nearest true probability distribution.

Cost scales with $|\mathcal B|\le N_{\rm shots}$ instead of $2^n$. With exact (noise-free) data every bit string is
observed and M3 equals the tensored inverse, which is what the app uses; the difference appears with finite shots on
many qubits. Variance amplification is the same as for the tensored inverse, $\prod_i(1-\epsilon^{(i)}_{01}-\epsilon^{(i)}_{10})^{-2}$
to leading order, so readout mitigation is cheap only while readout errors are small.

### 16.6 How the methods relate, and their limits

| method | knowledge spent | bias removed | overhead | scaling with circuit size |
|---|---|---|---|---|
| readout (tensored, M3, TREX) | calibration of the measurement | readout only | $\prod_i(1-2\bar\epsilon_i)^{-2}$ | per measured qubit |
| symmetry verification / expansion | a symmetry of the ideal state | errors that break it | $1/P_{\rm acc}$ | $P_{\rm acc}$ decays exponentially |
| ZNE (folding, pulse, PEA) | a noise knob $\lambda$ | up to order $\lambda^{m-1}$ | $(\sum_j\lvert\eta_j\rvert)^2$ | modest; bias grows with depth |
| PEC | a full noise model | all modeled noise | $\gamma^2=\prod_g\gamma_g^2$ | exponential in total error |
| learning (CDR, NIL) | near-Clifford training data | the learned map | training circuits, $\lVert c\rVert_1^2$ | depends on training fidelity |
| virtual distillation / QSE | copies / a subspace | incoherent errors | $\mathrm{Tr}(\rho^M)^{-2}$ / $O(m^2)$ terms | exponential in total error |

Two general results frame all of them. First, lower bounds show that for broad classes of noise every unbiased
mitigation protocol needs a number of samples exponential in circuit depth or size (§3.2) [Takagi 2022; Quek 2024]. Second, mitigation and correction
compose: logical qubits have much smaller error rates, so the same estimators at the logical level cost exponentially
less (§17.5) [Suzuki 2022]. A thesis that proposes a method should therefore report bias *and* sampling overhead as
functions of the circuit's total error $\sum_g\epsilon_g$, as the Cost vs precision and Head-to-head pages do.

### 16.7 Chained mitigation: an outer estimator over inner-mitigated circuits

The table above suggests combining methods that remove different parts of the error. A **chained** recipe
(estimator `chain`, `qemx.methods.compose`) runs an *outer* estimator, which decides which circuits $C_j$ are run and
how their values are combined. It replaces each circuit's noisy value with an *inner*-mitigated one:

$$\hat O=\mathcal C\bigl(\{\hat E_{\rm inner}(C_j)\}_j\bigr).$$

| role | options | what it does |
|---|---|---|
| outer | ZNE | folded copies of the target, extrapolated to $\lambda=0$ |
| outer | CDR | a linear map learned on near-Clifford training circuits |
| outer | NIL | neighbor circuits as regression features (§6.3) |
| inner | raw | the circuit's noisy value; `chain(outer, raw)` equals the outer method alone |
| inner | symmetry verification | post-selection on $Z$ parities, training circuits included |
| inner | virtual distillation | $M$ copies (§11) |
| inner | PEC | quasi-probability sampling of every circuit (§5) |

Twirling, dynamical decoupling and readout mitigation act on every circuit, below both levels.

**Known special cases.**

* NIL with fold neighbors and least squares without an intercept is variable-noise CDR [Lowe 2021].
* NIL over an inner virtual distillation combines folding, training and distillation data in the spirit of UNITED
  [Bultrini 2023].

**Accounting.** Every circuit the outer method runs costs what its inner method costs, and the ledger of §3.1 is
applied to the chain as a whole:

* the per-shot variance of each circuit is the inner one. For an outer ZNE this gives
  $V=(\sum_j|\eta_j|\sigma_j^{\rm inner})^2$, so ZNE over PEC pays roughly $(\sum_j|\eta_j|\gamma_j)^2$;
* post-selection counts its rejected shots, and the smallest usable target budget is $\sum_j\lceil5/P_j\rceil$;
* training shots are $T\times k\times$ `train_shots`, each spent on the inner method;
* the training variance $u$ and the calibration coefficient $w$ are measured by Monte-Carlo refits with the inner
  samplers.

The tests check that `chain(zne, raw)` and `chain(nil, raw)` reproduce ZNE and NIL exactly (mean, variances and
cost).

**When a chain pays.** The overheads multiply, while the biases are removed only to the extent that the two methods
target different errors. A chain therefore pays only when the inner method removes an error that breaks the outer
method's assumption. Examples are symmetry-breaking errors that bend $E(\lambda)$, or incoherent errors that a learned
linear map cannot follow. Whether it pays is decided at equal budget, as for any other contender (§18). Chains run in
simulation only in this version.

## 17. Toward fault tolerance: a QEC roadmap

Error mitigation (§3–§16) corrects *averages* after the fact, with a sampling cost that grows exponentially with the
total error of the circuit. Quantum error correction (QEC) removes errors *during* the computation, with a qubit cost
that grows only polylogarithmically with the circuit size once the physical error rate is below a threshold. This
section is a compact map of that second road — the vocabulary a thesis on mitigation needs to place its results — and
of the regime where the two roads meet (§17.5).

### 17.1 The stabilizer formalism

#### Pauli group and stabilizers

The $n$-qubit Pauli group $\mathcal P_n$ consists of tensor products of $I,X,Y,Z$
with phases $\pm1,\pm i$. A **stabilizer code** is defined by an abelian subgroup $\mathcal S=\langle g_1,\dots,g_{n-k}\rangle
\subset\mathcal P_n$ with $-I\notin\mathcal S$; its code space

$$\mathcal C(\mathcal S)=\{|\psi\rangle:\ g|\psi\rangle=|\psi\rangle\ \ \forall g\in\mathcal S\}$$

has dimension $2^k$: $k$ logical qubits in $n$ physical ones [Gottesman 1997].

#### Logical operators and distance

Operators that commute with every stabilizer but are not stabilizers themselves,
$\mathcal N(\mathcal S)\setminus\mathcal S$, act on the code space as logical Paulis $\bar X_j,\bar Z_j$. The **distance** $d$ is
the minimum weight of such an operator; the code is written $[[n,k,d]]$ and corrects any error on up to
$t=\lfloor(d-1)/2\rfloor$ qubits, or detects up to $d-1$.

#### Syndromes

A Pauli error $E$ either commutes or anticommutes with each generator; the **syndrome** is the bit
vector $s_j=1$ iff $Eg_j=-g_jE$, measured with ancilla qubits without disturbing the encoded information. Errors that
differ by a stabilizer have the same effect (**degeneracy**); errors that differ by a logical operator have the same
syndrome but act differently on the encoded information (they differ by a logical operation) — the decoder's
fundamental ambiguity.

#### Knill–Laflamme conditions

A set of errors $\{E_a\}$ is correctable iff $P E_a^\dagger E_b P=c_{ab}P$ for the code
projector $P$ [Knill 1997]; for stabilizer codes this means $E_a^\dagger E_b\notin\mathcal N(\mathcal S)\setminus\mathcal S$.

#### Symplectic picture and CSS codes

Writing a Pauli as a binary vector $(\mathbf x|\mathbf z)\in\mathbb F_2^{2n}$ turns
commutation into a symplectic inner product and a code into a check matrix $H=(H_X|H_Z)$. **CSS codes** [Calderbank
1996; Steane 1996] use separate $X$- and $Z$-type checks from two classical codes with $H_XH_Z^{\mathsf T}=0$; their $X$
and $Z$ errors are decoded independently (Steane, surface, color and most qLDPC codes are CSS).

#### Clifford group and Gottesman–Knill

Unitaries that map Paulis to Paulis form the Clifford group (generated by $H$,
$S$, CX). Clifford circuits with Pauli measurements are simulable in polynomial time with a stabilizer tableau
[Aaronson 2004] — which is why QEC circuits can be simulated at scale, and why universality needs a non-Clifford gate
such as $T$.

**The codes in this app.**

| template | parameters | stabilizers | corrects / detects |
|---|---|---|---|
| 3-qubit bit-flip | $[[3,1,1]]$ (distance 3 against $X$) | $Z_0Z_1,\ Z_1Z_2$ | one $X$ |
| repetition memory | $d$ bits, distance $d$ against $X$ | $Z_iZ_{i+1}$ | $\lfloor(d-1)/2\rfloor$ flips |
| surface code, one plaquette | $[[4,1,2]]$ | $X_0X_1X_2X_3,\ Z_0Z_1,\ Z_2Z_3$ | detects any one error (the app post-selects on both $Z$ checks: the $X$/$Y$ errors that can flip $Z_L$) |
| five-qubit code | $[[5,1,3]]$ | $XZZXI$ and cyclic shifts | any one error (perfect code) |
| Steane code | $[[7,1,3]]$ | $X$ and $Z$ on the Hamming checks | any one error (CSS) |

### 17.2 A taxonomy of codes

| family | examples | parameters and geometry | why it matters |
|---|---|---|---|
| repetition | bit-flip, phase-flip | $[[d,1]]$ against one error type | the classical intuition; biased-noise building block |
| small perfect / CSS codes | Shor $[[9,1,3]]$ [Shor 1995], five-qubit [Laflamme 1996; Bennett 1996], Steane $[[7,1,3]]$ | few qubits, non-local checks | first demonstrations; transversal Cliffords (Steane) |
| surface / toric codes | Kitaev's toric code [Kitaev 2003], rotated surface code [Fowler 2012] | toric $[[2d^2,2,d]]$; rotated patch $[[d^2,1,d]]$; 2D grid, weight-4 checks | circuit-level threshold ≈ 1 % with local checks; the leading architecture |
| heavy-hex and subsystem codes | Bacon–Shor [Bacon 2006], heavy-hex codes [Chamberland 2020] | degree-3 connectivity, gauge operators | matched to IBM's lattice; flags catch hook errors |
| color codes | [Bombin 2006] | 2D/3D, weight-6 checks | transversal Clifford group in 2D |
| quantum LDPC codes | hypergraph product [Tillich 2014], bivariate bicycle "gross" code $[[144,12,12]]$ [Bravyi 2024] | constant check weight, long-range links | many logical qubits per physical qubit |
| bosonic codes | cat codes [Mirrahimi 2014], GKP [Gottesman 2001] | one oscillator per logical qubit | hardware-efficient; biased noise |
| concatenated codes | a code inside a code | $[[n^L,1,d^L]]$ | the original route to the threshold theorem |

### 17.3 Decoding

A **decoder** maps a syndrome history to a correction. For a stabilizer code with independent errors of
probability $p$, the optimal (maximum-likelihood) decoder picks the most likely *logical class* of errors consistent
with the syndrome; practical decoders approximate it:

* **Lookup tables** for small codes — the app decodes the five-qubit and Steane codes this way (16 and 8 syndromes).
* **Minimum-weight perfect matching** (MWPM) [Edmonds 1965; Dennis 2002]: for codes where each error flips at most two
  checks (surface and repetition codes), syndromes are vertices of a graph weighted by $-\log p$ and the most likely
  error is a minimum-weight matching. PyMatching [Higgott 2022] and Sparse Blossom [Higgott 2025] run at the speed
  hardware needs.
* **Union–find** [Delfosse 2021]: almost-linear time with a threshold close to matching.
* **Belief propagation with ordered-statistics decoding** (BP+OSD) [Panteleev 2021; Roffe 2020] for qLDPC codes.
* **Tensor-network / maximum-likelihood** decoders [Bravyi 2014]: near-optimal, used as a benchmark.
* **Neural decoders** trained on device data, e.g. AlphaQubit [Bausch 2024], learn correlated and leakage errors.

Circuit-level decoding works on **detectors** (parities of consecutive syndrome measurements) and a detector error
model generated from the noisy syndrome-extraction circuit, e.g. with Stim [Gidney 2021]. Below threshold the logical
error per cycle falls by a factor $\Lambda$ each time $d$ increases by 2,

$$\varepsilon_L(d)\approx A\,\Lambda^{-(d+1)/2},\qquad \Lambda\approx p_{\rm th}/p ,$$

and $\Lambda>1$ is the experimental signature of operating below threshold. Surface-code memories have shown
$\Lambda>1$ [Google 2023] and $\Lambda\approx2$ up to distance 7, with real-time decoding demonstrated in the
same experiment [Google 2025].

### 17.4 Fault tolerance

#### Threshold theorem

If every component fails independently with probability below a constant $p_{\rm th}$, an ideal
circuit of size $G$ can be simulated with error $\varepsilon$ using $O(G\,\mathrm{polylog}(G/\varepsilon))$ components
[Aharonov 2008; Knill 1998; Kitaev 2003]. It requires **fault-tolerant gadgets**: syndrome extraction that does not
spread one fault into an uncorrectable error (repeated measurement, flag qubits [Chao 2018], careful CX ordering to
avoid hook errors).

#### Logical gates

Transversal gates apply the same gate to every physical qubit and cannot spread errors, but no code
has a universal transversal gate set (Eastin–Knill [Eastin 2009]). The standard solution: Clifford gates by
transversal operations or **lattice surgery** [Horsman 2012], and the $T$ gate by **magic-state distillation**
[Bravyi 2005] — a protocol that turns many noisy $|T\rangle$ states into fewer, cleaner ones and dominates the space-time
cost of fault-tolerant algorithms.

#### Resource estimates

Factoring RSA-2048 with surface codes was estimated at about 20 million noisy qubits
[Gidney 2021b], later reduced to under one million with better arithmetic and magic-state cultivation [Gidney 2025].
Such numbers — and the depth of the "repeated powers" Shor template in this app — are why error correction, not
mitigation, is the route to large algorithms.

### 17.5 Logical error mitigation

Early fault-tolerant machines will have a modest code distance: logical error rates $\varepsilon_L$ of perhaps
$10^{-3}$–$10^{-6}$ per logical operation, noisy magic states, and circuits with more logical operations than
$1/\varepsilon_L$. Mitigation applied **to logical qubits** fills that gap, and it is exponentially cheaper there because the
overhead depends on the *logical* error budget: with $G$ logical gates, $\gamma^2\approx e^{4G\varepsilon_L}$ instead of
$e^{4Gp}$ [Suzuki 2022].

* **Logical PEC.** The noise of logical operations after decoding is close to Pauli and can be characterized from
  syndrome statistics; PEC with that logical noise model cancels residual logical errors, including those of
  non-Clifford gates implemented with imperfect magic states [Piveteau 2021; Lostaglio 2021].
* **Logical ZNE.** The code distance is a natural noise knob: $\varepsilon_L(d)$ from runs at $d=3,5,7$ extrapolated to
  zero logical error [Wahl 2023].
* **Detection and post-selection.** Small codes used as *detection* codes (distance 2, like the surface-code plaquette
  here, post-selected on both of its $Z$ checks) discard shots with a non-trivial syndrome — symmetry verification with stabilizers as the symmetry (§16.1).
  The acceptance rate is the price, exactly as in §10.
* **Hybrid error budgets.** For a fixed number of physical qubits one can trade code distance (qubits) against
  mitigation overhead (shots); lower distance plus mitigation can beat higher distance alone for short circuits
  [Suzuki 2022]. First experiments demonstrate mitigation on encoded qubits [Logical QEM 2025].

**In the app:** the *surface code d = 2* template with *Symmetry verification* is detection + post-selection; the
*five-qubit code* with *ZNE* or *PEC* is correction + mitigation. The *Logical error & syndromes* view reports the
effective logical error $p_L=\tfrac12(1-\langle Z_L\rangle/\langle Z_L\rangle_{\rm ideal})$ of each.

### 17.6 The roadmap at a glance

| regime | what protects the computation | cost grows with | the app's examples |
|---|---|---|---|
| NISQ | error mitigation (ZNE, PEC, learning, readout, twirling, DD) | shots: $\sim e^{c\,Gp}$ | every template, *Mitigate & compare* |
| early fault tolerance | small codes for detection + mitigation of residual logical errors | shots $\sim e^{c\,G\varepsilon_L}$ and qubits $\sim d^2$ | surface plaquette + post-selection, five-qubit code + ZNE/PEC |
| fault tolerance | error correction below threshold, lattice surgery, magic states | qubits $\sim d^2\,\mathrm{polylog}$, time $\sim d$ per step | repetition memory ($d$), Steane and five-qubit decoders |

Reading the table from top to bottom is reading the next decade of the field: the estimators this app compares
stay useful at every stage — first on physical qubits, then on logical ones.

## 18. Equal-budget phase diagrams

§3.1 defines the ledger for one circuit and one noise level. The research backend (`qemx.phase`) evaluates it on a
grid: circuit families × widths × noise levels × noise engines, with every method run on every point (a **cell**).
It then asks which method wins where, and why. This section collects the definitions behind those maps.

### 18.1 The light-cone fault rate $\lambda_{LC}$

A **fault location** $f$ is one elementary noise instruction of the executed circuit. Examples are the error after a
gate, the relaxation of one idle segment, a crosstalk kick, or a noise operation placed by hand. Its **fault
probability** is the probability that its Pauli twirl applies a non-identity Pauli:

$$p_f=1-p_I,\qquad p_I=\frac1{4^k}\sum_P\lambda_P ,$$

with $\lambda_P$ the diagonal of the channel's Pauli transfer matrix (§2.1). The formula is exact for any channel. A
coherent rotation by $\theta$, for example, has $p_f=\sin^2(\theta/2)$. Summing over locations gives the circuit fault
rate $\lambda=\sum_fp_f$ [Cai 2023]. Under independent faults, $e^{-\lambda}$ is the probability that no fault occurs.

Most faults cannot reach a local observable, so the phase diagrams use the sum restricted to the observable's
**backward light cone**. Walk the instruction list backward from the measured qubits that $O$ depends on; every
multi-qubit instruction that touches the current set adds its qubits to the set. A location counts only through the
Pauli components that act on light-cone qubits:

$$\lambda_{LC}=\sum_{f\in{\rm LC}}p_f^{\rm LC},\qquad p_f^{\rm LC}=\sum_{P:\,P|_{\rm LC}\ne I}p_P .$$

**Conventions.**

* Readout is reported separately as $\lambda_{\rm ro}=\sum_{q\in{\rm support}}(P(1|0)+P(0|1))/2$, because readout
  mitigation removes it with its own calibration.
* A qubit's idling counts from its first gate. Waiting in $|0\rangle$ before that gate is not a fault.

**Three properties make $\lambda_{LC}$ the first coordinate of every map:**

* it can be computed from the calibration *before* any circuit runs;
* it is proportional to the noise scale $s$ within each circuit (to 1 % over $s=0.25$–6 on fake_fez);
* PEC's cost follows it: the light-cone ratio $\lambda/\lambda_{LC}$ (1.37 for a random brickwork circuit, 1.00 for
  QFT and GHZ) is the extra cost a gate-by-gate PEC pays for faults that cannot reach the observable [Tran 2023].

**Two properties keep it from being the only coordinate:**

* **Observable sensitivity.** The unmitigated signal decays as $\langle O\rangle_{\rm noisy}\approx\langle
  O\rangle_{\rm ideal}e^{-\kappa\lambda_{LC}}$. The constant $\kappa$ differs between circuits, from 0.34 (Ising-6) to
  1.09 (GHZ-6), because $\lambda_{LC}$ also counts faults that commute with the propagated observable. For Pauli noise,
  $\kappa=2\lambda_{\rm anti}/\lambda_{LC}$, where $\lambda_{\rm anti}$ counts only the anticommuting faults.
* **Coherence.** A fault count cannot tell a stochastic error from a coherent one of the same infidelity. Yet the two
  behave differently when a gate is repeated (§2.3, §4.6), which §18.3 shows directly.

### 18.2 Scaling the noise exactly

A phase diagram needs a noise axis that moves $\lambda_{LC}$ by a known factor. Two engines provide it:

* **E2, the device itself.** Every gate error and every relaxation rate of the fake device's calibration is
  multiplied by $s$, then executed exactly.
* **E1, a homogenized model.** One `NoiseSpec` is calibrated to the device's median rates, and each channel's
  **fault probability** is scaled by exactly $s$. That is not the same as scaling its parameter. A rotation, for
  example, has $p_f=\sin^2(\theta/2)$, so its angle scales as $\theta'=2\arcsin(\sqrt s\sin(\theta/2))$, which is about
  $\sqrt s\,\theta$ for small angles. Scaling the angle linearly would multiply its fault probability by $s^2$.

E1 also carries a **coherent fraction** $\chi$: a fraction $\chi$ of every gate's non-relaxation error is turned into
a systematic over-rotation, at fixed infidelity. Because $\lambda_{LC}$ is unchanged, any change in the winners is due
to the coherence of the noise alone.

### 18.3 Regimes, crossovers and unreachable precision

With $F_i=b_i^2+u_i$ the bias-and-training floor of method $i$ and $V_i$ its effective per-shot variance, the RMSE
at total budget $B$ is approximately $\sqrt{F_i+V_i/B}$. Along the budget axis every map then passes through up to
three regimes:

1. **Variance-limited** (small $B$). The estimator with the smallest $V$ wins among those that can run at all.
   Unmitigated estimation is a genuine contender here.
2. **Crossover.** A less-biased but noisier method $j$ overtakes $i$ at
   $$B^*_{i\to j}=\frac{V_j-V_i}{F_i-F_j}.$$
3. **Floor-limited** (large $B$). The smallest $F$ wins.

Two consequences shape how the maps must be read:

* **Cancellation wins.** A method's *signed* bias $b(\lambda)$ can cross zero. ZNE, for instance, over-corrects at
  small $\lambda$ and under-corrects at large $\lambda$. Near the crossing $|b|$ is accidentally tiny, and the method
  wins the top of that column without being less biased in general. The maps therefore show $b/|b_{\rm RAW}|$
  against $\lambda_{LC}$ beside the winners.
* **Winning is not reaching.** The cheapest budget for a target precision $\varepsilon$ is
  $B_i(\varepsilon)=V_i/(\varepsilon^2-F_i)$, and it is infinite when $F_i\ge\varepsilon^2$. The method that wins at
  large $B$ is often not the cheapest route to $\varepsilon$. A region where every method's $B_i(\varepsilon)$ exceeds
  the available budget is a measured result in its own right: there, *nothing works*.

Winners within 5 % of the runner-up are marked as ties: a winner inside its own error bar is not a winner.

### 18.4 Cost exponents: the two-constant law

For a fixed training design the variance coefficient of a method follows, to good approximation,

$$\ln V_i=\ln A_i+\beta_i\lambda_{LC},\qquad V_i=\bigl(\sqrt{v_i}+\sqrt{c\,w_i}\bigr)^2,$$

so that $\mathrm{RMSE}_i(\lambda,B)\approx\sqrt{b_i(\lambda)^2+u_i+A_ie^{\beta_i\lambda}/B}$. The signed bias is fitted
as $b(\lambda)=\beta_1\lambda+\beta_2\lambda^2$ through the origin. These fits (`qemx.phase.fit`, with cell-bootstrap
intervals) turn a map into a few numbers per method.

The lower bounds of §3.2 say $\beta$ cannot be small for an unbiased method. The measured values say how large it is
in practice, and that it depends on the circuit. For PEC, $\beta$ is 2.7–5.1 over the n = 6 circuits on fake_fez.
For Richardson ZNE it is close to 0 over this range, because its cost is a large constant ($\Sigma|\eta|$) and its
weakness is bias. Exponential ZNE does not follow a two-constant law at all, because its variance depends on how
close the folded values come to zero.

### 18.5 Validating the ledger by Monte Carlo

The ledger is exact in $B$ for linear estimators and first-order for nonlinear ones. Exponential ZNE, refitted
regressions and readout inversion are nonlinear, so the ledger is checked by simulation. `qemx.phase.validate`
simulates $R$ complete experiments that spend exactly $B$ shots in the ledger's integer split. Calibration is
resampled, training is refitted and ZNE re-extrapolated in every replicate. Each experiment yields z-scores for the
bias and the RMSE against the ledger.

The acceptance criteria are fixed before any production run:

* a Kolmogorov–Smirnov test of the z-scores against $N(0,1)$;
* $|z|>4$ in fewer than 1 % of the checks.

**What the checks find.** The ledger fails where a nonlinear estimator is used outside its domain:

* the logarithm in exponential ZNE is biased by about $-\tfrac12\mathrm{Var}/E^2$ at small $N$;
* when the folded values $E_j$ change sign the fit does not exist, which happens with probability about
  $\Phi(-|E_j|\sqrt{N_j}/\sigma_j)$.

State that domain rather than move the threshold after the fact.

### 18.6 What the maps say (fake_fez, 1,705 cells)

* **No method wins everywhere.** The budget matters as much as the noise.
* **Unmitigated estimation wins small budgets.** Random brickwork (n = 6) below $7.9\times10^5$ shots at
  $\lambda_{LC}=0.016$; Ising-6 below $2\times10^4$ at 0.045.
* **ZNE's large-budget wins sit next to the zeros of its bias.**
* **Learning methods pay an entry fee** (their training shots) **and hit a floor** set by those shots.
* **At fixed $\lambda_{LC}$, a coherent fraction $\chi=0.5$ rewrites the map.** On AQFT-6 the bias of exponential ZNE
  goes from 0.016 to 0.609, and CDR, which learns the actual map, degrades least.
* **Large regions are unreachable.** For QFT-6, no method reaches RMSE 0.01 at any budget for six of eight noise
  levels.

**The lesson for the rest of this part.** A scalar noise level does not determine what a device does to a circuit.
How the noise is *structured* matters, and that structure is exactly what a component-level model gets wrong (§19).

## 19. A device and its component model

### 19.1 What a fake device leaves out

A fake device (§2.6) is built from component-level numbers measured one gate at a time, on otherwise idle qubits:

* average gate infidelities from randomized benchmarking;
* readout assignment errors;
* $T_1$, $T_2$ and gate durations.

A circuit, however, runs gates *in context*, and the same gate can then err differently [Rudinger 2019]:

* its neighbors are spectators coupled to it by always-on ZZ interactions [Krantz 2019; Zhao 2022];
* neighboring gates run simultaneously and cross-talk [Sarovar 2020; Murali 2020];
* a cross-resonance gate also drives its control qubit's other neighbors [Sundaresan 2020];
* idle qubits dephase under low-frequency noise that an exponential $T_2$ does not describe;
* a single qubit may couple to a two-level-system defect whose frequency wanders [Klimov 2018; Müller 2019];
* the device drifts between calibration and run.

Layer fidelities measured on whole simultaneous layers are accordingly lower than the product of isolated gate
fidelities [McKay 2023], and circuit performance departs from component metrics in structured ways [Proctor 2022].

The divergence study (`qemx.divergence`, `qemx.context_noise`, `scripts/dirA_divergence.py`) measures that departure
circuit by circuit. It then asks which mechanism is responsible.

### 19.2 Fidelity of an output distribution

For an ideal output distribution $p$ and a measured or predicted one $q$, the **classical (Hellinger) fidelity** is

$$F_c(p,q)=\Bigl(\sum_x\sqrt{p_x\,q_x}\Bigr)^2 ,$$

the square of the Bhattacharyya coefficient. It is 1 for identical distributions and 0 for disjoint ones. A wide
circuit's output can be close to uniform even when every shot is wrong, so the **normalized fidelity**
[Lubinski 2023]

$$F_N=\max\!\Bigl(\frac{F_c(p,q)-F_c(p,u)}{1-F_c(p,u)},\,0\Bigr),\qquad u=\text{uniform},$$

measures only the part above what random guessing achieves. It is undefined when $p$ itself is uniform. The study's
regressions use $\ln F_N$: under independent faults it is additive in the error events, which $F_c$ is not.

### 19.3 Three predictors and a comparison at the same shots

Every as-executed circuit is predicted from the device's calibration, taken at execution time when the service keeps
the history. Three models are used:

| predictor | what it contains |
|---|---|
| **depolarizing** (primary) | every gate followed by the depolarizing channel whose average infidelity is the reported error, $p=r\,d/(d-1)$; asymmetric readout per qubit; no relaxation, no idle error. This is the "component model" |
| **median** | one median rate per gate class (the model of [Chen 2024]), no readout |
| **device** | the full fake-device model: relaxation during gates and idle windows on the ALAP schedule, readout |

**A fair comparison means the same estimator on both sides.** Every prediction is passed through the *same*
estimator as the observation, at the *same* shots: raw, readout-mitigated, ZNE on the distribution, projected onto
the simplex. This is done for $R$ Monte-Carlo replicates. The difference

$$\Delta=\overline{F_c^{\rm pred}}-F_c^{\rm obs}$$

then carries the same finite-shot bias on both sides, and its significance is

$$z=\frac{\Delta}{\sqrt{(\sigma_{\rm rep}^2+\mathrm{SE}_{\rm boot}^2)/2}},$$

pooling the predictor's replicate SD with the observation's bootstrap standard error. When the model is right these
are two estimates of the same shot noise. Pooling also stops $z$ from exploding when the model predicts an almost
deterministic estimator, such as a plurality vote on a clean peak.

**Null calibration.** If the emulated device *is* the depolarizing model, $z$ must be standard normal, and it is:

* 236 scores, mean −0.06, SD 0.90, KS $p=0.13$;
* none with $|z|>3$.

A significant $\Delta$ on hardware is therefore not an artifact of the estimators or the budget.

### 19.4 The effective error rate and the fidelity gap

Over the circuits with $F_c\ge0.05$, $\ln F_c$ is fitted against the post-compilation two-qubit gate count $n_2$:

$$\ln F_c\approx a+n_2\ln(1-\varepsilon_{\rm eff}).$$

The fit is made for the observation and for each predictor, giving $\varepsilon_{\rm eff}^{\rm obs}$ and
$\varepsilon_{\rm eff}^{\rm pred}$. Their ratio is the simplest summary of a divergence: how much faster the device
loses fidelity per two-qubit gate than its model says.

* The intercept $a$ absorbs state preparation and readout.
* The fit is made for unmitigated and readout-mitigated estimates only: a ZNE-extrapolated distribution does not
  decay exponentially in $n_2$.
* The `fidelity_gap` figure draws both decays on a log axis and joins each circuit's two points. The length of a
  joining segment is that circuit's unmodeled error.

### 19.5 The difference volumetric plot and the excess error per gate

The **difference volumetric plot** (after [Blume-Kohout 2020; Proctor 2022; Chen 2024]) places every circuit at
(post-compilation two-qubit gates, width), in logarithmic bins. Each tile is colored by the mean $\Delta$ of its
circuits on a symmetric diverging scale: red means the device is worse, blue better, gray as modeled. A dot marks a
tile in which most circuits lie beyond $|z|=3$. The figure shows *where* on the size plane the model fails.

$\Delta$ grows with circuit size and vanishes again when both distributions decay to the uniform floor, so large and
small circuits are not comparable on $\Delta$. The **excess error per two-qubit gate**,

$$e=\frac{\ln F_N^{\rm pred}-\ln F_N^{\rm obs}}{n_2},$$

is comparable across the whole plane, and it has its own volumetric plot.

### 19.6 Emulating unmodeled noise

Before any hardware time, every diagnostic is tested on an **emulated device**: a predictor's model of the snapshot
plus a known error the model does not contain (`emulate --inject`). The counts are multinomial samples with the
planned shots.

| injection | what it adds | expected reading |
|---|---|---|
| `scale=s` | every gate error and relaxation rate × $s$: a calibration that under-reports | gate-attached; a rescaled calibration |
| `coherent=χ` | a coherent fraction $\chi$ of every gate's error, at the same infidelity | non-additive |
| `crosstalk=ζ` | $e^{-i\zeta Z\otimes Z/2}$ on every spectator pair after every two-qubit gate | spectator context; repetition-coherent |
| `t2=τ` | every qubit's $T_2$ lowered to $\tau$ µs: dephasing in idle windows | time-attached; idle dephasing |
| `hotspot=q` | every gate on physical qubit $q$ at 5× its reported error | position-dependent |

An emulation is never a substitute for hardware data. It shows that the pipeline names a cause **when that cause is
present**, and what each figure looks like when the cause is known.

## 20. Context-dependent noise on heavy hex: designed experiments

### 20.1 Context features of an as-executed circuit

Every circuit of a plan is scheduled as late as possible (ALAP), with the backend's durations, as IBM backends do.
Its context is then counted on the device's coupling graph (`qemx.context_noise.circuit_features`):

| feature | definition | mechanism |
|---|---|---|
| `n2q`, `n1q` | two-qubit and physical one-qubit gates ($R_Z$ is virtual) | what the model already prices |
| `gate_error`, `readout_error` | the sums of the reported errors | what the model already prices |
| `active_spectators` | Σ over two-qubit gates of the circuit's *own* qubits coupled to the gate's qubits | ZZ to spectators in superposition |
| `passive_spectators` | the same for device qubits *outside* the circuit (in $\|0\rangle$) | ZZ to idle neighbors |
| `control_spectators` | cross-resonance gates only: neighbors of the driven control qubit | CR drive crosstalk (ECR devices) |
| `parallel_adjacent` | pairs of two-qubit gates at the same time on adjacent couplers | simultaneous-gate crosstalk |
| `idle_us` | idle time between a qubit's first gate and its measurement | dephasing exposure |

The model's own predicted log-infidelity, $-\ln F_{\rm pred}$ (feature `priced`), stands for everything the
calibration already accounts for.

### 20.2 Layout variants on disjoint chains

Each benchmark is laid out on a **chain**, a path of the coupling graph found by beam search over calibrated errors,
so that routing stays confined and comparable across widths. With `variants: {mode: layouts}` it runs on several
chains:

1. disjoint chains, best calibrated first;
2. the same chains **reversed**, which gives another qubit-to-position assignment;
3. overlapping chains.

No assignment is used twice in either orientation. The benchmark's budget is shared by its variants.

| mode | what changes between variants |
|---|---|
| `layouts` | the qubit assignment (where on the chip) |
| `twirl` | the gate frame: an independent Pauli twirl of every two-qubit gate |
| `layouts+twirl` | both (the variant design of [Chen 2024]) |
| `seeds` | the transpiler seed |

**Heterogeneity.** Do the variants of one benchmark agree? With $\Delta_i$ and standard errors $\mathrm{SE}_i$ of its
$k$ variants, Cochran's statistic [Cochran 1954]

$$Q=\sum_iw_i(\Delta_i-\bar\Delta)^2,\qquad w_i=\mathrm{SE}_i^{-2},\qquad \bar\Delta=\frac{\sum_iw_i\Delta_i}{\sum_iw_i},$$

is $\chi^2_{k-1}$-distributed when they share one divergence. $I^2=\max(0,(Q-(k-1))/Q)$ is the share of the spread
that shot noise does not explain [Higgins 2002]. The study flags **position dependence** when more benchmarks have
$p<0.01$ than chance allows: at least 3, and over 10 % of them (the null expectation is 1 %).

### 20.3 The plurality vote over variants

The variants' histograms can be merged by **simple aggregation** (an equal-weight average) or by Chen et al.'s
**approximate plurality vote** [Chen 2024; Maksymov 2023]. Let $f_{vb}$ be variant $v$'s measured frequency of
bitstring $b$, and $o_b$ the number of variants that saw $b$. Following the authors' reference implementation:

1. **Threshold.** $t=\min(t_0,\max_bo_b)$, with $t_0=7$ for 25 variants (scaled as $\mathrm{round}(0.28V)$, at least
   3). If $t\le2$ there is no consensus, and the vote returns the average.
2. **Survivors.** The survivors are $\mathcal B=\{b:o_b\ge t\}$. A single survivor gets probability 1.
3. **Vote probability.** Each survivor's vote is the probability that it is drawn by at least $t$ variants when each
   variant draws once from its *raw* frequencies:
   $$p_b\propto\Pr\Bigl[\sum_v X_{vb}\ge t\Bigr],\qquad X_{vb}\sim\mathrm{Bernoulli}(f_{vb}),$$
   a Poisson-binomial tail. QEM Explorer computes it by the exact $O(V^2)$ recursion; the paper writes it as a sum
   over subsets (its Eq. 6).
4. **Normalization.** The $p_b$ are normalized.

`renormalize=True` instead renormalizes each variant over the survivors first, which is a literal reading of the
paper's text.

The vote is a **decoder for peaked distributions**. It suppresses errors that differ between variants, and it sharpens
a peaked output (BV, a QFT on a basis state). But it *lowers* the fidelity of broad outputs, so the study reports
both aggregations.

### 20.4 Spectator-context probes

Application circuits mix every mechanism. The **context probe** separates them by design. A random two-qubit core
runs on $(c_0,c_1)$ of the chain $a$–$s_0$–$c_0$–$c_1$–$s_1$–$b$, with its two spectators $s_0,s_1$ in one of three
modes:

| mode | spectators | exposes |
|---|---|---|
| `zero` | idle in $\|0\rangle$ | the reference |
| `plus` | idle in $\|+\rangle$ | dephasing of waiting qubits, ZZ phases on superposed neighbors |
| `driven` | running their own CZ gates with $a$, $b$ | gate error next to the core, simultaneous-drive crosstalk |

The ideal output is the same in all three modes. Alignment barriers keep the schedule identical through compilation
and every fold. For each mode the **excess contrast** compares the drop against the idle reference on the device with
the drop the model predicts:

$$C_{\rm mode}=\bigl[F_c(\text{zero})-F_c(\text{mode})\bigr]_{\rm obs}-\bigl[F_c(\text{zero})-F_c(\text{mode})\bigr]_{\rm pred}.$$

The per-depth z-scores are combined over core depths and layout variants by Stouffer's method,
$Z=\sum_iz_i/\sqrt n$ [Stouffer 1949]. The **reading** uses $Z>3$ per mode:

* superposed only → idle dephasing;
* driven only → gate error;
* both → spectator crosstalk;
* neither → no spectator effect.

### 20.5 Per-qubit attribution and the heavy-hex map

Where on the chip does the model under-predict? Each circuit's excess log-infidelity is regressed on the
two-qubit-gate load of every physical qubit it used. The fit is weighted (by $\mathrm{SE}^{-2}$), non-negative and
ridge-stabilized, with bootstrap intervals over circuits. The coefficient of a qubit is its unmodeled error per
two-qubit-gate participation, and it is drawn at the qubit's heavy-hex coordinates on a sequential scale, ringed where
the interval excludes 0. The fit needs circuits on different qubits, which the layout variants provide.

## 21. Explaining the gap: the diagnostics

### 21.1 The response

For every circuit $c$, the response is the excess log-infidelity of the readout-mitigated estimate (so that readout
drops out):

$$y_c=\ln F_{N,c}^{\rm pred}-\ln F_{N,c}^{\rm obs}.$$

Its standard error $\mathrm{SE}_c$ comes from the delta method, using the pooled shot noise of §19.3. Every fit below
is weighted by $\mathrm{SE}_c^{-2}$, so that $\chi^2/\mathrm{dof}\approx1$ marks a sufficient explanation.

### 21.2 The χ² ladder: calibration or context?

Four nested non-negative fits of $y_c$:

| model | $y_c=$ | hypothesis |
|---|---|---|
| none | $0$ | the component model is right |
| rescaled calibration | $c\,(-\ln F_{\rm pred,c})$ | every calibrated error $(1+c)\times$ too low, propagated through the circuit by the model itself |
| context | $\sum_k\beta_kx_{ck}$ over the context features of §20.1 | context alone |
| calibration + context | both | context beyond a rescaling |

The regressor $-\ln F_{\rm pred}$ is the key choice. A gate count, or a sum of reported errors, misses how an error
propagates to the output distribution: on a pure ×1.6 rescaling such regressors fitted with $\chi^2/\mathrm{dof}$
2.3–4.1 instead of 1.25.

**Verdicts.**

* **As modeled** when the null model suffices.
* **Rescaled calibration** when context adds nothing significant ($\Delta\chi^2$ test, $p>0.01$) *or* nothing
  material, meaning the calibration alone removes ≥ 95 % of the $\chi^2$ that all mechanisms together remove. With
  many precise circuits a first-order rescaling is rejected somewhere, because $\ln F$ is linear in the error scale
  only to first order.
* **Context** otherwise.
* **"+ non-linear"** is appended when even the full model leaves $\chi^2/\mathrm{dof}>2$. This flags coherent or
  non-additive mechanisms (hypothesis H3).

### 21.3 The divergence budget: which context?

The same weighted non-negative least squares over `priced` and the context features splits each circuit's excess
into mechanisms [Lawson 1974]. Each coefficient carries a bootstrap interval over circuits [Efron 1993]. A
**variance inflation factor** $\mathrm{VIF}_k=1/(1-R_k^2)$ above 10 marks a mechanism that these circuits cannot
separate from the others. The family budget reports each mechanism's mean contribution per circuit, plus what the
mechanisms leave unexplained, which may be negative.

**The limit is collinearity.** In application circuits spectator exposure, parallel gates and idle time grow
together (VIF up to 23), and a coherent error is not additive. The regression then splits a crosstalk error over all
three mechanisms. The designed probes (§20.4) and the fingerprint (§21.4) separate what the regression cannot.

### 21.4 The folding fingerprint: gate, time or coherent build-up?

Global and local folding (§4.2, §4.6) amplify different things:

* **global folding** repeats every gate *and* stretches every idle window;
* **local folding** repeats only the two-qubit gates, so the same gate is applied back to back.

For every circuit and both foldings, the decay rate $\kappa=-\mathrm{d}\ln F/\mathrm{d}\lambda$ is fitted over
$\lambda=1,3,5$, for the observation and for the model. For the model, $\kappa$ is the mean over replicates at the
same shots: ZNE gives its most folded circuits the fewest shots, and an exact $\kappa_{\rm pred}$ would make the
observed decay look steeper by a sampling bias. The **amplification anomaly**

$$A=\frac{\kappa_{\rm obs}}{\kappa_{\rm pred}}$$

is pooled per circuit family by a weighted fit of $\kappa_{\rm obs}=A\,\kappa_{\rm pred}$, with standard error
$\mathrm{SE}_A$. With $k=3$ standard errors, the pair $(A_{\rm global},A_{\rm local})$ reads:

| reading | condition | mechanism |
|---|---|---|
| as modeled | neither differs from 1 | — |
| gate-attached | both above 1, not distinguishable | unmodeled error that rides on the gates (a rescaled calibration) |
| time-attached | $A_{\rm global}-A_{\rm local}>k\,\mathrm{SE}$ | error attached to time: idle dephasing, drift (only global folding stretches idle windows) |
| repetition-coherent | $A_{\rm local}-A_{\rm global}>k\,\mathrm{SE}$ | an error that builds up in amplitude when the same gate is repeated (§2.3), such as a ZZ phase after every CZ |
| over-modeled | both below 1 | the model is too pessimistic |

The fingerprint needs no extra circuits: it reuses the ZNE runs that mitigation already pays for. The *values*
depend on the suite (fits flatten where $F$ reaches its floor); the *reading* transfers.

### 21.5 The diagnostic matrix and the recovery protocol

`dirA_divergence.py recovery` reads the tables of several analyses. It draws one row per data set and one column per
diagnostic:

* the ladder;
* the budget;
* the fingerprint;
* the probes;
* the layout variants;
* $\varepsilon_{\rm eff}^{\rm obs}/\varepsilon_{\rm eff}^{\rm pred}$.

A filled red dot marks an anomaly, an open ring "as modeled". The matrix is the **validation protocol** of the study.
Before a reading on hardware is trusted, the same pipeline, on the same circuits and budget, must name each planted
cause.

On the v2 suite (52 benchmarks × 3 layouts, fake_fez) every planted cause is recovered:

| planted | ladder | budget | fingerprint $(A_g,A_l)$ | probes | $\varepsilon$ ratio |
|---|---|---|---|---|---|
| none | as modeled | — | 0.96, 0.99 | none | 1.03 |
| gate errors ×1.6 | rescaled calibration ×1.57 | rescaled calibration 87 % | 1.31, 1.27: gate-attached | driven only | 1.60 |
| $T_2=30$ µs | context + non-linear | idle time 100 % | 1.12, 0.97: time-attached | superposed only | 1.28 |
| ZZ 0.05 rad | context + non-linear | parallel 45 %, active spectators 28 %, idle 27 % | 2.11, 2.75: repetition-coherent | both | 1.66 |

A defect on a single qubit is seen only by the position column. Every other diagnostic pools over the chip, which is
why the layout variants are part of the design.

### 21.6 From emulation to hardware

**On the day of the run.** The hardware study uses the same plan:

1. lay out the chains on that day's calibration (`plan --backend ibm_fez`);
2. check every circuit as ISA and every fold after compilation;
3. submit everything in one Batch, with dynamical decoupling and service-side twirling switched **off**, because the
   analysis measures the idle windows itself;
4. collect the counts together with the calibration at execution time;
5. add the device's row to the diagnostic matrix.

**Budget.** At about 0.57 ms per shot on ibm_fez, a 10-minute window holds about $10^6$ shots. The
`direction_A_divergence_v2_qpu10` manifest fits the full set of diagnostics into 720,000 shots:

* 40 benchmarks × 3 layouts;
* RAW, RO and global and local ZNE.

**Pre-registration and drift.** The expected readings and their nulls are written down before the run. A second day
turns position across days into a drift test.

**The token** is read from an environment variable, a saved account or a hidden prompt. It is never a command-line
argument and is never written to a file.

## References

* [Aaronson 2004] S. Aaronson, D. Gottesman, *Improved simulation of stabilizer circuits*, Phys. Rev. A **70**, 052328 (2004). doi:10.1103/PhysRevA.70.052328 · arXiv:quant-ph/0406196
* [Aharonov 2008] D. Aharonov, M. Ben-Or, *Fault-tolerant quantum computation with constant error rate*, SIAM J. Comput. **38**, 1207 (2008). doi:10.1137/S0097539799359385 · arXiv:quant-ph/9906129
* [Bacon 2006] D. Bacon, *Operator quantum error-correcting subsystems for self-correcting quantum memories*, Phys. Rev. A **73**, 012340 (2006). doi:10.1103/PhysRevA.73.012340 · arXiv:quant-ph/0506023
* [Barenco 1996] A. Barenco, A. Ekert, K.-A. Suominen, P. Törmä, *Approximate quantum Fourier transform and decoherence*, Phys. Rev. A **54**, 139 (1996). doi:10.1103/PhysRevA.54.139 · arXiv:quant-ph/9601018
* [Bausch 2024] J. Bausch et al., *Learning high-accuracy error decoding for quantum processors*, Nature **635**, 834 (2024). doi:10.1038/s41586-024-08148-8
* [Bennett 1996] C. H. Bennett, D. P. DiVincenzo, J. A. Smolin, W. K. Wootters, *Mixed-state entanglement and quantum error correction*, Phys. Rev. A **54**, 3824 (1996). doi:10.1103/PhysRevA.54.3824 · arXiv:quant-ph/9604024
* [Bennett 1997b] C. H. Bennett, E. Bernstein, G. Brassard, U. Vazirani, *Strengths and weaknesses of quantum computing*, SIAM J. Comput. **26**, 1510 (1997). doi:10.1137/S0097539796300933 · arXiv:quant-ph/9701001
* [Bernstein 1997] E. Bernstein, U. Vazirani, *Quantum complexity theory*, SIAM J. Comput. **26**, 1411 (1997). doi:10.1137/S0097539796300921
* [Blume-Kohout 2020] R. Blume-Kohout, K. C. Young, *A volumetric framework for quantum computer benchmarks*, Quantum **4**, 362 (2020). doi:10.22331/q-2020-11-15-362 · arXiv:1904.05546
* [Bombin 2006] H. Bombin, M. A. Martin-Delgado, *Topological quantum distillation*, Phys. Rev. Lett. **97**, 180501 (2006). doi:10.1103/PhysRevLett.97.180501 · arXiv:quant-ph/0605138
* [Bonet-Monroig 2018] X. Bonet-Monroig, R. Sagastizabal, M. Singh, T. E. O'Brien, *Low-cost error mitigation by symmetry verification*, Phys. Rev. A **98**, 062339 (2018). doi:10.1103/PhysRevA.98.062339 · arXiv:1807.10050
* [Boyer 1998] M. Boyer, G. Brassard, P. Høyer, A. Tapp, *Tight bounds on quantum searching*, Fortschr. Phys. **46**, 493 (1998). doi:10.1002/(SICI)1521-3978(199806)46:4/5%3C493::AID-PROP493%3E3.0.CO;2-P · arXiv:quant-ph/9605034
* [Bravyi 2005] S. Bravyi, A. Kitaev, *Universal quantum computation with ideal Clifford gates and noisy ancillas*, Phys. Rev. A **71**, 022316 (2005). doi:10.1103/PhysRevA.71.022316 · arXiv:quant-ph/0403025
* [Bravyi 2014] S. Bravyi, M. Suchara, A. Vargo, *Efficient algorithms for maximum likelihood decoding in the surface code*, Phys. Rev. A **90**, 032326 (2014). doi:10.1103/PhysRevA.90.032326 · arXiv:1405.4883
* [Bravyi 2021] S. Bravyi, S. Sheldon, A. Kandala, D. C. Mckay, J. M. Gambetta, *Mitigating measurement errors in multiqubit experiments*, Phys. Rev. A **103**, 042605 (2021). doi:10.1103/PhysRevA.103.042605 · arXiv:2006.14044
* [Bravyi 2024] S. Bravyi et al., *High-threshold and low-overhead fault-tolerant quantum memory*, Nature **627**, 778 (2024). doi:10.1038/s41586-024-07107-7 · arXiv:2308.07915
* [Bultrini 2023] D. Bultrini, M. H. Gordon, P. Czarnik, A. Arrasmith, M. Cerezo, P. J. Coles, L. Cincio, *Unifying and benchmarking state-of-the-art quantum error mitigation techniques*, Quantum **7**, 1034 (2023). doi:10.22331/q-2023-06-06-1034 · arXiv:2107.13470
* [Cai 2021] Z. Cai, *Quantum error mitigation using symmetry expansion*, Quantum **5**, 548 (2021). doi:10.22331/q-2021-09-21-548 · arXiv:2101.03151
* [Cai 2023] Z. Cai et al., *Quantum error mitigation*, Rev. Mod. Phys. **95**, 045005 (2023). doi:10.1103/RevModPhys.95.045005 · arXiv:2210.00921
* [Calderbank 1996] A. R. Calderbank, P. W. Shor, *Good quantum error-correcting codes exist*, Phys. Rev. A **54**, 1098 (1996). doi:10.1103/PhysRevA.54.1098 · arXiv:quant-ph/9512032
* [Cao 2012] Y. Cao, A. Daskin, S. Frankel, S. Kais, *Quantum circuit design for solving linear systems of equations*, Mol. Phys. **110**, 1675 (2012). doi:10.1080/00268976.2012.668289 · arXiv:1110.2232
* [Chamberland 2020] C. Chamberland, G. Zhu, T. J. Yoder, J. B. Hertzberg, A. W. Cross, *Topological and subsystem codes on low-degree graphs with flag qubits*, Phys. Rev. X **10**, 011022 (2020). doi:10.1103/PhysRevX.10.011022 · arXiv:1907.09528
* [Chao 2018] R. Chao, B. W. Reichardt, *Quantum error correction with only two extra qubits*, Phys. Rev. Lett. **121**, 050502 (2018). doi:10.1103/PhysRevLett.121.050502 · arXiv:1705.02329
* [Chen 2021] S. Chen, W. Yu, P. Zeng, S. T. Flammia, *Robust shadow estimation*, PRX Quantum **2**, 030348 (2021). doi:10.1103/PRXQuantum.2.030348 · arXiv:2011.09636
* [Chen 2024] J.-S. Chen et al., *Benchmarking a trapped-ion quantum computer with 30 qubits*, Quantum **8**, 1516 (2024). arXiv:2308.05071
* [Chen 2025] Z. Chen, B. Cheng, M. Gao, X. Lin, R. Zhang, Z. Wei, Z. Ji, *Scalable quantum error mitigation with neighbor-informed learning*, arXiv:2512.12578 (2025)
* [Childs 2010] A. M. Childs, W. van Dam, *Quantum algorithms for algebraic problems*, Rev. Mod. Phys. **82**, 1 (2010). doi:10.1103/RevModPhys.82.1 · arXiv:0812.0380
* [Cochran 1954] W. G. Cochran, *The combination of estimates from different experiments*, Biometrics **10**, 101 (1954). doi:10.2307/3001666
* [Coppersmith 1994] D. Coppersmith, *An approximate Fourier transform useful in quantum factoring*, IBM Research Report RC19642 (1994). arXiv:quant-ph/0201067
* [Czarnik 2021] P. Czarnik, A. Arrasmith, P. J. Coles, L. Cincio, *Error mitigation with Clifford quantum-circuit data*, Quantum **5**, 592 (2021). doi:10.22331/q-2021-11-26-592 · arXiv:2005.10189
* [Delfosse 2021] N. Delfosse, N. H. Nickerson, *Almost-linear time decoding algorithm for topological codes*, Quantum **5**, 595 (2021). doi:10.22331/q-2021-12-02-595 · arXiv:1709.06218
* [Dennis 2002] E. Dennis, A. Kitaev, A. Landahl, J. Preskill, *Topological quantum memory*, J. Math. Phys. **43**, 4452 (2002). doi:10.1063/1.1499754 · arXiv:quant-ph/0110143
* [Eastin 2009] B. Eastin, E. Knill, *Restrictions on transversal encoded quantum gate sets*, Phys. Rev. Lett. **102**, 110502 (2009). doi:10.1103/PhysRevLett.102.110502 · arXiv:0811.4262
* [Edmonds 1965] J. Edmonds, *Paths, trees, and flowers*, Can. J. Math. **17**, 449 (1965). doi:10.4153/CJM-1965-045-4
* [Efron 1993] B. Efron, R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall (1993)
* [Endo 2018] S. Endo, S. C. Benjamin, Y. Li, *Practical quantum error mitigation for near-future applications*, Phys. Rev. X **8**, 031027 (2018). doi:10.1103/PhysRevX.8.031027 · arXiv:1712.09271
* [Ezzell 2023] N. Ezzell, B. Pokharel, L. Tewala, G. Quiroz, D. A. Lidar, *Dynamical decoupling for superconducting qubits: a performance survey*, Phys. Rev. Applied **20**, 064027 (2023). doi:10.1103/PhysRevApplied.20.064027 · arXiv:2207.03670
* [Farhi 2014] E. Farhi, J. Goldstone, S. Gutmann, *A quantum approximate optimization algorithm*, arXiv:1411.4028 (2014)
* [Fowler 2012] A. G. Fowler, M. Mariantoni, J. M. Martinis, A. N. Cleland, *Surface codes: towards practical large-scale quantum computation*, Phys. Rev. A **86**, 032324 (2012). doi:10.1103/PhysRevA.86.032324 · arXiv:1208.0928
* [Gidney 2021] C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum **5**, 497 (2021). doi:10.22331/q-2021-07-06-497 · arXiv:2103.02202
* [Gidney 2021b] C. Gidney, M. Ekerå, *How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits*, Quantum **5**, 433 (2021). doi:10.22331/q-2021-04-15-433 · arXiv:1905.09749
* [Gidney 2025] C. Gidney, *How to factor 2048 bit RSA integers with less than a million noisy qubits*, arXiv:2505.15917 (2025)
* [Giurgica-Tiron 2020] T. Giurgica-Tiron et al., *Digital zero noise extrapolation for quantum error mitigation*, IEEE QCE 2020, p. 306. doi:10.1109/QCE49297.2020.00045 · arXiv:2005.10921
* [Google 2023] Google Quantum AI, *Suppressing quantum errors by scaling a surface code logical qubit*, Nature **614**, 676 (2023). doi:10.1038/s41586-022-05434-1
* [Google 2025] Google Quantum AI and collaborators, *Quantum error correction below the surface code threshold*, Nature **638**, 920 (2025). doi:10.1038/s41586-024-08449-y · arXiv:2408.13687
* [Gottesman 1997] D. Gottesman, *Stabilizer codes and quantum error correction*, PhD thesis, California Institute of Technology (1997). arXiv:quant-ph/9705052
* [Gottesman 2001] D. Gottesman, A. Kitaev, J. Preskill, *Encoding a qubit in an oscillator*, Phys. Rev. A **64**, 012310 (2001). doi:10.1103/PhysRevA.64.012310 · arXiv:quant-ph/0008040
* [Grover 1996] L. K. Grover, *A fast quantum mechanical algorithm for database search*, Proc. 28th ACM STOC, 212 (1996). doi:10.1145/237814.237866 · arXiv:quant-ph/9605043
* [Harrow 2009] A. W. Harrow, A. Hassidim, S. Lloyd, *Quantum algorithm for linear systems of equations*, Phys. Rev. Lett. **103**, 150502 (2009). doi:10.1103/PhysRevLett.103.150502 · arXiv:0811.3171
* [Hashim 2021] A. Hashim et al., *Randomized compiling for scalable quantum computing on a noisy superconducting quantum processor*, Phys. Rev. X **11**, 041039 (2021). doi:10.1103/PhysRevX.11.041039 · arXiv:2010.00215
* [Havlíček 2019] V. Havlíček et al., *Supervised learning with quantum-enhanced feature spaces*, Nature **567**, 209 (2019). doi:10.1038/s41586-019-0980-2 · arXiv:1804.11326
* [Higgins 2002] J. P. T. Higgins, S. G. Thompson, *Quantifying heterogeneity in a meta-analysis*, Stat. Med. **21**, 1539 (2002). doi:10.1002/sim.1186
* [Higgott 2022] O. Higgott, *PyMatching: a Python package for decoding quantum codes with minimum-weight perfect matching*, ACM Trans. Quantum Comput. **3**, 16 (2022). doi:10.1145/3505637 · arXiv:2105.13082
* [Higgott 2025] O. Higgott, C. Gidney, *Sparse Blossom: correcting a million errors per core second with minimum-weight matching*, Quantum **9**, 1600 (2025). doi:10.22331/q-2025-01-20-1600 · arXiv:2303.15933
* [Horsman 2012] C. Horsman, A. G. Fowler, S. Devitt, R. Van Meter, *Surface code quantum computing by lattice surgery*, New J. Phys. **14**, 123011 (2012). doi:10.1088/1367-2630/14/12/123011 · arXiv:1111.4022
* [Huang 2020] H.-Y. Huang, R. Kueng, J. Preskill, *Predicting many properties of a quantum system from very few measurements*, Nat. Phys. **16**, 1050 (2020). doi:10.1038/s41567-020-0932-7 · arXiv:2002.08953
* [Huggins 2021] W. J. Huggins et al., *Virtual distillation for quantum error mitigation*, Phys. Rev. X **11**, 041036 (2021). doi:10.1103/PhysRevX.11.041036 · arXiv:2011.07064
* [Javadi-Abhari 2024] A. Javadi-Abhari et al., *Quantum computing with Qiskit*, arXiv:2405.08810 (2024)
* [Jozsa 2001] R. Jozsa, *Quantum factoring, discrete logarithms, and the hidden subgroup problem*, Comput. Sci. Eng. **3**(2), 34 (2001). doi:10.1109/5992.909000 · arXiv:quant-ph/0012084
* [Kandala 2017] A. Kandala et al., *Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets*, Nature **549**, 242 (2017). doi:10.1038/nature23879 · arXiv:1704.05018
* [Kandala 2019] A. Kandala et al., *Error mitigation extends the computational reach of a noisy quantum processor*, Nature **567**, 491 (2019). doi:10.1038/s41586-019-1040-7 · arXiv:1805.04492
* [Kim 2023a] Y. Kim et al., *Scalable error mitigation for noisy quantum circuits produces competitive expectation values*, Nat. Phys. **19**, 752 (2023). doi:10.1038/s41567-022-01914-3 · arXiv:2108.09197
* [Kim 2023b] Y. Kim et al., *Evidence for the utility of quantum computing before fault tolerance*, Nature **618**, 500 (2023). doi:10.1038/s41586-023-06096-3
* [Kitaev 1995] A. Yu. Kitaev, *Quantum measurements and the Abelian stabilizer problem*, arXiv:quant-ph/9511026 (1995)
* [Kitaev 2003] A. Yu. Kitaev, *Fault-tolerant quantum computation by anyons*, Ann. Phys. **303**, 2 (2003). doi:10.1016/S0003-4916(02)00018-0 · arXiv:quant-ph/9707021
* [Klimov 2018] P. V. Klimov et al., *Fluctuations of energy-relaxation times in superconducting qubits*, Phys. Rev. Lett. **121**, 090502 (2018). doi:10.1103/PhysRevLett.121.090502 · arXiv:1809.01043
* [Knill 1997] E. Knill, R. Laflamme, *Theory of quantum error-correcting codes*, Phys. Rev. A **55**, 900 (1997). doi:10.1103/PhysRevA.55.900 · arXiv:quant-ph/9604034
* [Knill 1998] E. Knill, R. Laflamme, W. H. Zurek, *Resilient quantum computation*, Science **279**, 342 (1998). doi:10.1126/science.279.5349.342 · arXiv:quant-ph/9702058
* [Knill 2005] E. Knill, *Quantum computing with realistically noisy devices*, Nature **434**, 39 (2005). doi:10.1038/nature03350 · arXiv:quant-ph/0410199
* [Koczor 2021] B. Koczor, *Exponential error suppression for near-term quantum devices*, Phys. Rev. X **11**, 031057 (2021). doi:10.1103/PhysRevX.11.031057 · arXiv:2011.05942
* [Koh 2022] D. E. Koh, S. Grewal, *Classical shadows with noise*, Quantum **6**, 776 (2022). doi:10.22331/q-2022-08-16-776 · arXiv:2011.11580
* [Krantz 2019] P. Krantz et al., *A quantum engineer's guide to superconducting qubits*, Appl. Phys. Rev. **6**, 021318 (2019). doi:10.1063/1.5089550 · arXiv:1904.06560
* [Laflamme 1996] R. Laflamme, C. Miquel, J. P. Paz, W. H. Zurek, *Perfect quantum error correcting code*, Phys. Rev. Lett. **77**, 198 (1996). doi:10.1103/PhysRevLett.77.198 · arXiv:quant-ph/9602019
* [Lawson 1974] C. L. Lawson, R. J. Hanson, *Solving Least Squares Problems*, Prentice-Hall (1974)
* [Li 2017] Y. Li, S. C. Benjamin, *Efficient variational quantum simulator incorporating active error minimization*, Phys. Rev. X **7**, 021050 (2017). doi:10.1103/PhysRevX.7.021050 · arXiv:1611.09301
* [Li 2019] G. Li, Y. Ding, Y. Xie, *Tackling the qubit mapping problem for NISQ-era quantum devices*, ASPLOS 2019. doi:10.1145/3297858.3304023 · arXiv:1809.02573
* [Liao 2024] H. Liao, D. S. Wang, I. Sitdikov, C. Salcedo, A. Seif, Z. K. Minev, *Machine learning for practical quantum error mitigation*, Nat. Mach. Intell. **6**, 1478 (2024). doi:10.1038/s42256-024-00927-2 · arXiv:2309.17368
* [Logical QEM 2025] *Demonstrating quantum error mitigation on logical qubits*, Nat. Commun. (2025). doi:10.1038/s41467-025-67768-4 · arXiv:2501.09079
* [Lostaglio 2021] M. Lostaglio, A. Ciani, *Error mitigation and quantum-assisted simulation in the error corrected regime*, Phys. Rev. Lett. **127**, 200506 (2021). doi:10.1103/PhysRevLett.127.200506 · arXiv:2103.07526
* [Lowe 2021] A. Lowe et al., *Unified approach to data-driven quantum error mitigation*, Phys. Rev. Research **3**, 033098 (2021). doi:10.1103/PhysRevResearch.3.033098 · arXiv:2011.01157
* [Lubinski 2023] T. Lubinski et al., *Application-oriented performance benchmarks for quantum computing*, IEEE Trans. Quantum Eng. **4**, 3100332 (2023). arXiv:2110.03137
* [Maksymov 2023] A. Maksymov, J. Nguyen, Y. Nam, I. Markov, *Enhancing quantum computer performance via symmetrization*, arXiv:2301.07233 (2023)
* [Mari 2021] A. Mari, N. Shammah, W. J. Zeng, *Extending quantum probabilistic error cancellation by noise scaling*, Phys. Rev. A **104**, 052607 (2021). doi:10.1103/PhysRevA.104.052607 · arXiv:2108.02237
* [McArdle 2019] S. McArdle, X. Yuan, S. Benjamin, *Error-mitigated digital quantum simulation*, Phys. Rev. Lett. **122**, 180501 (2019). doi:10.1103/PhysRevLett.122.180501 · arXiv:1807.02467
* [McClean 2017] J. R. McClean, M. E. Kimchi-Schwartz, J. Carter, W. A. de Jong, *Hybrid quantum-classical hierarchy for mitigation of decoherence and determination of excited states*, Phys. Rev. A **95**, 042308 (2017). doi:10.1103/PhysRevA.95.042308 · arXiv:1603.05681
* [McClean 2020] J. R. McClean, Z. Jiang, N. C. Rubin, R. Babbush, H. Neven, *Decoding quantum errors with subspace expansions*, Nat. Commun. **11**, 636 (2020). doi:10.1038/s41467-020-14341-w · arXiv:1903.05786
* [McKay 2023] D. C. McKay et al., *Benchmarking quantum processor performance at scale*, arXiv:2311.05933 (2023)
* [Mirrahimi 2014] M. Mirrahimi et al., *Dynamically protected cat-qubits: a new paradigm for universal quantum computation*, New J. Phys. **16**, 045014 (2014). doi:10.1088/1367-2630/16/4/045014 · arXiv:1312.2017
* [Müller 2019] C. Müller, J. H. Cole, J. Lisenfeld, *Towards understanding two-level-systems in amorphous solids: insights from quantum circuits*, Rep. Prog. Phys. **82**, 124501 (2019). doi:10.1088/1361-6633/ab3a7e · arXiv:1705.01108
* [Murali 2020] P. Murali, D. C. McKay, M. Martonosi, A. Javadi-Abhari, *Software mitigation of crosstalk on noisy intermediate-scale quantum computers*, ASPLOS 2020. doi:10.1145/3373376.3378477 · arXiv:2001.02826
* [Nation 2021] P. D. Nation, H. Kang, N. Sundaresan, J. M. Gambetta, *Scalable mitigation of measurement errors on quantum computers*, PRX Quantum **2**, 040326 (2021). doi:10.1103/PRXQuantum.2.040326 · arXiv:2108.12518
* [Nielsen 2002] M. A. Nielsen, *A simple formula for the average gate fidelity of a quantum dynamical operation*, Phys. Lett. A **303**, 249 (2002). doi:10.1016/S0375-9601(02)01272-0 · arXiv:quant-ph/0205035
* [Nielsen 2010] M. A. Nielsen, I. L. Chuang, *Quantum Computation and Quantum Information*, 10th anniversary edition, Cambridge University Press (2010)
* [Panteleev 2021] P. Panteleev, G. Kalachev, *Degenerate quantum LDPC codes with good finite length performance*, Quantum **5**, 585 (2021). doi:10.22331/q-2021-11-22-585 · arXiv:1904.02703
* [Pérez-Salinas 2020] A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, J. I. Latorre, *Data re-uploading for a universal quantum classifier*, Quantum **4**, 226 (2020). doi:10.22331/q-2020-02-06-226 · arXiv:1907.02085
* [Peruzzo 2014] A. Peruzzo et al., *A variational eigenvalue solver on a photonic quantum processor*, Nat. Commun. **5**, 4213 (2014). doi:10.1038/ncomms5213 · arXiv:1304.3061
* [Piveteau 2021] C. Piveteau, D. Sutter, S. Bravyi, J. M. Gambetta, K. Temme, *Error mitigation for universal gates on encoded qubits*, Phys. Rev. Lett. **127**, 200505 (2021). doi:10.1103/PhysRevLett.127.200505 · arXiv:2103.04915
* [Proctor 2022] T. Proctor, K. Rudinger, K. Young, E. Nielsen, R. Blume-Kohout, *Measuring the capabilities of quantum computers*, Nat. Phys. **18**, 75 (2022). doi:10.1038/s41567-021-01409-7 · arXiv:2008.11294
* [Quek 2024] Y. Quek, D. Stilck França, S. Khatri, J. J. Meyer, J. Eisert, *Exponentially tighter bounds on limitations of quantum error mitigation*, Nat. Phys. **20**, 1648 (2024). doi:10.1038/s41567-024-02536-7 · arXiv:2210.11505
* [Roffe 2020] J. Roffe, D. R. White, S. Burton, E. Campbell, *Decoding across the quantum low-density parity-check code landscape*, Phys. Rev. Research **2**, 043423 (2020). doi:10.1103/PhysRevResearch.2.043423 · arXiv:2005.07016
* [Rudinger 2019] K. Rudinger et al., *Probing context-dependent errors in quantum processors*, Phys. Rev. X **9**, 021045 (2019). doi:10.1103/PhysRevX.9.021045 · arXiv:1810.05651
* [Sarovar 2020] M. Sarovar et al., *Detecting crosstalk errors in quantum information processors*, Quantum **4**, 321 (2020). doi:10.22331/q-2020-09-11-321 · arXiv:1908.09855
* [Seif 2023] A. Seif, Z.-P. Cian, S. Zhou, S. Chen, L. Jiang, *Shadow distillation: quantum error mitigation with classical shadows for near-term quantum processors*, PRX Quantum **4**, 010303 (2023). doi:10.1103/PRXQuantum.4.010303 · arXiv:2203.07309
* [Shor 1995] P. W. Shor, *Scheme for reducing decoherence in quantum computer memory*, Phys. Rev. A **52**, R2493 (1995). doi:10.1103/PhysRevA.52.R2493
* [Shor 1997] P. W. Shor, *Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer*, SIAM J. Comput. **26**, 1484 (1997). doi:10.1137/S0097539795293172 · arXiv:quant-ph/9508027
* [Simon 1997] D. R. Simon, *On the power of quantum computation*, SIAM J. Comput. **26**, 1474 (1997). doi:10.1137/S0097539796298637
* [Steane 1996] A. M. Steane, *Error correcting codes in quantum theory*, Phys. Rev. Lett. **77**, 793 (1996). doi:10.1103/PhysRevLett.77.793
* [Stouffer 1949] S. A. Stouffer et al., *The American Soldier, Vol. 1: Adjustment during Army Life*, Princeton University Press (1949)
* [Strikis 2021] A. Strikis, D. Qin, Y. Chen, S. C. Benjamin, Y. Li, *Learning-based quantum error mitigation*, PRX Quantum **2**, 040330 (2021). doi:10.1103/PRXQuantum.2.040330 · arXiv:2005.07601
* [Sundaresan 2020] N. Sundaresan et al., *Reducing unitary and spectator errors in cross resonance with optimized rotary echoes*, PRX Quantum **1**, 020318 (2020). doi:10.1103/PRXQuantum.1.020318 · arXiv:2007.02925
* [Suzuki 2022] Y. Suzuki, S. Endo, K. Fujii, Y. Tokunaga, *Quantum error mitigation as a universal error reduction technique*, PRX Quantum **3**, 010345 (2022). doi:10.1103/PRXQuantum.3.010345 · arXiv:2010.03887
* [Takagi 2022] R. Takagi, S. Endo, S. Minagawa, M. Gu, *Fundamental limits of quantum error mitigation*, npj Quantum Inf. **8**, 114 (2022). doi:10.1038/s41534-022-00618-z · arXiv:2109.04457
* [Temme 2017] K. Temme, S. Bravyi, J. M. Gambetta, *Error mitigation for short-depth quantum circuits*, Phys. Rev. Lett. **119**, 180509 (2017). doi:10.1103/PhysRevLett.119.180509 · arXiv:1612.02058
* [Tillich 2014] J.-P. Tillich, G. Zémor, *Quantum LDPC codes with positive rate and minimum distance proportional to the square root of the blocklength*, IEEE Trans. Inf. Theory **60**, 1193 (2014). doi:10.1109/TIT.2013.2292061 · arXiv:0903.0566
* [Tran 2023] M. C. Tran, K. Sharma, K. Temme, *Locality and error mitigation of quantum circuits*, arXiv:2303.06496 (2023)
* [van den Berg 2022] E. van den Berg, Z. K. Minev, K. Temme, *Model-free readout-error mitigation for quantum expectation values*, Phys. Rev. A **105**, 032620 (2022). doi:10.1103/PhysRevA.105.032620 · arXiv:2012.09738
* [van den Berg 2023] E. van den Berg, Z. K. Minev, A. Kandala, K. Temme, *Probabilistic error cancellation with sparse Pauli–Lindblad models on noisy quantum processors*, Nat. Phys. **19**, 1116 (2023). doi:10.1038/s41567-023-02042-2 · arXiv:2201.09866
* [Vandersypen 2001] L. M. K. Vandersypen et al., *Experimental realization of Shor's quantum factoring algorithm using nuclear magnetic resonance*, Nature **414**, 883 (2001). doi:10.1038/414883a · arXiv:quant-ph/0112176
* [Viola 1998] L. Viola, S. Lloyd, *Dynamical suppression of decoherence in two-state quantum systems*, Phys. Rev. A **58**, 2733 (1998). doi:10.1103/PhysRevA.58.2733 · arXiv:quant-ph/9803057
* [Wahl 2023] M. Wahl et al., *Zero noise extrapolation on logical qubits by scaling the error correction code distance*, IEEE International Conference on Quantum Computing and Engineering (QCE 2023). arXiv:2304.14985
* [Wallman 2016] J. J. Wallman, J. Emerson, *Noise tailoring for scalable quantum computation via randomized compiling*, Phys. Rev. A **94**, 052325 (2016). doi:10.1103/PhysRevA.94.052325 · arXiv:1512.01098
* [Yoshioka 2022] N. Yoshioka, H. Hosaka, T. Kobayashi, S. Endo, Y. Suzuki, *Generalized quantum subspace expansion*, Phys. Rev. Lett. **129**, 020502 (2022). doi:10.1103/PhysRevLett.129.020502 · arXiv:2107.02611
* [Zhao 2022] P. Zhao et al., *Quantum crosstalk analysis for simultaneous gate operations on superconducting qubits*, PRX Quantum **3**, 020301 (2022). doi:10.1103/PRXQuantum.3.020301 · arXiv:2110.12570
