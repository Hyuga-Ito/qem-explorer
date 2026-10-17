# Context noise: when a gate errs differently in company

A device's calibration lists one error rate per gate, measured on otherwise idle qubits. A circuit never runs a gate
alone. The neighbors may be in superposition, other gates may fire at the same moment, and qubits wait while others
work. When the error of a gate depends on what happens around it, the noise is **context-dependent**, and a model
built from isolated gate errors cannot predict it. This chapter explains the mechanisms on IBM's heavy-hex devices,
how QEM Explorer counts a circuit's exposure to them, and the designed experiments that separate them.

## 1. What a calibration measures, and what it leaves out

A fake device, like the calibration it copies, is a **component model**:

* average gate errors from randomized benchmarking, one gate at a time;
* readout assignment errors;
* $T_1$, $T_2$ and gate durations.

Whatever happens only *between* components is missing:

| mechanism | physical origin | grows with | closest knob in the app's noise injector |
|---|---|---|---|
| ZZ to spectators | always-on coupling between neighboring transmons | neighbors in superposition during a gate | *Crosstalk (spectator)*, ZZ term |
| spectator phase kicks | a drive on one qubit shifts its neighbor's frequency | idle neighbors of driven qubits | *Crosstalk (spectator)*, Z term |
| simultaneous-gate crosstalk | two gates on adjacent couplers at the same time | parallel two-qubit gates | (study pipeline only) |
| control-qubit drive crosstalk | a cross-resonance pulse also drives the control's other neighbors | ECR gates on Eagle devices | (study pipeline only) |
| non-exponential dephasing | low-frequency (1/f) noise that $T_2$ does not describe | idle time in superposition | *Random field (quasi-static)* |
| Markovian dephasing | white phase noise | idle time in superposition | *Markovian dephasing* |
| two-level-system defects | one qubit couples to a wandering defect | where on the chip the circuit runs | (emulated as a hot spot) |
| drift | the device changes between calibration and run | time since calibration | (a second run day) |

Layer fidelities measured on whole simultaneous layers are accordingly lower than the product of isolated gate
fidelities, and circuit performance departs from component metrics in structured ways. These are the effects this
part of the app is built to see.

## 2. Heavy hex, chains and spectators

IBM's heavy-hex lattice gives every qubit at most three neighbors. The divergence study lays each benchmark on a
**chain**, a path of the coupling graph chosen from the calibration, so that routing stays confined and comparable
across widths. Every neighbor of a chain qubit is a **spectator**:

* an **active** spectator is one of the circuit's own qubits next to a gate. It is often in superposition;
* a **passive** spectator is a device qubit outside the circuit. It idles in $\lvert0\rangle$.

Two device generations behave differently:

* **Heron** devices (fake_fez, fake_marrakesh, fake_kingston, fake_aachen, fake_torino) use tunable couplers and CZ
  gates.
* **Eagle** devices (fake_brisbane, fake_sherbrooke, fake_kyiv, fake_osaka) use fixed couplers and echoed
  cross-resonance (ECR) gates. Only there does a gate also drive the control qubit's other neighbors, and only there
  is the `control_spectators` count non-zero.

## 3. Counting context in a compiled circuit

Every planned circuit is scheduled as late as possible, as IBM backends do, and its exposure is counted on the
device's coupling graph (`qemx.context_noise.circuit_features`):

| feature | what is counted |
|---|---|
| `n2q`, `n1q` | two-qubit and physical one-qubit gates ($R_Z$ is free) |
| `gate_error`, `readout_error` | the sums of the reported errors |
| `active_spectators` | over all two-qubit gates: the circuit's own qubits coupled to the gate's qubits |
| `passive_spectators` | the same for device qubits outside the circuit |
| `control_spectators` | ECR gates only: neighbors of the driven control qubit |
| `parallel_adjacent` | pairs of two-qubit gates at the same time on adjacent couplers |
| `idle_us` | idle time between each qubit's first gate and its measurement |
| `priced` | the model's own predicted log-infidelity $-\ln F_{\rm pred}$: everything the calibration already accounts for |

`priced` is the key regressor. A spectator count only means something next to the error the model already expects.

## 4. The same circuit, somewhere else: layout variants

A gate fidelity is a number per gate. A context effect is a number per **situation**. The cleanest way to expose it is
to run the same benchmark in several places and see whether the places disagree. `variants: {mode: …}` in a manifest
chooses what changes:

| mode | what differs between the variants of one benchmark |
|---|---|
| `layouts` | the chain: disjoint chains (best calibrated first), then the same chains reversed, then overlapping chains; never one assignment twice |
| `twirl` | the gate frame: an independent Pauli twirl of every two-qubit gate, on the same chain |
| `layouts+twirl` | both at once |
| `seeds` | the transpiler seed, on the same chain |

None of these changes the ideal output. **Do the variants agree?** With each variant's divergence $\Delta_i$ and
standard error $\mathrm{SE}_i$, Cochran's $Q=\sum_iw_i(\Delta_i-\bar\Delta)^2$ (weights $w_i=\mathrm{SE}_i^{-2}$) is
$\chi^2$-distributed with $k-1$ degrees of freedom when all $k$ variants share one divergence. $I^2=(Q-(k-1))/Q$ is
the share of the spread that shot noise does not explain (0 when $Q<k-1$). The study flags **position dependence** when at least three
benchmarks, and more than 10 % of them, have $p<0.01$, where 1 % would be chance.

## 5. A designed experiment: spectator-context probes

Application circuits mix every mechanism at once. A **context probe** isolates them. A random two-qubit core runs on
$(c_0,c_1)$ of the chain $a$–$s_0$–$c_0$–$c_1$–$s_1$–$b$, and its spectators $s_0,s_1$ are put into one of three modes:

| mode | spectators | what it exposes |
|---|---|---|
| `zero` | idle in $\lvert0\rangle$ | the reference |
| `plus` | idle in $\lvert+\rangle$ | dephasing of waiting qubits; ZZ phases on superposed neighbors |
| `driven` | running their own CZ gates with $a$ and $b$ | gate errors next to the core; simultaneous-drive crosstalk |

The ideal output is the same in all three modes, and barriers keep the schedule identical. For each mode the
**excess contrast** compares the drop on the device with the drop the model predicts:

$$C_{\rm mode}=\bigl[F_c(\text{zero})-F_c(\text{mode})\bigr]_{\rm obs}-\bigl[F_c(\text{zero})-F_c(\text{mode})\bigr]_{\rm pred}.$$

The z-scores over core depths and layout variants are combined by Stouffer's method, $Z=\sum_iz_i/\sqrt n$. With
$Z>3$ as the threshold, the reading is:

* superposed only → idle dephasing;
* driven only → gate error;
* both → spectator crosstalk;
* neither → no spectator effect.

## 6. The probe in miniature, in the app

The two setups below run a four-qubit version on a line $s_0$–$c_0$–$c_1$–$s_1$ (qubits 0–3). The core is eight CZ
gates between two Hadamard layers, an identity overall. The spectators q0 and q3 are either left in
$\lvert0\rangle$ (`zero`) or put into $\lvert+\rangle$ and back with Hadamards (`plus`). The observable is the
spectator $s_0$ itself, $\langle Z_0\rangle$, whose ideal value is 1 in both modes. Run both probes under each
noise preset:

| sidebar noise | `zero` | `plus` | `plus` with DD (XY4) | `plus` with ZNE |
|---|---|---|---|---|
| Depolarizing | 1.000 | 0.996 | 0.988 | 1.000 |
| Crosstalk (spectator) | 1.000 | 0.842 | 0.997 | 1.473 |
| Markovian dephasing | 1.000 | 0.881 | 0.863 | 0.996 |
| Random field (quasi-static) | 1.000 | 0.491 | 1.000 | 0.881 |

What the table teaches:

* **An idle $\lvert0\rangle$ spectator is blind.** Phase errors do nothing to it, which is why the `zero` mode is the
  reference and the `plus` mode is the probe.
* **The depolarizing model predicts almost no drop** (0.996, from the two extra Hadamards). Everything beyond that is
  context that a component model would miss. Under *Crosstalk*, the excess contrast is
  $(1-0.842)-(1-0.996)\approx0.15$.
* **The response to a remedy names the mechanism.**
  * Dynamical decoupling refocuses errors with a fixed sign: crosstalk kicks and a quasi-static field are removed
    (0.997, 1.000).
  * White Markovian dephasing has no sign to refocus, and the extra pulses make it slightly worse (0.863).
  * Global ZNE, which stretches every idle window, removes the Markovian dephasing (0.996) but overshoots on the
    coherent kicks (1.473). These are the *time-attached* and *coherent* readings of the folding fingerprint.

**Limits of the miniature.**

* The injector's crosstalk acts only on *idle* neighbors of a driven qubit, so two simultaneous gates protect each
  other here. A `driven` mode therefore needs the study pipeline, whose emulated crosstalk adds a ZZ phase on every
  spectator pair after every two-qubit gate.
* The study also measures the whole output distribution ($F_c$) rather than one qubit.

## 7. Merging variants: the plurality vote

Variants that disagree can also be *combined*. Simple aggregation averages their histograms. Chen et al.'s
**approximate plurality vote** keeps only the outcomes that at least $t$ variants saw:

* the threshold is $t=\min(t_0,\max_b o_b)$, with $t_0=\mathrm{round}(0.28V)$ and at least 3;
* each survivor is weighted by the probability that at least $t$ variants would draw it.

QEM Explorer computes that probability exactly, with a Poisson-binomial recursion, and follows the authors' reference
implementation (raw frequencies). The vote is a **decoder for peaked outputs**. It sharpens BV or a QFT on a basis
state, but it lowers the fidelity of broad outputs, so the study reports both aggregations.

## Try it

Press **Spectator probe · spectators in |0⟩**, then **Spectator probe · spectators in |+⟩**, and run each under the
four noise presets of the table. The hands-on guide *Spectators: find the context noise* has the steps. For the
driven direction, *Spectator Ramsey* drives the middle qubit and watches its idle neighbors. The theory is in
*Theory & sources* §19.1 and §20.
