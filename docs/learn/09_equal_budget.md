# Comparing methods fairly: equal budgets and phase diagrams

"Which mitigation method is best?" has no answer until two things are fixed: **how many shots** you may spend in
total, and **how noisy** the circuit is. QEM Explorer therefore compares every method at the same *total* shot budget
and maps the winner over both axes. This chapter explains the bookkeeping, how to read the resulting charts, and the
noise axis behind the research maps, the light-cone fault rate $\lambda_{LC}$.

## 1. One budget for everything

A mitigated estimate does not come from the target circuit alone. ZNE runs folded copies, CDR and NIL first run
training circuits, and readout mitigation runs calibration circuits. All of these shots come out of one budget $B$:

$$B=N_t+T\,s_{\rm tr}+c\,s ,$$

with $N_t$ shots on the target circuits, $T$ training circuits of $s_{\rm tr}$ shots each, and $c$ calibration
circuits of $s$ shots each. Each part adds its own term to the error:

$$\mathrm{RMSE}(B)=\sqrt{b^2+\frac{v}{N_t}+\frac{w}{s}+u}.$$

| term | what it is | example |
|---|---|---|
| $b$ | the bias that remains after mitigation | ZNE's extrapolation error; PEC with a wrong model |
| $v$ | the variance per target shot, about $\gamma^2\sigma^2$ | grows with ZNE's $\sum_j\lvert\eta_j\rvert$ and PEC's $\gamma$ |
| $w$ | how much a noisy readout calibration moves the estimate | readout inversion, M3, TREX |
| $u$ | the shot noise of the training data, frozen into the fitted map | CDR, NIL |

The training shots are an **entry fee**: a method cannot give an estimate below $T\,s_{\rm tr}$ shots. $b$ and $u$
together form a **floor** that more target shots cannot lower.

**In the app:** the cost table of *Mitigate & compare* gives every method's bias, sd, $\gamma$, its target, training
and calibration shots, and the number of distinct circuits it sends, all at the sidebar's budget. *Cost vs precision*
turns those numbers into curves.

## 2. Three regimes and the crossover budget

Write $F=b^2+u$ for a method's floor and $V$ for its effective variance coefficient. Then roughly
$\mathrm{RMSE}\approx\sqrt{F+V/B}$, and along the budget axis every comparison passes through up to three regimes:

1. **Variance-limited** (small $B$). The method with the smallest $V$ wins among those that can run at all. This is
   often unmitigated estimation.
2. **Crossover.** A less biased but noisier method $j$ overtakes method $i$ at
   $$B^*_{i\to j}=\frac{V_j-V_i}{F_i-F_j}.$$
3. **Floor-limited** (large $B$). The smallest floor wins, whatever it costs per shot.

**Winning is not reaching.** The smallest budget that reaches a target precision $\varepsilon$ is
$B(\varepsilon)=V/(\varepsilon^2-F)$. It is infinite when $F\ge\varepsilon^2$: no number of shots helps. The method
that wins at large $B$ is often not the cheapest way to reach $\varepsilon$, and sometimes *no* method reaches it.

**Ties.** A winner within 5 % of the runner-up is marked as a tie in the research maps. A winner inside its own error
bar is not a winner.

## 3. Reading the app's budget charts

| page | chart | how to read it |
|---|---|---|
| Cost vs precision | RMSE against budget | each curve falls like $1/\sqrt B$ and flattens at its floor; dots mark where a method overtakes the unmitigated one |
| Cost vs precision | table | *budget for ε* is $B(\varepsilon)$, "∞" when the floor is above ε; *beats unmitigated above B* is $B^*$ |
| Cost vs precision | shots needed against ε | the price of precision, with your QPU time budget drawn as a line |
| Head-to-head | who wins where | the winner on the plane of a noise knob × budget: the app's own phase diagram |
| Batch run | best method per scenario | the winners at the budget you choose; change the budget without rerunning |

**Worked example (QFT on 4 qubits, *Hardware-realistic* noise).** The unmitigated estimate of $P(0101)$ is biased by
−0.18. At the default 4,096 shots the cost table reads:

* PEC overtakes the unmitigated estimate above about 9 shots, exponential ZNE above about 37, Richardson ZNE above
  about 85. CDR does so only above about 16,400 shots, because its 16 training circuits × 1,024 shots must be paid
  first.
* For ε = 0.01 every row says ∞. Every method's floor lies above 0.01 here: PEC's depolarizing model misses the
  coherent and idle parts of this noise, and ZNE's fit cannot follow them.
* For ε = 0.05 only CDR reaches the target, at about $1.7\times10^4$ shots (about 4 s of QPU time at 250 µs per
  shot).

In *Head-to-head*, sweeping the noise scale from 0.5 to 3 gives three bands: PEC at the smallest budgets, exponential
ZNE between about $10^2$ and $10^4$ shots, and CDR above about $3\times10^4$. Switch the noise to *Depolarizing* and the
map changes completely. PEC's model is then exact, and on this small circuit ($\gamma\approx1.4$) PEC wins almost
everywhere. **The structure of the noise, not only its strength, decides the winner.**

## 4. A noise axis you can trust: the light-cone fault rate

A phase diagram needs a horizontal axis that means the same thing for every circuit and device. A gate count does
not: it ignores idle time, readout and gate quality. QEM Explorer uses the expected number of faults that can reach
the observable.

* **Fault probability.** Every noise location $f$ (the error after a gate, an idle window, a crosstalk kick) has a
  probability $p_f=1-p_I$ of applying a non-identity Pauli, read from the diagonal of its Pauli transfer matrix. The
  formula is exact for any channel: a coherent rotation by $\theta$ has $p_f=\sin^2(\theta/2)$.
* **Light cone.** Only faults in the observable's backward light cone can change it. Summing over them gives
  $\lambda_{LC}=\sum_{f\in{\rm LC}}p_f^{\rm LC}$, and $e^{-\lambda_{LC}}$ is the probability that no relevant fault
  occurs.
* **Readout** is reported separately as $\lambda_{\rm ro}$, because readout mitigation removes it with its own
  calibration.

Why it is a good axis:

* it can be computed from the calibration **before** any circuit runs;
* it grows in proportion to the noise scale $s$ (within 1 % over $s$ = 0.25–6 on fake_fez);
* PEC's overhead follows it, so phase boundaries drawn in $\lambda_{LC}$ can be compared across circuits and
  devices. Stage C of the research manifest tests whether they collapse.

Two warnings:

* **Sensitivity differs.** The unmitigated signal decays like $e^{-\kappa\lambda_{LC}}$ with a circuit-dependent
  $\kappa$, between 0.34 (Ising-6) and 1.09 (GHZ-6) on fake_fez, because some faults commute with the observable.
* **Coherence is invisible.** A coherent error and a stochastic error of the same infidelity have the same
  $\lambda_{LC}$, but they behave differently when gates are repeated (see *Inside ZNE*).

**In the app:** ZNE's *Inside each method* panel reports $\lambda_{\rm eff}$, a ratio of light-cone fault rates
(next chapter). The routing guide's *expected success F* is the same idea at the level of a whole layout.

## 5. Phase diagrams at research scale

The research backend `qemx.phase` evaluates the ledger on a grid of **cells**. A cell is one circuit at one noise
level on one noise engine, with every method run on it. Two engines move the noise along a known axis:

* **E2** multiplies every gate error and relaxation rate of a fake device by $s$, then simulates it exactly;
* **E1** homogenizes the device to its median rates, scales every channel's *fault probability* by exactly $s$, and
  can turn a fraction $\chi$ of each gate's error into a coherent over-rotation at fixed infidelity, hence at fixed
  $\lambda_{LC}$.

Each method's cost then follows a **two-constant law**, $\ln V=\ln A+\beta\lambda_{LC}$. The cost exponent $\beta$
is 2.7–5.1 for PEC on the six-qubit circuits of fake_fez and close to 0 for Richardson ZNE, whose weakness is bias,
not variance. Before any production run the ledger is validated by Monte Carlo against pre-registered criteria: a
Kolmogorov–Smirnov test of the z-scores, and $\lvert z\rvert>4$ in fewer than 1 % of the checks.

```
python -m qemx.phase.run plan  manifests/direction_A_v1.yaml
python -m qemx.phase.run run   manifests/direction_A_v1.yaml --out data/dirA_v1 --workers 8
python scripts/dirA_checks.py  data/dirA_v1 --eps 0.01
python scripts/dirA_figures.py data/dirA_v1 --out figs --only phase --eps 0.01
```

The manifest's stages ask separate questions:

* **A1** is fake_fez at eight noise scales; **A2** is the same without readout error;
* **B** adds a coherent fraction $\chi$ at fixed $\lambda_{LC}$;
* **C** transfers the maps to other calibrations (fake_torino, fake_kingston, fake_brisbane);
* **D–F** are an Aer cross-check, hyperparameter sensitivity, and transpiler level 3.

## 6. What the fake_fez maps say

On fake_fez (1,705 cells):

* **No method wins everywhere.** The budget matters as much as the noise.
* **Unmitigated estimation wins small budgets.** For example, a six-qubit random brickwork circuit below
  $7.9\times10^5$ shots at $\lambda_{LC}=0.016$.
* **ZNE's large-budget wins sit next to the zeros of its signed bias.** These are cancellation wins, not a smaller
  bias in general. The maps therefore show $b/\lvert b_{\rm RAW}\rvert$ beside the winners.
* **Learning methods pay an entry fee and hit a floor**, both set by their training shots.
* **A coherent fraction rewrites the map at fixed $\lambda_{LC}$.** On AQFT-6, $\chi=0.5$ raises exponential ZNE's
  bias from 0.016 to 0.609.
* **Large regions are unreachable.** For QFT-6, no method reaches RMSE 0.01 at six of eight noise levels.

The lesson leads straight into the rest of the Research topics: a single noise number does not determine what a device
does to a circuit. **How the noise is structured** matters, and that structure is what a component-level noise model
gets wrong.

## Try it

Press **Cost vs precision · QFT under realistic noise**, then **Run simulation** in the sidebar, and read the table
for ε = 0.01 and ε = 0.05. Then open *Head-to-head*: the sweep over the noise scale is already set up; press **Run
sweep** and find the three regimes in *Who wins where*. The hands-on guide *Who wins depends on the budget* walks
through it step by step. The theory is in Part VI of *Theory & sources* (§18).
