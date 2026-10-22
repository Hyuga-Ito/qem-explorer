# Device versus model: measuring and explaining the gap

The divergence study asks two questions of an IBM device. **Where** does it perform worse than the noise model built
from its own calibration? And **why**? This chapter walks through the pipeline from a manifest to the diagnostic
matrix, explains every figure it draws and how to read it, and shows how each conclusion is tested on emulated devices
with known faults before any QPU time is spent.

## 1. The question, made precise

For every benchmark circuit the study compares two output distributions:

* the **observed** one, measured on the device (or on an emulated device);
* the **predicted** one, simulated from the device's calibration at the time of the run.

Both are scored against the ideal output $p$ with the **classical (Hellinger) fidelity**
$F_c(p,q)=\bigl(\sum_x\sqrt{p_xq_x}\bigr)^2$. Wide circuits approach the uniform distribution, which still scores
above 0, so the study also uses the **normalized fidelity** $F_N=(F_c-F_c^{\rm unif})/(1-F_c^{\rm unif})$, the part
above random guessing. The **divergence** is

$$\Delta=\overline{F_c^{\rm pred}}-F_c^{\rm obs},$$

positive when the device is worse than its model.

**Three predictors** span "how much the model knows":

| predictor | contains |
|---|---|
| depolarizing (primary) | every gate followed by the depolarizing channel with the reported error, plus readout; no idle error. The component model |
| median | one median rate per gate class (Chen et al.'s model), no readout |
| device | the full fake-device model: relaxation during gates and idle windows on the ALAP schedule, plus readout |

**Same estimator, same shots.** A prediction is passed through exactly the estimator the observation went through
(raw, readout-mitigated or ZNE), at the same shots, over $R$ Monte-Carlo replicates. Both sides then carry the same
finite-shot bias, and

$$z=\frac{\Delta}{\sqrt{(\sigma_{\rm rep}^2+\mathrm{SE}_{\rm boot}^2)/2}}$$

measures significance in units of shot noise. **Null check:** when the emulated device *is* the depolarizing model,
the 236 z-scores have mean −0.06 and SD 0.90 (KS $p=0.13$), and none exceeds 3. A significant Δ is therefore not an
artifact of the estimators.

## 2. Reading the figures

All figures share one **color grammar**:

* **red** means the device is worse than its model, **blue** better, **gray** as modeled, on a symmetric diverging
  scale;
* circuit families are **shapes**, never colors, so color stays free for the value;
* a **filled** marker is beyond shot noise ($\lvert z\rvert>3$), an open marker is within it;
* every figure of an emulated device says "emulated" in its title.

| figure | the question it answers | how to read it |
|---|---|---|
| fidelity gap | how much faster does the device lose fidelity per two-qubit gate? | $\ln F_c$ against two-qubit gates for observation and prediction, each circuit's two points joined; the fitted rates $\varepsilon_{\rm eff}^{\rm obs}$, $\varepsilon_{\rm eff}^{\rm pred}$ at the lines. Their ratio is the headline number |
| difference volumetric | where on the size plane does the model fail? | tiles at (two-qubit gates, width), colored by mean Δ; a dot marks a tile whose circuits are mostly significant |
| volumetric, excess per gate | the same, comparable across sizes | $e=(\ln F_N^{\rm pred}-\ln F_N^{\rm obs})/n_2$: Δ vanishes again for large circuits (both sides reach the floor), $e$ does not |
| predicted vs observed | which circuits leave the diagonal? | color = significance class, shape = family |
| χ² ladder | is it the calibration or the context? | $\chi^2/\mathrm{dof}$ of four nested explanations, with the verdict underneath |
| divergence budget | which context? | each mechanism's mean contribution per circuit, with intervals; per family, with the unexplained part hatched |
| folding fingerprint | gate, time or coherent build-up? | $(A_{\rm global},A_{\rm local})$ per family against the regions of the reading table |
| context probes | do spectators matter, and which kind? | the three probe modes, observed solid and predicted dashed; the excess ±2 SE |
| heavy-hex map | where on the chip? | each qubit's unmodeled error per two-qubit-gate participation, ringed where significant |
| diagnostic matrix | do all diagnostics agree? | one row per data set, one column per diagnostic; a red dot for an anomaly, an open ring for "as modeled" |

## 3. Explaining the gap

The response is each circuit's excess log-infidelity of the readout-mitigated estimate,
$y=\ln F_N^{\rm pred}-\ln F_N^{\rm obs}$, weighted by its standard error so that $\chi^2/\mathrm{dof}\approx1$ marks a
sufficient explanation.

1. **The χ² ladder.** Four nested non-negative fits answer "calibration or context?":
   * *none*: the model is right;
   * *rescaled calibration*: $y=c\,(-\ln F_{\rm pred})$, every calibrated error $(1+c)$ times too low;
   * *context*: $y=\sum_k\beta_kx_k$ over the context features;
   * *both*.

   The verdict is *rescaled calibration* when context adds nothing significant, or nothing material (the calibration
   alone removes ≥ 95 % of what all mechanisms together remove). "+ non-linear" is appended when even the full fit
   leaves $\chi^2/\mathrm{dof}>2$, the mark of coherent or non-additive errors.
2. **The divergence budget.** A non-negative least-squares split over the mechanisms, with bootstrap intervals. A
   variance inflation factor above 10 marks mechanisms that these circuits cannot separate. In application circuits,
   spectators, parallel gates and idle time grow together, which is why the designed probes exist.
3. **The folding fingerprint** (*Inside ZNE*, §5) separates gate-attached, time-attached and repetition-coherent
   errors from the ZNE runs alone.
4. **The probes** (*Context noise*, §5) separate idle dephasing, gate error and spectator crosstalk.
5. **The layout variants** detect position dependence: a defect on one qubit is invisible to every pooled diagnostic.
6. **The $\varepsilon$ ratio** $\varepsilon_{\rm eff}^{\rm obs}/\varepsilon_{\rm eff}^{\rm pred}$ summarizes the size.

## 4. Emulated devices: test the pipeline on known causes

Before hardware, every diagnostic is run on an **emulated device**: a predictor's model of the calibration plus a
known error that the model does not contain (`emulate --inject`).

| injection | what it adds | the expected reading |
|---|---|---|
| `scale=s` | every gate error and relaxation rate × s | gate-attached; rescaled calibration |
| `coherent=χ` | a coherent fraction χ of every gate's error, at the same infidelity | non-additive |
| `crosstalk=ζ` | $e^{-i\zeta Z\otimes Z/2}$ on every spectator pair after every two-qubit gate | spectator context; repetition-coherent |
| `t2=τ` | every $T_2$ lowered to τ µs | time-attached; idle dephasing |
| `hotspot=q` | every gate on physical qubit q at 5× its reported error | position-dependent |

On the v2 suite (52 benchmarks × 3 layouts on fake_fez) every planted cause is named correctly:

| planted | ladder | fingerprint $(A_g,A_l)$ | probes | ε ratio |
|---|---|---|---|---|
| none | as modeled | 0.96, 0.99 | none | 1.03 |
| gate errors × 1.6 | rescaled calibration × 1.57 | 1.31, 1.27: gate-attached | driven only | 1.60 |
| $T_2$ = 30 µs | context + non-linear | 1.12, 0.97: time-attached | superposed only | 1.28 |
| ZZ 0.05 rad | context + non-linear | 2.11, 2.75: repetition-coherent | both | 1.66 |

**This is the validation protocol.** A reading on hardware is trusted only after the same pipeline, on the same
circuits and budget, has named each planted cause. An emulation never replaces hardware data. It shows what each
figure looks like when the cause is known.

## 5. The pipeline, command by command

```
# 1. plan: compile every benchmark on its chains (offline on the fake twin, or on today's calibration)
python scripts/dirA_divergence.py plan manifests/direction_A_divergence_v2.yaml --out data/div_v2
# 2. observe: an emulated device, the local Sampler on the fake twin, or the real device
python scripts/dirA_divergence.py emulate   data/div_v2 --tag xtalk --truth depolarizing --inject crosstalk=0.05
python scripts/dirA_divergence.py run-local data/div_v2
python scripts/dirA_divergence.py run-ibm   data/div_ibm       # a plan made with --backend ibm_fez; then collect-ibm
# 3. predict, compare, draw
python scripts/dirA_divergence.py analyze  data/div_v2 --tag xtalk --out figs_xtalk --workers 8
python scripts/dirA_divergence.py recovery "no injection=figs_null" "ZZ 0.05=figs_xtalk" --out figs_recovery
python scripts/dirA_divergence.py figures  figs_xtalk --run-dir data/div_v2 --out figs_xtalk_thesis --font lm
```

**Other fake devices.** `plan --backend fake_torino` (or any fake device of the app) compiles the same manifest on
another calibration. An Eagle device such as fake_brisbane adds the ECR control-spectator mechanism. Comparing devices
asks whether a reading belongs to the device or to the method.

## 6. In the app: the closest view

The Playground compares methods, not a device with its model. Its closest view of the same idea is *Model mismatch →
Assumed vs true rate* with a fake device as the simulator:

* the truth is the full device model;
* PEC is told the device's **median** depolarizing rates, times a ratio.

On a four-qubit QFT on fake_fez, PEC at ratio 1 removes only part of the bias, and the curve crosses zero well above
ratio 1. The crossing absorbs everything the median model leaves out: relaxation in idle windows, the spread of
per-qubit rates and the readout error. That is why the study separates readout (readout-mitigated estimates) and uses
three predictors.

## Try it

Press **Model mismatch · QFT on fake_fez** and run *the mismatch curve*. Then run an emulated analysis yourself
(`python scripts/rehearse_divergence.py manifests/direction_A_divergence_v3_qpu600.yaml --out data/reh_v3`) and read
its figures, starting with the divergence gap and the diagnostic matrix; the *Noise Models & Discrepancy* page shows
the bundled rehearsal. The theory is in *Theory & sources* §19–21.
