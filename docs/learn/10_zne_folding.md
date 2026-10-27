# Inside ZNE: global, local and fractional folding

Zero-noise extrapolation measures $E(\lambda)$ at amplified noise and extrapolates to $\lambda=0$. Everything rests on
one assumption: the circuit run at "scale λ" really carries λ times the noise. This chapter shows that the way you fold
decides whether that is true, how QEM Explorer measures the scale actually reached, and how the difference between two
foldings becomes a measurement of the noise itself.

## 1. Two ways to fold

| | global folding | local folding |
|---|---|---|
| what is repeated | the whole circuit: $U\to U(U^\dagger U)^k$ | every two-qubit gate: $G\to G(G^\dagger G)^k$ |
| two-qubit gates at λ = 3 | 3× | 3× |
| one-qubit gates | repeated | **not** repeated |
| idle windows | stretched: the whole schedule runs three times | barely: only qubits waiting while a folded gate runs |
| the same gate back to back | no: $U^\dagger$ sits between two copies of $U$ | **yes**: $G\,G^\dagger G$ |
| depth | about 3× | smaller |

At the same nominal λ both run the same number of two-qubit gates, so on hardware they cost the same shots. They do
**not** amplify the same noise.

## 2. Nominal and effective λ

The nominal scale counts two-qubit gates. The **effective** scale is the factor by which folding raises the
light-cone fault rate of the noise model (previous chapter):

$$\lambda_{\rm eff}=\frac{\lambda_{LC}(\text{folded})}{\lambda_{LC}(\text{original})}.$$

For a four-qubit QFT in the app (λ = 1, 3, 5):

| noise preset | global $\lambda_{\rm eff}$ | local $\lambda_{\rm eff}$ | why |
|---|---|---|---|
| Depolarizing | 1, 3, 5 | 1, 2.85, 4.70 | local folding leaves the one-qubit gates' errors unfolded |
| Markovian dephasing | 1, 3, 5 | 1, 2.33, 3.67 | idle dephasing grows with time, and local folding adds little time |
| Coherent over-rotation | 1, 3, 5 | 1, 1, 1 | the preset's error sits on one-qubit gates, which local folding never repeats |

The gap matters. Richardson extrapolation with weights $\eta_j$ computed for the nominal $\lambda_j$ leaves, to first
order, the fraction

$$\rho=\sum_j\eta_j\,\lambda_{{\rm eff},j}$$

of the unmitigated bias. For λ = 1, 3, 5 the weights are $\eta=(15/8,\,-5/4,\,3/8)$. With the depolarizing row,
$\rho=1.875-1.25\cdot2.85+0.375\cdot4.70\approx0.07$, and the app's local-folding bias is −0.012 against global
folding's −0.005. With idle dephasing, $\rho\approx0.33$ and the local bias is −0.033 against −0.001. With
$\lambda_{\rm eff}=1$ everywhere, $\rho=1$: local ZNE simply returns the unmitigated value.

**Fit in the effective scale.** The ZNE parameter *noise scale λ of the fit* = *fault-rate* extrapolates in
$\lambda_{\rm eff}$ instead of the nominal λ. On the depolarizing QFT this cuts local folding's bias from −0.012 to
−0.003. If the folded circuits do not raise the fault rate at all, as with the over-rotation preset, the method
refuses to extrapolate and says why. That is better than a confident wrong answer.

**In the app:** *Mitigate & compare → Inside each method → ZNE* prints $\lambda_{\rm eff}$ for every folded circuit,
and whether the fit used it.

## 3. Fractional ZNE

Deep circuits decay toward the noise floor quickly, and at λ = 5 there may be little signal left. **Fractional
scales** between 1 and 3 (e.g. 1, 1.5, 2, 2.5, 3) sample $E(\lambda)$ while the signal is still large. Local folding
reaches them by folding a **subset** of the two-qubit gates once more:

$$\lambda=1+2\Bigl(k+\frac mn\Bigr)\qquad\text{for } m \text{ of the } n \text{ gates folded } k+1 \text{ times.}$$

Things to know:

* **λ is quantized.** A partial fold can only add whole gates. A GHZ-4 circuit has three CNOTs, so its reachable
  scales are 1, 1.67, 2.33, 3. If two requested scales fold to the same circuit, the app refuses and shows the scales
  they actually reach.
* **Which gates matter.** Gates differ in how strongly the observable feels them, so the choice of subset moves
  $E(\lambda)$ by more than shot noise. The *subset* parameter has four options:
  * `random`, a seeded random subset;
  * `nested`, where the gates folded at a smaller λ stay folded at every larger one;
  * `stratified`, spread evenly over the circuit;
  * `balanced`, whose mean calibrated error matches the circuit's, so that λ in gates ≈ λ in noise. It needs a
    device model.
* **Averaging subsets.** `fold_draws` K > 1 averages K independent subsets per fractional scale at the same total
  shots.
* **Use a low-degree fit.** Closely spaced scales give Richardson huge weights $\lvert\eta_j\rvert$. Fractional
  scales go with the linear or quadratic (`poly2`) rule.

**The price is the lever arm.** On the four-qubit QFT with *Hardware-realistic* noise, the quadratic fit through
λ = 1, 1.33, 1.67, 2 has the smallest bias of the ZNE contenders: −0.052, against −0.087 for global Richardson. Its
per-shot sd, however, is 9.2 against 1.5, so it needs about 37 times the shots for the same variance. Which one wins
is a budget question (previous chapter).

## 4. Coherent errors under folding

A stochastic error adds up in **probability**, so $E(\lambda)$ decays smoothly, roughly like $f^{\lambda}$. A coherent
error adds up in **amplitude**: three copies of a rotation by $\phi$ make a rotation by $3\phi$, and
$E(\lambda)\approx\cos(\lambda\phi)$ oscillates. A polynomial or exponential fit through such points can overshoot,
and on the four-qubit QFT with the over-rotation preset, global Richardson ZNE lands at 1.15 for an ideal value of 1.

Local folding makes the build-up explicit. A ZZ phase $\theta$ after every CZ commutes with the CZ, so
$G\,G^\dagger G$ carries the phase $3\theta$ in one block.

**Twirl the gates that carry the error.** Twirling turns a coherent error on a Clifford gate into a stochastic one, and
$E(\lambda)$ becomes smooth again. It must reach the gates that carry the error:

* On the QFT with the over-rotation preset, ZNE with *randomized compiling (all gates)*, chosen under *Build a custom
  contender*, lands at 1.000.
* Twirling only the two-qubit gates, as the *ZNE + RC + TREX* preset does, leaves this one-qubit error untouched.

**Errors that folding cannot see.** Some errors are invisible to folding altogether:

* A coherent error on a non-Clifford rotation such as $R_X(\theta)$ cannot be twirled away.
* An over-rotation proportional to the angle cancels exactly in $U^\dagger U$, so global folding does not amplify it.
  On the Ising template with the over-rotation preset, every ZNE variant returns the unmitigated value. A flat
  $E(\lambda)$ is a warning, not a success.

## 5. Folding as a measurement of the noise

Because the two foldings amplify different things, comparing them **on a device** says what kind of noise the device
has. For every circuit the decay rate $\kappa=-\mathrm d\ln F/\mathrm d\lambda$ is fitted over λ = 1, 3, 5, for the
measurement and for the model at the same shots. The **amplification anomaly** is $A=\kappa_{\rm obs}/\kappa_{\rm
pred}$, pooled per circuit family with a standard error.

| reading | condition (k = 3 standard errors) | what it points to |
|---|---|---|
| as modeled | $A_{\rm global}\approx A_{\rm local}\approx1$ | nothing unmodeled |
| gate-attached | both above 1, not distinguishable | errors that ride on the gates: a calibration that under-reports |
| time-attached | $A_{\rm global}-A_{\rm local}>k\,\mathrm{SE}$ | idle dephasing or drift (only global folding stretches time) |
| repetition-coherent | $A_{\rm local}-A_{\rm global}>k\,\mathrm{SE}$ | an error that builds up when the same gate repeats, such as a ZZ phase |
| over-modeled | both below 1 | the model is too pessimistic |

This **folding fingerprint** costs nothing extra: it reuses the ZNE circuits that mitigation runs anyway. On the
emulated v2 suite it read 1.31, 1.27 for gate errors × 1.6 (gate-attached), 1.12, 0.97 for $T_2$ = 30 µs
(time-attached), and 2.11, 2.75 for ZZ crosstalk of 0.05 rad (repetition-coherent).

## 6. In the scripts

```
python -m qemx.phase.run run manifests/direction_A_zne_folding_v1.yaml --out data/dirA_zne_folding_v1 --workers 8
python scripts/dirA_zne_folding.py data/dirA_zne_folding_v1 --out figs_zne_folding
```

`dirA_zne_folding.py` reports, per cell:

* the nominal and effective λ of every folded circuit;
* the depth and two-qubit counts that check the costs are equal;
* the residual ρ against the measured $b/b_{\rm RAW}$;
* the exact crossover budgets between global, local and fractional ZNE.

## Try it

Press **Global vs local folding · QFT** and **Run simulation**. Then open *Inside each method* for the two ZNE
contenders and compare their $\lambda_{\rm eff}$. Switch the sidebar's noise preset to *Markovian dephasing*, then
*Coherent over-rotation*, and rerun each time. The hands-on guide *Global versus local folding* has the steps. The
theory is in *Theory & sources* §4.6–4.8 and §21.4.
