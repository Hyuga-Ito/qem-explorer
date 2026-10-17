# Running on IBM hardware within ten minutes

Everything in the Research topics can run on a fake device, and none of it needs a real one to be understood. The
final test of a thesis built on QEM Explorer, though, is a real device. This chapter plans that run: how to fit it into
a ten-minute QPU allowance, how to rehearse it, what to fix before the day, and how to keep the API token safe.

## 1. Two routes to hardware

| route | for | what it does |
|---|---|---|
| the app's **Run on IBM Quantum** page | a few contenders on one circuit | records the circuits each estimator asks for, gives every contender the same shot budget split as in simulation, runs them with the Sampler primitive, and replays the same estimator code on the measured counts. The results page is the Playground's |
| the study's **command line** (`dirA_divergence.py`) | a whole manifest of benchmarks | plans every circuit on its chains, submits everything as one Batch, collects counts together with the calibration at execution time, and analyzes them like any emulated run |

Both have a **local mode** that runs the identical pipeline on a fake backend:

* in the app, *Where = local*;
* on the command line, `run-local`, or `emulate` with planted causes.

## 2. Budgeting QPU time

QPU time is close to proportional to shots: $t\approx B\,t_{\rm shot}$, where $t_{\rm shot}$ includes reset and
readout. The study's estimate for ibm_fez is about **0.57 ms per shot**, so ten minutes hold about $10^6$ shots before
any overhead. The app's sidebar uses 250 µs per shot by default. Set it to the device's value before reading any QPU
time in *Cost vs precision*.

The `direction_A_divergence_v2_qpu10` manifest is the full v2 design trimmed to one window:

| | v2 | qpu10 |
|---|---|---|
| benchmarks × layout variants | 52 × 3 | 40 × 3 |
| methods | RAW, RO, ZNE global, ZNE local, fractional ZNE | RAW, RO, ZNE global, ZNE local (the fingerprint needs only the two foldings) |
| spectator probes | core depths 2, 4, 8, 16; two seeds | core depths 4 and 16; two seeds |
| shots per benchmark and method | 9,000 (3,000 per variant) | 4,500 (1,500 per variant) |
| total | 2,214,000 shots in 2,868 circuits: about 21 minutes at 0.57 ms per shot | **720,000 shots in 1,680 circuits**: about 410 s at 0.57 ms per shot, about 530 s with a 1.3× margin |

**Does the trimmed design still work?** The same emulation protocol as v2 answers it. Each cause is planted on fake_fez
and all four analyses go through the recovery matrix:

| planted | ladder | budget | fingerprint $(A_g,A_l)$ | probes | ε ratio |
|---|---|---|---|---|---|
| none | as modeled | — | 0.97, 0.96: as modeled | none | 1.00 |
| gate errors × 1.6 | rescaled calibration × 1.45 | rescaled calibration 74 % | 1.42, 1.43: gate-attached | gate error (driven z = 13.3) | 1.61 |
| $T_2$ = 30 µs | context + non-linear | idle time 100 % | 1.38, 0.91: time-attached | idle dephasing (plus z = 21.3) | 1.30 |
| ZZ 0.05 rad | context + non-linear | parallel 33 %, calibration 31 %, active spectators 22 % | 1.94, 2.56: repetition-coherent | spectator crosstalk (z = 4.4, 12.9) | 1.70 |

Every cause is still named correctly at 720,000 shots. No position-dependent cause was planted in this check, so the
layout column was tested only for false alarms (none). The pilot's hot-spot emulation tests it the other way.

## 3. The token is a password

* **In the app.** The token goes into a password field inside a form. It is read once, in the submit callback, to
  open the connection, and the field is cleared before the page reruns. It never enters the app's model, the
  exported script or the state file.
* **On the command line.** The token comes only from:
  * the environment variable `QISKIT_IBM_TOKEN`;
  * a saved account;
  * a hidden prompt (`--ask-token`).

  It is never a command-line argument (shell history and process lists would keep it), never written to the run
  directory, and never printed. Connection errors show it masked.
* **In notebooks.** A token pasted into a cell ends up in the file. Clear it before the notebook goes into a
  repository or an appendix, and use one of the other two ways. Rotate a token that was ever pasted anywhere.

## 4. The day of the run

1. **Rehearse.** The emulated recovery matrix must name every planted cause with the exact manifest you will run (the
   table above).
2. **Pre-register.** Write down the expected readings and their nulls before the run, so that a result cannot be
   reinterpreted afterwards.
3. **Plan on today's calibration.** `plan MANIFEST --out data/div_ibm --backend ibm_fez`. Chains are chosen from the
   day's calibration, and `plan` prints the exact totals.
4. **Check the circuits.** Every circuit must be ISA for the backend, and every fold must survive compilation. The
   plan stores a SHA-256 digest of every circuit.
5. **Submit once.** `run-ibm data/div_ibm` sends everything in one Batch when the account allows it, and as single
   jobs otherwise. Dynamical decoupling and service-side twirling are switched **off**: the analysis measures the idle
   windows itself, and the variants are planned explicitly. After a partial submission, `--resume` sends only what is
   missing.
6. **Collect with the calibration.** `collect-ibm data/div_ibm --wait` stores the counts and the calibration at
   execution time when the service keeps the history.
7. **Analyze like an emulation.** `analyze data/div_ibm --tag ibm --out figs_ibm`, then add the device's row to the
   diagnostic matrix with `recovery`.
8. **A second day, if possible.** Position effects that move between days are drift. A second run turns the layout
   variants into a drift test.

## 5. What a fake device can and cannot rehearse

| rehearsed by a fake device | not rehearsed |
|---|---|
| the compilation, the chains, the ISA check, the shot split | queueing and wall-clock time (only QPU time counts against the window) |
| every estimator and every figure | calibration changes between planning and running |
| the planted-cause recovery | the device's actual context noise: that is the measurement |

**In the app:** open *Run on IBM Quantum* with *Where = local* and a fake backend. Every step of the pipeline runs
except the network call, so a mistake costs nothing.

## Try it

Press **IBM Quantum · rehearse locally**: a GHZ circuit on fake_fez with raw, TREX and both ZNE foldings. Run it in
local mode and compare the results with the Playground's simulation of the same state. The research plan
(`docs/v14/research_plan_divergence.md`) and *Theory & sources* §15 and §21.6 have the details.
