"""Learning material: the fundamentals, algorithm and research-topic chapters (Markdown in docs/learn) and the
hands-on guides, each with a *setup* the Learn page applies in one click (``ui.state.apply_setup``: the Playground,
or an analysis page with its controls preset)."""
from __future__ import annotations

import os
import re

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "learn")

# the pages a setup can open, and how a setup button names them
PAGES = {"pages/1_Playground.py": "in the Playground", "pages/2_Cost_vs_Precision.py": "in Cost vs precision",
         "pages/4_Model_Mismatch.py": "in Model mismatch", "pages/5_Head_to_Head.py": "in Head-to-head",
         "pages/7_IBM_Quantum.py": "on the IBM Quantum page", "pages/8_Batch_Run.py": "the Batch run"}


def page_of(setup: dict) -> str:
    return setup.get("page", "pages/1_Playground.py")


def setup_label(setup: dict) -> str:
    """'Set up in the Playground', 'Set up in Cost vs precision', … (the default label of a setup button)."""
    return "Set up " + PAGES.get(page_of(setup), "in the Playground")


CHAPTERS = {
    "fundamentals": ("Fundamentals for QEM", "01_fundamentals.md"),
    "vqe": ("VQE", "02_vqe.md"),
    "qaoa": ("QAOA · MaxCut", "03_qaoa.md"),
    "shor": ("Shor (order finding)", "04_shor.md"),
    "hhl": ("HHL (linear systems)", "05_hhl.md"),
    "qec": ("QEC primitives", "06_qec.md"),
    "grover": ("Grover's search", "07_grover.md"),
    "hsp": ("Hidden subgroup problem", "08_hsp.md"),
}

# the Playground setups behind each algorithm chapter: (button label, setup)
ALGORITHM_SETUPS = {
    "vqe": [("VQE · hardware-efficient ansatz", dict(template=("hea_vqe", {}), noise="Hardware-realistic (illustrative)",
                                                    recipes=["Unmitigated", "ZNE · Richardson", "PEC · inverse",
                                                             "Readout · M3"], tab="Time travel"))],
    "qaoa": [("QAOA · MaxCut (ring of 4)", dict(template=("qaoa", {"n": 4, "graph": "ring", "p": 1}),
                                                noise="Hardware-realistic (illustrative)",
                                                recipes=["Unmitigated", "ZNE · Richardson", "Readout · M3",
                                                         "ZNE + RC + TREX"], tab="Time travel"))],
    "shor": [("Shor, a = 7 (precomputed powers)", dict(template=("shor", {"a": 7, "powers": "precomputed"}),
                                                        noise="Depolarizing",
                                                        recipes=["Unmitigated", "ZNE · Richardson", "Readout · M3"],
                                                        tab="Mitigate & compare")),
             ("Shor, a = 7 (repeated powers)", dict(template=("shor", {"a": 7, "powers": "repeated"}),
                                                    noise="Depolarizing", recipes=["Unmitigated", "ZNE · Richardson"],
                                                    tab="Time travel"))],
    "grover": [("Grover search, n = 3", dict(template=("grover_search", {"n": 3}),
                                             noise="Hardware-realistic (illustrative)",
                                             recipes=["Unmitigated", "ZNE · Richardson", "Readout · M3"],
                                             tab="Mitigate & compare")),
               ("Grover, 3 iterations (overshoot)", dict(template=("grover_search", {"n": 3, "iterations": "custom",
                                                                                    "k": 3}),
                                                         noise="Depolarizing", recipes=["Unmitigated", "ZNE · Richardson"],
                                                         tab="Mitigate & compare"))],
    "hsp": [("HSP over Z₂³, |H| = 4", dict(template=("hsp", {}), noise="Depolarizing",
                                         recipes=["Unmitigated", "Symmetry verification", "ZNE · Richardson"],
                                         tab="Mitigate & compare")),
            ("HSP over Z₁₆ (period finding)", dict(template=("hsp", {"group": "Z_N cyclic, N = 2^n (QFT)", "n": 4,
                                                                     "j": 2}),
                                                  noise="Depolarizing",
                                                  recipes=["Unmitigated", "ZNE · Richardson", "Readout · M3"],
                                                  tab="Mitigate & compare"))],
    "hhl": [("HHL, 2×2 system", dict(template=("hhl", {}), noise="Depolarizing",
                                     recipes=["Unmitigated", "Symmetry verification", "ZNE · Richardson"],
                                     tab="Mitigate & compare"))],
    "qec": [("3-qubit bit-flip code, X on q1", dict(template=("qec_3q", {"error": "one error on qubit k", "k": 1}),
                                                     noise="Depolarizing", recipes=["Unmitigated", "ZNE · Richardson",
                                                                                    "Symmetry verification"],
                                                     tab="Time travel")),
            ("Repetition memory, d = 5", dict(template=("repetition", {"d": 5}), noise="Depolarizing",
                                              recipes=["Unmitigated", "ZNE · Richardson", "Symmetry verification"],
                                              tab="Mitigate & compare")),
            ("Surface code d = 2, one plaquette", dict(template=("surface_d2", {
                "error": "bit-flip channel on every data qubit", "syndrome": "ancilla extraction (3 ancillas)"}),
                noise="Depolarizing", recipes=["Unmitigated", "Symmetry verification", "ZNE · Richardson"],
                tab="Mitigate & compare")),
            ("Five-qubit code [[5,1,3]]", dict(template=("five_qubit", {"error": "depolarizing channel on every qubit",
                                                                       "p": 0.03}),
                                               noise="Depolarizing", recipes=["Unmitigated", "ZNE · Richardson",
                                                                              "PEC · inverse"],
                                               tab="Mitigate & compare")),
            ("Steane [[7,1,3]]", dict(template=("steane", {}), noise="Depolarizing",
                                     recipes=["Unmitigated", "Symmetry verification", "ZNE · Richardson"],
                                     tab="Mitigate & compare"))],
}

SPECTATOR_RAMSEY = "h 0\nh 2\nbarrier\n" + "\n".join(["sx 1"] * 8) + "\nbarrier\nh 0\nh 2"

# ------------------------------------------------------------------ research topics (QEM Explorer's research methods)
TOPICS = {
    "budget": ("Comparing methods fairly", "09_equal_budget.md"),
    "folding": ("Inside ZNE: folding", "10_zne_folding.md"),
    "context": ("Context noise", "11_context_noise.md"),
    "divergence": ("Device versus model", "12_divergence.md"),
    "hardware": ("Running on IBM hardware", "13_hardware.md"),
}

# a miniature spectator-context probe on a line s0–c0–c1–s1 (qubits 0–3): eight CZ gates on the core between two
# Hadamard layers (an identity); the spectators idle in |0⟩ (zero) or are put into |+⟩ and back (plus). The observable
# is the spectator ⟨Z0⟩, ideal 1 in both modes.
_PROBE_CORE = "\n".join(["cz 1 2"] * 8)
PROBE_ZERO = "h 1\nh 2\nbarrier\n" + _PROBE_CORE + "\nbarrier\nh 1\nh 2"
PROBE_PLUS = "h 0\nh 1\nh 2\nh 3\nbarrier\n" + _PROBE_CORE + "\nbarrier\nh 0\nh 1\nh 2\nh 3"
PROBE_RECIPES = ["Unmitigated", "Dynamical decoupling (XY4)", "ZNE · Richardson"]

# custom contenders the research setups add (validated by apply_setup like the recipe builder's)
ZNE_LOCAL_FAULT = dict(estimator="zne", params=dict(folding="local", scales=[1.0, 3.0, 5.0], rule="richardson",
                                                    lam_est="fault-rate"))
ZNE_FRACTIONAL = dict(estimator="zne", params=dict(folding="local", subset="stratified",
                                                   scales=[1.0, 1.333, 1.667, 2.0], rule="poly2"))
ZNE_RC_ALL = dict(estimator="zne", twirl="all", params=dict(folding="global", scales=[1.0, 3.0, 5.0],
                                                            rule="richardson"))

BUDGET_RECIPES = ["Unmitigated", "ZNE · Richardson", "ZNE · exponential", "PEC · inverse", "CDR · OLS"]
_BUDGET = dict(template=("qft", {"n": 4}), noise="Hardware-realistic (illustrative)", recipes=BUDGET_RECIPES,
               widgets={"w_cvp_eps": 0.01, "w_hh_rc": BUDGET_RECIPES, "w_hh_param": "scale", "w_hh_min": 0.5,
                        "w_hh_max": 3.0, "w_hh_pts": 6})
BUDGET_SETUP = dict(_BUDGET, page="pages/2_Cost_vs_Precision.py", tabs={"cvp": "Your circuit"})
FOLDING_SETUP = dict(template=("qft", {"n": 4}), noise="Depolarizing",
                     recipes=["Unmitigated", "ZNE · Richardson", "ZNE · local folding"], custom=[ZNE_LOCAL_FAULT],
                     tab="Mitigate & compare")
CONTEXT_SETUP = dict(circuit=("Spectator probe (plus)", 4, PROBE_PLUS), observable=("z", 0),
                     noise="Crosstalk (spectator)", compile={"mode": "abstract"}, recipes=PROBE_RECIPES,
                     tab="Mitigate & compare")

TOPIC_SETUPS = {
    "budget": [("Cost vs precision · QFT under realistic noise", BUDGET_SETUP),
               ("Head-to-head · who wins where", dict(_BUDGET, page="pages/5_Head_to_Head.py"))],
    "folding": [("Global vs local folding · QFT", FOLDING_SETUP),
                ("Fractional ZNE · QFT", dict(template=("qft", {"n": 4}), noise="Hardware-realistic (illustrative)",
                                               recipes=["Unmitigated", "ZNE · Richardson"], custom=[ZNE_FRACTIONAL],
                                               tab="Mitigate & compare")),
                ("Coherent error: twirl first · QFT", dict(template=("qft", {"n": 4}), noise="Coherent over-rotation",
                                                          recipes=["Unmitigated", "ZNE · Richardson",
                                                                   "ZNE · local folding", "ZNE + RC + TREX"],
                                                          custom=[ZNE_RC_ALL], tab="Mitigate & compare"))],
    "context": [("Spectator probe · spectators in |0⟩", dict(CONTEXT_SETUP,
                                                            circuit=("Spectator probe (zero)", 4, PROBE_ZERO))),
                ("Spectator probe · spectators in |+⟩", CONTEXT_SETUP),
                ("Spectator Ramsey (driven neighbor)", dict(circuit=("Spectator Ramsey", 3, SPECTATOR_RAMSEY),
                                                            observable=("z", 0), noise="Crosstalk (spectator)",
                                                            compile={"mode": "abstract"},
                                                            recipes=["Unmitigated", "Dynamical decoupling (XY4)"],
                                                            tab="Time travel"))],
    "divergence": [("Model mismatch · QFT on fake_fez", dict(template=("qft", {"n": 4}), backend="fake_fez",
                                                             recipes=["Unmitigated", "PEC · inverse"],
                                                             page="pages/4_Model_Mismatch.py",
                                                             tabs={"mm": "Assumed vs true rate"},
                                                             widgets={"w_mm_rmin": 0.0, "w_mm_rmax": 3.0,
                                                                      "w_mm_pts": 7})),
                   ("Unmodeled coherent error · QFT", dict(template=("qft", {"n": 4}), noise="Depolarizing",
                                                           recipes=["Unmitigated", "PEC · inverse",
                                                                    "PEC · learned model + RC"],
                                                           page="pages/4_Model_Mismatch.py",
                                                           tabs={"mm": "Unmodeled noise"}))],
    "hardware": [("IBM Quantum · rehearse locally", dict(template=("ghz", {"n": 4}), backend="fake_fez",
                                                         recipes=["Unmitigated", "Readout · TREX", "ZNE · Richardson",
                                                                  "ZNE · local folding"],
                                                         page="pages/7_IBM_Quantum.py",
                                                         widgets={"w_hw_mode": "local", "w_hw_fake": "fake_fez"}))],
}

GUIDES = [
    dict(key="zne_first", title="Your first mitigation: ZNE on a GHZ state", level="beginner", minutes=10,
         goal="See noise bias an expectation value, then remove most of the bias by extrapolating to zero noise.",
         setup=dict(template=("ghz", {"n": 3}), noise="Depolarizing",
                    recipes=["Unmitigated", "ZNE · linear", "ZNE · Richardson", "ZNE · exponential"],
                    tab="Mitigate & compare"),
         steps=[
             "Press **Set up in the Playground**: a 3-qubit GHZ state $(|000\\rangle+|111\\rangle)/\\sqrt2$, depolarizing "
             "noise, and four contenders.",
             "Press **Run simulation** in the sidebar. In *Mitigate & compare* the unmitigated estimate of "
             "$\\langle Z_0Z_2\\rangle$ sits below the ideal value 1 — that gap is the **bias**.",
             "Open *Circuits*, choose *ZNE* and λ = 3: every gate $U$ became $U(U^\\dagger U)$. The ideal answer is the "
             "same, the noise is three times larger.",
             "Back in *Mitigate & compare*, open the ZNE fit: measured points at λ = 1, 3, 5 and the curve continued "
             "to λ = 0. Linear, Richardson and exponential fits disagree because $E(\\lambda)$ is curved.",
             "Compare the **sd** and **γ** columns: extrapolation magnifies shot noise; γ² more shots buy back the "
             "precision.",
         ],
         check=[("Why is the linear fit still biased?",
                 "Depolarizing noise makes $E(\\lambda)\\approx f^{\\lambda}$, an exponential. A straight line through "
                 "the measured points misses the curvature; Richardson (a polynomial through all points) or an "
                 "exponential fit follow it."),
                ("What is γ for ZNE?",
                 "The sum of the absolute extrapolation weights, $\\gamma=\\sum_j|\\eta_j|$. With $\\hat E=\\sum_j\\eta_j "
                 "E(\\lambda_j)$ the standard deviation grows by about γ, so γ² times more shots give the same "
                 "precision.")]),
    dict(key="pec_price", title="PEC removes the bias — and the price is γ²", level="intermediate", minutes=15,
         goal="Watch probabilistic error cancellation reach zero bias, and see why its shot cost explodes with size.",
         setup=dict(template=("ghz", {"n": 4}), noise="Depolarizing",
                    recipes=["Unmitigated", "ZNE · Richardson", "PEC · inverse"], tab="Mitigate & compare"),
         steps=[
             "Press **Set up in the Playground**, then **Run simulation**.",
             "In *Mitigate & compare*, PEC's bias is essentially zero — it inverts the (known) noise exactly — but "
             "its **sd** is the largest. Read **γ** in the table.",
             "Open *Circuits → PEC*: each run is a random circuit with Paulis inserted after the gates, weighted by "
             "signed quasi-probabilities (the *show an instance with insertions* toggle finds one with insertions).",
             "Open **Cost vs precision**: the RMSE-versus-shots chart shows where PEC overtakes the others — only "
             "after many shots.",
             "Double the two-qubit error rate p₂ in the sidebar's noise injector and run again: γ grows as "
             "$\\gamma=\\prod_i\\gamma_i$, exponentially in the number of noisy gates.",
         ],
         check=[("What does γ = 1.2 cost in shots?", "$\\gamma^2=1.44$: 44 % more shots for the same standard error."),
                ("What happens if the model PEC inverts is wrong?",
                 "The bias comes back (in proportion to the mismatch). The **Model mismatch** page sweeps the assumed "
                 "error rate against the true one.")]),
    dict(key="dd_crosstalk", title="Dynamical decoupling against crosstalk", level="intermediate", minutes=10,
         goal="Drive one qubit and watch its idle neighbor's phase drift; refocus it with DD pulses.",
         setup=dict(circuit=("Spectator Ramsey", 3, SPECTATOR_RAMSEY), observable=("z", 0),
                    noise="Crosstalk (spectator)", compile={"mode": "abstract"},
                    recipes=["Unmitigated", "Dynamical decoupling (XY4)", "ZNE · Richardson",
                             "Pauli twirling (RC) only"], tab="Time travel"),
         steps=[
             "Press **Set up in the Playground**: q0 and q2 start in $|+\\rangle$, q1 (their neighbor) is driven with "
             "eight √X pulses, then q0 is measured in the X basis ($H$ then $Z$). Ideally $\\langle Z_0\\rangle=1$.",
             "Press **Run simulation**. In *Time travel*, scrub through the √X steps: q0's Bloch vector turns around "
             "the z axis a little with every pulse on q1 — a spectator phase kick $R_Z(\\delta)$, the same every time.",
             "In *Mitigate & compare*: DD recovers most of the loss, twirling does nothing (it randomizes two-qubit "
             "gates, not these one-qubit kicks), and ZNE **overshoots** — a coherent phase makes $E(\\lambda)=\\cos(\\lambda\\phi)$, "
             "which a polynomial extrapolation does not follow.",
             "Open *Circuits → DD*: X and Y pulses fill q0's idle window. Each pulse flips the sign of the kicks that "
             "follow, so they cancel in pairs.",
             "Keep *Gate set = Abstract* here: ISA compilation would merge the eight √X into nothing (they multiply to "
             "the identity), and there would be no drive left to cause crosstalk.",
         ],
         check=[("Why does DD cancel the kicks?",
                 "$X R_Z(\\delta) X=R_Z(-\\delta)$: between two pulses the accumulated phase changes sign, so "
                 "$R_Z(\\delta)\\,X\\,R_Z(\\delta)\\,X=I$."),
                ("Would DD help against depolarizing noise?",
                 "No: random errors have no fixed sign to refocus, and the pulses add gate noise of their own.")]),
    dict(key="readout", title="Readout errors: tensored inverse, M3 and TREX", level="beginner", minutes=8,
         goal="Separate measurement errors from gate errors and remove them three ways.",
         setup=dict(template=("bv", {"n": 4, "s": 11}), noise="Readout only",
                    recipes=["Unmitigated", "Readout · tensored", "Readout · M3", "Readout · TREX"],
                    tab="Mitigate & compare"),
         steps=[
             "Press **Set up in the Playground**: Bernstein–Vazirani with secret $s=1011$ (four input qubits and one "
             "oracle qubit) and *only* readout noise: "
             "every gate is perfect, the state before measurement is exact.",
             "Press **Run simulation**. The unmitigated probability of reading $s$ is below 1 — only because bits "
             "flip at readout. $P(0\\mid1)>P(1\\mid0)$ in this preset: excited states decay while being measured.",
             "The tensored inverse and M3 invert the calibrated assignment matrix; TREX flips bits at random before "
             "measurement and back afterwards, which makes the error symmetric and easy to divide out. All three "
             "remove the bias; compare their **training/cal. shots** and **sd**.",
             "Switch the noise to *Hardware-realistic* and run again: readout mitigation now removes only part of "
             "the bias — the rest comes from gates.",
         ],
         check=[("Why is TREX model-free?",
                 "Random bit flips before measurement (undone classically) turn any assignment error into a symmetric "
                 "one that only rescales $\\langle Z\\rangle$; one calibration of that factor removes it.")]),
    dict(key="routing", title="Routing costs: topology, SWAPs and the best qubits", level="intermediate", minutes=15,
         goal="See why a real coupling map adds SWAPs, and pick the physical qubits with the lowest error.",
         setup=dict(template=("qft", {"n": 4}), backend="fake_fez", noise="Hardware-realistic (illustrative)",
                    recipes=["Unmitigated", "ZNE · Richardson"],
                    tab="Topology & layout"),
         steps=[
             "Press **Set up in the Playground**: a 4-qubit QFT compiled for the simulated IBM Fez (156-qubit Heron, "
             "heavy-hex lattice).",
             "In *Topology & layout*: rings mark where q0…q3 start, amber couplers are where the router inserted "
             "SWAPs; the table says which logical gate each SWAP made possible. QFT couples every pair, the lattice "
             "gives each qubit at most three neighbors.",
             "Note the **expected success F**, then press **Find the lowest-error physical qubits**. The ranked "
             "table lists equivalent placements; highlight a few, then **Use layout #1**. F rises.",
             "On *Build*, switch on **Show the compiled (physical) circuit**: each logical gate sits above its native "
             "gates (H → RZ · √X · RZ, CP → CZ and rotations), SWAPs in amber frames.",
             "Run the simulation before and after choosing the layout: the unmitigated estimate follows F.",
         ],
         check=[("What does one SWAP cost?", "Three two-qubit gates: $\\mathrm{SWAP}=\\mathrm{CX}_{ab}\\mathrm{CX}_{ba}"
                                           "\\mathrm{CX}_{ab}$, so its fidelity cost is about $(1-e_2)^3$."),
                ("Why is RZ free?", "It is a frame change in the control electronics: no pulse, no duration, no "
                                    "error.")]),
    dict(key="vqe_landscape", title="Variational circuits under noise: a flatter landscape", level="intermediate",
         minutes=15, goal="Move QAOA's angles live and map how noise reshapes the optimization landscape.",
         setup=dict(template=("qaoa", {"n": 4, "graph": "ring", "p": 1}), noise="Hardware-realistic (illustrative)",
                    recipes=["Unmitigated", "ZNE · Richardson", "Readout · M3"], tab="Time travel"),
         steps=[
             "Press **Set up in the Playground**. In *Time travel*, the **Live parameters θ** card has sliders for γ₁ "
             "and β₁ (no need to press Run: live mode simulates on every move).",
             "Move γ₁: the Bloch spheres, the density matrix and ⟨C⟩/C_max follow. The noisy value lags the ideal one.",
             "Open **Energy landscape**, choose *2D: γ × β*, press **Scan**: two maps side by side. The noisy map is the "
             "ideal one squeezed toward 0.5 (random guessing) — the maximum is lower but in the same place.",
             "Press **Move the sliders to the noisy optimum**, then **Use these angles in the circuit**, and compare "
             "methods in *Mitigate & compare*.",
         ],
         check=[("Why do the optima coincide?",
                 "Depolarizing noise acts like $E_{\\rm noisy}\\approx fE+(1-f)E_{\\rm mixed}$, a rescaling that does "
                 "not move the maximum. Coherent errors would move it.")]),
    dict(key="batch", title="Which method wins where? A batch comparison", level="advanced", minutes=20,
         goal="Run several algorithms × noise models × noise scales × methods at once and read the winners.",
         setup=dict(page="pages/8_Batch_Run.py",
                    batch=dict(algos=["ghz", "qaoa", "qft"], methods=["Unmitigated", "ZNE · Richardson",
                                                                     "PEC · inverse", "ZNE + RC + TREX"],
                               noises=["Depolarizing", "Coherent over-rotation", "Crosstalk (spectator)"],
                               scales=[0.5, 1.0, 2.0], sims=["numpy"])),
         steps=[
             "Press **Set up the Batch run**: 3 algorithms × 3 noise models × 3 scales = 27 scenarios, 4 methods each.",
             "Press **Run all**. The heatmap colors every scenario × method by RMSE at the chosen shot budget; ★ "
             "marks the best method in each row.",
             "Read *Best method per scenario* and the *wins* chart: the winner changes with the noise type — PEC "
             "assumes depolarizing noise, so coherent over-rotation hurts it; twirling helps ZNE there.",
             "In *Error against the noise scale*, flat curves are methods that keep up as the noise grows.",
             "Change the **shot budget N** without rerunning: RMSE at that total budget is recomputed (training and "
             "calibration shots included), and the winners shift toward low-bias methods as N grows. **Download all (ZIP)** saves every table and figure.",
         ],
         check=[("Why can the winner change with N?",
                 "At small N the variance term v/N dominates and cheap methods win; at large N only the bias b "
                 "remains and low-bias (high-γ) methods win.")]),
    # ---------------------------------------------------------------- research guides (Research topics tab)
    dict(key="budget_regimes", track="research", title="Who wins depends on the budget", level="intermediate", minutes=15,
         goal="Read crossover budgets and unreachable targets, then map the winner over noise scale × budget.",
         setup=BUDGET_SETUP,
         steps=[
             "Press **Set up in Cost vs precision**: a 4-qubit QFT ($P(0101)$, ideal 1) with *Hardware-realistic* "
             "noise and five contenders. Press **Run simulation** in the sidebar.",
             "Read *RMSE against budget*: every curve falls like $1/\\sqrt B$ and then flattens at its floor "
             "$\\sqrt{b^2+u}$. The unmitigated curve flattens at its bias, 0.18.",
             "In the table, *beats unmitigated above B* is the crossover budget $B^*$: PEC after about 9 shots, "
             "exponential ZNE about 37, Richardson ZNE about 85, and CDR only above about 16,400, because its 16 "
             "training circuits × 1,024 shots come first.",
             "*Budget for ε* is ∞ in every row at ε = 0.01: every floor lies above 0.01. Set ε to 0.05: only CDR "
             "reaches it, at about $1.7\\times10^4$ shots. Winning at a large budget and reaching a target are "
             "different questions.",
             "Open **Head-to-head**: the same contenders and a sweep of the noise scale from 0.5 to 3 are already "
             "chosen. Press **Run sweep**. In *Who wins where*, PEC (or the unmitigated estimate) wins the smallest "
             "budgets, exponential ZNE the middle, and CDR everything above about $3\\times10^4$ shots.",
             "Switch the sidebar's noise preset to *Depolarizing* and run the sweep again. PEC's model is now exact, "
             "and on this small circuit it wins almost everywhere: the structure of the noise, not its strength, "
             "redrew the map.",
         ],
         check=[("What is the crossover budget between two methods?",
                 "With floors $F_i=b_i^2+u_i$ and variance coefficients $V_i$, "
                 "$\\mathrm{RMSE}_i\\approx\\sqrt{F_i+V_i/B}$, so method $j$ overtakes method $i$ at "
                 "$B^*=(V_j-V_i)/(F_i-F_j)$."),
                ("Why can CDR have the smallest bias and still lose at 10,000 shots?",
                 "It cannot run below its training cost (16,384 shots here), and the shot noise of its training data, "
                 "$u$, is part of its floor.")]),
    dict(key="folding", track="research", title="Global versus local folding", level="advanced", minutes=15,
         goal="See local folding amplify less noise than its nominal λ, measure the effective λ, and fit with it.",
         setup=FOLDING_SETUP,
         steps=[
             "Press **Set up in the Playground**: a 4-qubit QFT with depolarizing noise and four contenders, namely "
             "unmitigated, global and local Richardson ZNE (λ = 1, 3, 5), and local ZNE fitted in the effective "
             "scale. Press **Run simulation**.",
             "In *Mitigate & compare*, global ZNE leaves a bias of −0.005 and local ZNE −0.012, although both run the "
             "same two-qubit gates.",
             "Open *Inside each method* for local ZNE: the effective noise scale is $\\lambda_{\\rm eff}$ = 1, 2.85, "
             "4.70. Local folding does not repeat the one-qubit gates, so the noise grows more slowly than the gate "
             "count says.",
             "The contender with *λ from fault rate* extrapolates in $\\lambda_{\\rm eff}$ instead, and its bias "
             "drops to −0.003.",
             "Switch the noise preset to *Markovian dephasing* and run again. Local folding's "
             "$\\lambda_{\\rm eff}$ falls to 2.33, 3.67, and its bias grows to −0.033 against global folding's −0.001. "
             "Idle time is noise that only global folding stretches.",
             "Switch to *Coherent over-rotation* and run again. Global ZNE overshoots to 1.15, and local ZNE returns "
             "the unmitigated value ($\\lambda_{\\rm eff}$ = 1, 1, 1: this preset's error sits on one-qubit gates). "
             "The fault-rate contender refuses to extrapolate and says why.",
         ],
         check=[("How much bias does local folding leave?",
                 "To first order the fraction $\\rho=\\sum_j\\eta_j\\lambda_{{\\rm eff},j}$ of the unmitigated bias. "
                 "With $\\eta=(15/8,-5/4,3/8)$ and $\\lambda_{\\rm eff}$ = 1, 2.85, 4.70, $\\rho\\approx0.07$."),
                ("Why does the difference between the foldings say something about a device?",
                 "Global folding stretches idle time; local folding repeats the same gate back to back. An excess that "
                 "only global folding amplifies is attached to time (dephasing, drift). One that local folding "
                 "amplifies more is a coherent error that builds up. This is the folding fingerprint.")]),
    dict(key="context_probe", track="research", title="Spectators: find the context noise", level="advanced", minutes=15,
         goal="Use a miniature spectator probe to see errors a calibration cannot, and name the mechanism by how it "
              "responds to DD and ZNE.",
         setup=CONTEXT_SETUP,
         steps=[
             "Press **Set up in the Playground**: four qubits on a line. The core (q1, q2) runs eight CZ gates between "
             "Hadamard layers, and the spectators q0 and q3 are put into $|+\\rangle$ and back. The observable is the "
             "spectator $\\langle Z_0\\rangle$, ideal 1. The noise is *Crosstalk (spectator)*. Press **Run "
             "simulation**.",
             "The unmitigated $\\langle Z_0\\rangle$ is about 0.84: every pulse on q1 kicks its idle neighbor q0. In "
             "*Time travel*, scrub through the CZ steps and watch q0's Bloch vector turn about z.",
             "Switch the noise preset to *Depolarizing* and run again: 0.996. That is what a component model "
             "predicts. The difference, about 0.15, is the excess contrast: context noise. The probe with spectators "
             "in $|0\\rangle$ (*Research topics → Context noise*) gives 1 under every preset, because an idle "
             "$|0\\rangle$ spectator is blind to phase errors.",
             "Compare the remedies under three presets, rerunning each time. *Crosstalk*: DD 0.997, ZNE 1.47. "
             "*Random field (quasi-static)*: unmitigated 0.49, DD 1.000. *Markovian dephasing*: unmitigated 0.88, DD "
             "0.86, ZNE 0.996.",
             "Read the pattern. DD removes errors with a fixed sign (kicks, a slow field). White dephasing has none, so "
             "DD only adds pulses. Global ZNE stretches the idle time and extrapolates it away, but overshoots on "
             "coherent kicks.",
         ],
         check=[("Why put the spectators in |+⟩?",
                 "Phase errors (Z kicks, ZZ phases, dephasing) leave $|0\\rangle$ unchanged. In superposition the same "
                 "errors rotate or shrink the Bloch vector, so they become measurable."),
                ("What does the study's `driven` mode add on hardware?",
                 "Spectators that run their own gates next to the core: gate errors near the core and simultaneous-drive "
                 "crosstalk. With all three modes, superposed only means idle dephasing, driven only gate error, and "
                 "both spectator crosstalk.")]),
    dict(key="unmodeled", track="research", title="When the model misses an error", level="intermediate", minutes=10,
         goal="Watch PEC with the right depolarizing rates fail against a coherent error it does not model, and a "
              "learned model with randomized compiling recover.",
         setup=TOPIC_SETUPS["divergence"][1][1],
         steps=[
             "Press **Set up in Model mismatch**: a 4-qubit QFT with depolarizing noise ($p_1=0.002$, $p_2=0.02$) "
             "and the *Unmodeled noise* tab.",
             "Press **Run the coherent-error sweep**. PEC is told the exact depolarizing rates, but the truth also "
             "over-rotates every pulsed one-qubit gate by a fraction ε.",
             "PEC's bias grows roughly like ε²: −0.008, −0.031, −0.069, −0.119 for ε = 0.02, 0.04, 0.06, 0.08.",
             "*PEC · learned model + RC* stays unbiased. Randomized compiling turns the over-rotation into Pauli "
             "noise, and the learned model includes it. Its γ grows from 1.38 to 1.78: the model is right, and the "
             "price is honest.",
             "This is the divergence study in one chart: a model built from component error rates is blind to how "
             "the error is structured. *Research topics → Device versus model* measures that blindness on a device.",
         ],
         check=[("Why does the bias grow like ε² rather than ε?",
                 "A coherent rotation by θ has fault probability $\\sin^2(\\theta/2)\\approx\\theta^2/4$. Its first "
                 "effect on a probability is second order in the angle."),
                ("Would twirling alone have fixed PEC here?",
                 "No. Twirling turns the error into Pauli noise, but PEC must also know that noise. The learned model "
                 "measures the twirled channels; the depolarizing model with the old rates still misses them.")]),
]


GUIDE_TRACKS = {"basics": "Mitigation basics", "research": "Research methods"}


def guide_track(g: dict) -> str:
    return g.get("track", "basics")


def chapter(key: str) -> str:
    """The Markdown of a fundamentals, algorithm or research-topic chapter."""
    title, fn = CHAPTERS[key] if key in CHAPTERS else TOPICS[key]
    with open(os.path.join(DOCS, fn), encoding="utf-8") as f:
        return f.read()


def guides_markdown() -> str:
    """All guides as one Markdown document (for the PDF manual)."""
    out = ["# Hands-on guides", ""]
    for g in GUIDES:
        out += ["## %s" % g["title"], "", "*%s · about %d minutes.* %s" % (g["level"].capitalize(), g["minutes"],
                                                                          g["goal"]), ""]
        out += ["%d. %s" % (i + 1, re.sub(r"\*\*Set up (in |on )?(the )?[A-Za-z -]+\*\*", "**Set up**", s))
                for i, s in enumerate(g["steps"])]
        out.append("")
        for q, a in g["check"]:
            out += ["**Check:** %s  " % q, "*%s*" % a, ""]
    return "\n".join(out)
