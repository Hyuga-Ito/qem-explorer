# Quantum error correction primitives

Error mitigation removes the *effect* of noise on averages; **error correction** removes the errors themselves,
during the computation, by encoding one logical qubit in several physical ones. The app's QEC templates show the
basic mechanisms at a scale that can be simulated exactly.

## Stabilizers and syndromes

A code is the joint $+1$ eigenspace of commuting Pauli operators, the **stabilizers**. An error anticommutes with
some of them; measuring the stabilizers (with ancillas, without touching the encoded information) gives a
**syndrome** that says which error happened, and a correction undoes it.

## The three-qubit bit-flip code

$$|0_L\rangle=|000\rangle,\quad|1_L\rangle=|111\rangle,\qquad \text{stabilizers } Z_0Z_1,\ Z_1Z_2 .$$

| error | $Z_0Z_1$ | $Z_1Z_2$ | correction |
|---|---|---|---|
| none | $+1$ | $+1$ | — |
| $X_0$ | $-1$ | $+1$ | $X_0$ |
| $X_1$ | $-1$ | $-1$ | $X_1$ |
| $X_2$ | $+1$ | $-1$ | $X_2$ |

Any single bit flip is corrected. If each qubit flips independently with probability $p$, the logical error
probability is $p_L=3p^2-2p^3$, below $p$ whenever $p<\tfrac12$. The **phase-flip code** is the same code in the
Hadamard basis ($|{+}{+}{+}\rangle$, $|{-}{-}{-}\rangle$) and corrects one $Z$ error.

**In the app** (*3-qubit bit-/phase-flip code*): the logical state $R_Y(\theta)|0\rangle$ is encoded on q0–q2;
ancillas q3, q4 measure $Z_0Z_1$ and $Z_1Z_2$; Toffoli gates apply the correction coherently; decoding returns the
state to q0, so $\langle Z_0\rangle=\cos\theta$. Inject *one error on qubit k* or a *channel on every data qubit*, and
switch the mode between *syndrome + correction*, *syndrome only* and *majority vote readout*.

<!-- fig:qec3 -->

## Repetition-code memory

$d$ data qubits (odd), $d-1$ parity checks per round, and a **majority vote** at the end. A bit-flip channel with
probability $p$ acts before every round and once more before readout, so after $r$ rounds each bit is flipped with
$p_{\rm eff}=\tfrac12[1-(1-2p)^{r+1}]$. The vote fails when more than half the bits flipped:

$$p_L=\sum_{j>d/2}\binom{d}{j}\,p_{\rm eff}^j(1-p_{\rm eff})^{d-j}\ \sim\ p_{\rm eff}^{(d+1)/2} .$$

Below a threshold, a larger $d$ suppresses errors exponentially — the principle behind every fault-tolerant
architecture. Above it, more qubits make things worse.

<!-- fig:repetition -->

## The Steane [[7,1,3]] code

A CSS code built from the classical Hamming [7,4,3] code: $X$- and $Z$-type stabilizers on the rows of its check
matrix. It encodes one qubit in seven and corrects **any** single-qubit error ($X$, $Y$ or $Z$). Logical $Z$ is
transversal ($Z^{\otimes7}$): measure all seven qubits, compute the Hamming syndrome of the bit string, flip the
indicated bit, and take the parity.

<!-- fig:steane -->

## The five-qubit code [[5,1,3]]

The smallest code that corrects **any** single-qubit error (Laflamme, Miquel, Paz and Zurek, 1996; Bennett et al.,
1996). Its four stabilizers are cyclic shifts of one Pauli string,

$$S_1=XZZXI,\quad S_2=IXZZX,\quad S_3=XIXZZ,\quad S_4=ZXIXZ,\qquad Z_L=Z^{\otimes5},\ X_L=X^{\otimes5}.$$

There are $2^4=16$ syndromes and exactly 16 correctable events (no error, or $X$, $Y$, $Z$ on one of five qubits):
the code is *perfect* — it saturates the quantum Hamming bound $2^{n-k}\ge 1+3n$.

**In the app** (*Five-qubit code [[5,1,3]]*): a Clifford encoder $U$ maps $Z_0\to Z_L$, $X_0\to X_L$ and $Z_j\to S_j$.
After the error, $U^\dagger$ turns any single-qubit error $E$ into (a logical Pauli on q0) ⊗ (the syndrome written on
q1–q4), so reading all five qubits gives the syndrome *and* the logical bit. The decoded observable looks the
syndrome up in the 16-entry table and flips the logical bit when the error acted as $X$ or $Y$ on it. Any single $X$,
$Y$ or $Z$ error leaves $\langle Z_L\rangle=\cos\theta$; the *raw* readout shows the damage.

<!-- fig:five -->

## The smallest surface code: one plaquette ([[4,1,2]])

Four data qubits on a 2 × 2 patch (q0 q1 / q2 q3), one weight-4 **X plaquette** and two weight-2 **Z boundary checks**:

$$X_0X_1X_2X_3,\quad Z_0Z_1,\quad Z_2Z_3;\qquad Z_L=Z_0Z_2,\quad X_L=X_0X_1,$$

$$|0_L\rangle=\tfrac{1}{\sqrt2}(|0000\rangle+|1111\rangle),\qquad |1_L\rangle=\tfrac{1}{\sqrt2}(|1100\rangle+|0011\rangle).$$

With distance 2 every single-qubit error is **detected** but none can be corrected (two different errors give the
same syndrome and differ by a logical operator). The decoder is post-selection: symmetry verification on both Z
checks $Z_0Z_1$ and $Z_2Z_3$ (transversal readout; their product $Z_0Z_1Z_2Z_3$ alone would miss $X_0X_3$ and
$X_1X_2$, which flip $Z_L$) or on the two Z-check ancillas. With *ancilla extraction* (7 qubits), q4
measures the plaquette — a $Z$ error on any data qubit flips it — and q5, q6 the Z checks. This is the unit cell of
the rotated surface code: larger patches tile the same weight-4 plaquettes.

<!-- fig:surface -->

## Correction versus mitigation

The syndrome circuits add many gates, and those gates are noisy: whether encoding helps depends on the physical
error rate (the *break-even* question). Mitigation and correction also meet: **symmetry verification** uses a
stabilizer as a symmetry and discards shots that violate it — error *detection* with post-selection instead of
correction. The templates set that symmetry, so you can compare detection (post-selection), correction and
mitigation on the same circuit.

## Try it

1. Bit-flip code, *one error on qubit 1*: with correction $\langle Z_0\rangle$ stays at $\cos\theta$; with *syndrome only*
   it does not.
2. Repetition memory: raise $d$ from 3 to 5 at $p=0.05$ and watch $p_L$ in the notes fall.
3. Steane: switch the readout between *decoded* and *raw parity* with an $X$ error on one qubit.
4. Five-qubit code: inject a $Y$ error on any qubit — decoded $\langle Z_L\rangle$ does not move; the *Analysis* view
   shows the syndrome it produced.
5. Surface code d = 2 with a bit-flip channel: compare *Unmitigated* and *Symmetry verification*; the syndrome
   histogram shows how many shots post-selection keeps.
