# Variational quantum eigensolver (VQE)

**The problem.** Find the lowest energy $E_0$ of a Hamiltonian $H=\sum_k c_kP_k$ — a molecule, a spin model, an
optimization cost. Exact diagonalization needs $2^n\times2^n$ matrices; a quantum computer can hold the state
instead.

**The idea.** For any state, the energy is an upper bound on the ground energy (the *variational principle*):

$$E(\vec\theta)=\langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle\;\ge\;E_0 .$$

VQE (Peruzzo et al., 2014) prepares a parameterized state $|\psi(\vec\theta)\rangle=U(\vec\theta)|0\rangle^{\otimes n}$ on
the quantum computer, estimates $E(\vec\theta)$ from measurements, and lets a classical optimizer change
$\vec\theta$ to lower it. The quantum part is short (one circuit per energy evaluation), which is why VQE was the
flagship algorithm of the noisy era.

## The loop

1. Choose an **ansatz** $U(\vec\theta)$. The *hardware-efficient ansatz* (Kandala et al., 2017) uses what the device
   does well: layers of one-qubit rotations $R_Y(\theta)R_Z(\theta')$ on every qubit, separated by CX entanglers.
   With $L$ entangling layers it has $2n(L+1)$ angles.
2. **Measure** $E(\vec\theta)$: group the Pauli terms of $H$ into qubit-wise commuting sets, run one circuit per set,
   combine $E=\sum_k c_k\langle P_k\rangle$.
3. **Update** $\vec\theta$ with an optimizer (COBYLA, SPSA, L-BFGS-B); repeat until converged.

<!-- fig:vqe -->

## In the app

The template *VQE · hardware-efficient ansatz* targets the antiferromagnetic chain

$$H=\frac{1}{n-1}\sum_{i=0}^{n-2}Z_iZ_{i+1},\qquad E_0=-1 ,$$

whose ground states are the Néel states $|0101\ldots\rangle$ and $|1010\ldots\rangle$. When the template loads, the
angles are optimized on a noiseless statevector (a stand-in for the loop above), so the ideal energy is $-1$.
Choose *random* angles to start somewhere else. You can also type any Hamiltonian under *Observable →
Hamiltonian* (the H₂ molecule in a two-qubit encoding, the Heisenberg or transverse-field Ising chain are
examples).

In *Time travel*, the **Live parameters θ** sliders change the angles and the Bloch spheres, the state and
$\langle H\rangle$ follow at once; *Energy landscape* scans one angle with and without noise.

## What noise does

Depolarizing noise mixes the state with the identity, so the energy is pulled toward the average eigenvalue
$\mathrm{Tr}(H)/2^n$ (here 0):

$$E_{\rm noisy}(\vec\theta)\approx f\,E(\vec\theta)+(1-f)\,\frac{\mathrm{Tr}H}{2^n},\qquad f\approx(1-p)^{\#\text{gates}} .$$

The landscape keeps its shape but flattens: the minimum is shallower and **gradients shrink by $f$**. With
depth, this becomes a *noise-induced barren plateau* (Wang et al., 2021): the optimizer sees an exponentially flat
landscape. Coherent errors instead shift the minimum to wrong angles. Readout errors bias every $\langle Z_iZ_j\rangle$
toward 0.

## Which mitigation fits

* **ZNE** is the standard companion of VQE energies (Kandala et al., 2019): cheap, model-free, and the energy is a
  smooth function of the noise scale.
* **Readout mitigation** (M3, TREX) removes the assignment error on every $Z_iZ_j$.
* **Twirling** turns coherent over-rotations into Pauli noise that ZNE extrapolates well.
* **Symmetry verification** helps when $H$ conserves a quantity (particle number, parity) that the ansatz keeps.
* **PEC** removes the bias completely for small circuits, at an overhead $\gamma^2$ in shots that grows with depth.

## Try it

1. Load the VQE template, open *Time travel*, and move $\theta_0$: watch $\langle H\rangle_{\rm ideal}$ and
   $\langle H\rangle_{\rm noisy}$ separate.
2. Open *Energy landscape* and scan one angle: the noisy curve is the ideal curve squeezed toward 0.
3. In *Mitigate & compare*, compare *Unmitigated*, *ZNE · Richardson* and *PEC · inverse* at the optimum.
