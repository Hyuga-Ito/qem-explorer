# HHL: solving a linear system

**The problem.** Given a Hermitian $N\times N$ matrix $A$ and a vector $\vec b$, find $\vec x=A^{-1}\vec b$. HHL
(Harrow, Hassidim and Lloyd, 2009) prepares the *quantum state* $|x\rangle\propto A^{-1}|b\rangle$ in time polynomial in
$\log N$ — for sparse, well-conditioned $A$ and when a few expectation values of $|x\rangle$ are all you need.

## The algorithm

Write $|b\rangle=\sum_j\beta_j|u_j\rangle$ in the eigenbasis of $A$ ($A|u_j\rangle=\lambda_j|u_j\rangle$). Then
$A^{-1}|b\rangle=\sum_j\beta_j\lambda_j^{-1}|u_j\rangle$: we need to multiply each component by $1/\lambda_j$.

1. **Phase estimation** of $e^{iAt}$ writes each eigenvalue into a *clock* register:
   $\sum_j\beta_j|u_j\rangle|\tilde\lambda_j\rangle$.
2. **Controlled rotation** of an ancilla, conditioned on the clock:
   $|\tilde\lambda\rangle|0\rangle\mapsto|\tilde\lambda\rangle\bigl(\sqrt{1-C^2/\tilde\lambda^2}\,|0\rangle+\tfrac{C}{\tilde\lambda}|1\rangle\bigr)$,
   i.e. $R_Y(2\arcsin(C/\tilde\lambda))$.
3. **Uncompute** the phase estimation, returning the clock to $|0\rangle$.
4. **Post-select** the ancilla on $|1\rangle$: the system register holds $\sum_j\beta_j\,C\lambda_j^{-1}|u_j\rangle\propto|x\rangle$.
   The success probability is $C^2\sum_j|\beta_j|^2/\lambda_j^2$.

<!-- fig:hhl -->

## In the app

The template *HHL, 2×2 system* is the textbook example

$$A=\begin{pmatrix}1&-\tfrac13\\-\tfrac13&1\end{pmatrix},\qquad |b\rangle=|0\rangle .$$

Its eigenvalues are $\tfrac23$ (eigenvector $(1,1)/\sqrt2$) and $\tfrac43$ (eigenvector $(1,-1)/\sqrt2$). With
$t=3\pi/4$ the phases $\lambda t/2\pi$ are $\tfrac14$ and $\tfrac12$, so two clock qubits (q1, q2) read them exactly as
$\tilde\lambda=1$ and $2$. With $C=1$ the ancilla (q3) rotates by $\pi$ and $\pi/3$. The solution is

$$|x\rangle=\frac{3|0\rangle+|1\rangle}{\sqrt{10}},\qquad \langle x|Z|x\rangle=\frac{9-1}{10}=0.8 ,$$

and the success probability is $\tfrac12\cdot1+\tfrac12\cdot\tfrac14=0.625$. The observable is
$\langle|1\rangle\langle1|_{\rm anc}\otimes Z_b\rangle=0.625\times0.8=0.5$; dividing by $P(\text{anc}=1)$ gives 0.8.

## The fine print

HHL's speed-up needs (Aaronson, 2015): efficient preparation of $|b\rangle$; a sparse, efficiently simulable $A$; a
small condition number $\kappa=\lambda_{\max}/\lambda_{\min}$ (the cost grows with $\kappa$); and a question about
$|x\rangle$ that a few measurements answer — reading out all $N$ amplitudes would erase the advantage.

## What noise does and which mitigation fits

The circuit is two phase estimations with controlled-$e^{iAt}$ gates. Noise lowers the post-selection probability
and mixes the system qubit. **Post-selection is itself a mitigation**: the template's symmetry (ancilla parity)
lets *symmetry verification* discard shots with the ancilla in $|0\rangle$ — including many shots where an error
happened. ZNE applies to the joint observable; the ratio $\langle P_1Z\rangle/\langle P_1\rangle$ then removes part
of the remaining bias (errors that reduce both).

## Try it

1. Load the template, compare *Unmitigated*, *Symmetry verification* and *ZNE · Richardson*.
2. In *Time travel*, watch the clock qubits return to $|0\rangle$ after the uncomputation — only in the ideal state.
