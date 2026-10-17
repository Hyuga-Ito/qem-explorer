# Security and privacy

## Credentials

An IBM Quantum token is a password.

- **In the app:** the token goes into a password field (`st.text_input(..., type="password")`). It is used to open the connection and is then removed from the session. It is never part of the state file (`qem_explorer_state.json`), the experiment YAML, the exported Qiskit script, a ZIP bundle or any log.
- **On the command line:** the token comes from the environment variable `QISKIT_IBM_TOKEN`, from a saved account (`QiskitRuntimeService.save_account`), or from a hidden prompt (`--ask-token`, `getpass`). It is never a command-line argument, which would leave it in shell history and process lists. It is never written to a run directory or printed. Connection errors are reported with the token masked.
- **Experiment documents:** `python -m qemx.experiment` rejects any field whose name looks like a secret (`token`, `password`, `secret`, `api_key`, `credential`) and names its location.
- **Notebooks:** before a notebook goes into a repository or an appendix, clear any cell that held a token and switch to the environment variable or a saved account. If a token was ever pasted into a file, rotate it.

## Untrusted input

- **Circuit text, Hamiltonians, layouts and method parameters** are parsed by restricted readers: a whitelist of gates, numbers and a small set of math names. Nothing is passed to `eval` or `exec`.
- **State files and experiment YAML:**
  - Validation is field by field (`qemx/state_schema.py`), with size limits of 20 MB.
  - YAML is read with `yaml.safe_load`, so no tags and no objects.
  - LaTeX labels are restricted to a whitelist of commands.
- **Manifests** (`qemx/phase`, divergence) are validated before anything runs. Unknown keys are errors, not silently ignored.

## Resource limits

There are deliberately no hard limits on qubits or depth. Every execution path catches out-of-memory errors, timeouts and interrupts, and reports them as messages rather than crashes. The image renderer for exports runs in a separate, time-limited process (`qemx/render_worker.py`).

## Hardware spending

No command submits QPU jobs without an explicit `run-ibm`/approval step (see `docs/hardware_preflight.md`). The planner keeps its own usage ledger, because a per-job `max_execution_time` does not bound a campaign.

## Reporting a vulnerability

Contact the maintainer privately, not through a public issue, and describe how to reproduce the problem. *(Maintainer contact to be added by the project owner before publication.)*
