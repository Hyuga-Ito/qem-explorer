r"""Learning-based mitigation: Clifford data regression (CDR; Czarnik, Arrasmith, Coles & Cincio, Quantum 5, 592
(2021)) and a feature-regression extension of it (FR, this project's design, see below).

Naming and attribution. ``features="fr"`` selects *feature regression* (FR): CDR's regression with more features of
the target's
own shots — ⟨O⟩, every ⟨Z_i⟩ and ⟨Z_iZ_j⟩ — fitted by OLS, Ridge or Lasso. FR is a heuristic of this project, not a
published protocol; the closest published work is ML-QEM (Liao et al., arXiv:2309.17368, Nat. Mach. Intell. 2024),
which feeds noisy expectation values of the target observables and circuit/noise descriptors into regression models.
It is **not** the learning-based quasi-probability method of Strikis et al. (PRX Quantum 2, 040330 (2021)), which
learns the coefficients of a linear combination of noisy values of *Pauli-inserted variants* of the circuit; that
idea, scaled up with random weight-1 neighbors, an ℓ1 bound and 2-design training circuits, is neighbor-informed
learning (Chen et al., arXiv:2512.12578), the separate estimator ``nil`` (``methods/nil.py``).

Training circuits replace the target's non-Clifford rotations by Clifford angles, so their ideal
values are classically computable, while their noise stays that of the target. A regression
learns ideal ≈ b + c·features(noisy) and applies it to the target.

features  cdr   the noisy expectation value alone: ŷ = a x + b
          fr    (FR) ⟨O⟩, every ⟨Z_i⟩ and ⟨Z_iZ_j⟩ of the noisy distribution. Until 1.4.2 this value was called
                ``nil``, which read as neighbor-informed learning; manifests and states that say ``nil`` are migrated
                explicitly (methods.base.PARAM_ALIASES) — the estimator itself is unchanged.
models    OLS, Ridge (L2 penalty α), Lasso (L1 penalty α; sparse), all for the objective
          Σ_i (y_i − b − c·x_i)² + α Ω(c). The estimator is linear in the target's features, so its shot noise is
          amplified by γ = ‖c‖₁ at most.

Variance  v/N_t from the target's shots (var1), plus the constant var_fixed u from the training circuits' shot noise:
          with a fixed number of training shots it does not shrink as N_t grows. u is the variance of the estimate
          over refits on resampled training data (Monte Carlo, a fixed seed): the fit is nonlinear in the data, and
          with collinear features (FR: ⟨O⟩ is often a combination of the ⟨Z_i⟩, ⟨Z_iZ_j⟩) a first-order (delta-method)
          value is not defined.
Readout   The calibration error reaches the estimate through every training feature and the refit. Its coefficient w
          (Var = w/s for s shots per calibration circuit) is measured by Monte Carlo as well: θ̂ drawn from the
          calibration circuits at a reference s, the training data resampled at the protocol's shots, paired with
          the same training data at the exact θ (``cal_w``). Samplers accept per-replicate θ̂ (``joint_calibration``).

Training set  A regression needs training values that vary. If every training circuit has (nearly) the same ideal
value, the fit collapses to that constant. The *adaptive* strategy therefore checks the spread of the training
values, which needs only classical (ideal) simulations, before any noisy circuit is run, and escalates until it is
large enough: near-Clifford angles (σ = 1) → uniformly drawn Clifford angles → one more kept non-Clifford gate →
… → also redrawing the rotations that already have Clifford angles → also random basis-state inputs (X gates). The
choice depends only on the circuit, the observable and the seed, so a hardware run records and replays the same
training circuits. If no stage gives a usable spread, the result is flagged as degenerate and falls back to the
unmitigated value instead of reporting a constant. The same fallback applies when the noisy training features vary by
less than the shot noise of the training shots can resolve; that test depends on the number of training shots, and
the training-shot resampler applies it at every number of shots it is asked for.
"""
from __future__ import annotations

import warnings

import numpy as np

from .base import MethodResult, Param, QEMMethod, register
from .raw import cal_fields, display_circuit, readout_fun
from ..resources import check
from ..stats import protocol_contract, training_moments
from ..transforms import clifford_replace, non_clifford_slots


def feature_matrix(n: int, o: np.ndarray, kind: str) -> tuple:
    """Per-shot feature matrix F[x, :] and names; histogram features are p @ F."""
    x = np.arange(2 ** n)
    z = 1.0 - 2.0 * ((x[:, None] >> np.arange(n)[None, :]) & 1)
    cols, names = [o], ["⟨O⟩"]
    if kind in ("fr", "nil"):                    # "nil": the legacy name of FR (methods.base.PARAM_ALIASES)
        for i in range(n):
            cols.append(z[:, i])
            names.append("⟨Z%d⟩" % i)
        for i in range(n):
            for j in range(i + 1, n):
                cols.append(z[:, i] * z[:, j])
                names.append("⟨Z%dZ%d⟩" % (i, j))
    return np.stack(cols, 1), names


def fit(X, y, model: str, alpha: float, diag: dict | None = None):
    """(coef, intercept). With ``diag`` (a dict) the solver's diagnostics are recorded in it: ``converged`` (False
    when scikit-learn raised a ConvergenceWarning — recorded, not hidden), ``n_iter``, ``coef_l1``, ``condition``
    (2-norm condition number of the centred design)."""
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if model == "ols" or X.shape[0] < 3:
            m = LinearRegression().fit(X, y)
        elif model == "ridge":
            m = Ridge(alpha=alpha).fit(X, y)
        else:
            # scikit-learn minimizes (1/2N)‖y − b − Xc‖² + α'‖c‖₁; α' = α/(2N) is the documented ‖·‖² + α‖c‖₁
            m = Lasso(alpha=alpha / (2.0 * len(y)), max_iter=50000).fit(X, y)
    if diag is not None:
        Xc = np.asarray(X, float) - np.asarray(X, float).mean(0)
        sv = np.linalg.svd(Xc, compute_uv=False) if Xc.size else np.zeros(0)
        diag.update(converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                    n_iter=int(np.max(getattr(m, "n_iter_", 0) or 0)) if hasattr(m, "n_iter_") else None,
                    coef_l1=float(np.abs(m.coef_).sum()),
                    condition=float(sv[0] / sv[-1]) if len(sv) and sv[-1] > 1e-15 * max(sv[0], 1e-300) else float("inf"))
    return np.asarray(m.coef_, float), float(m.intercept_)


CAL_REF_SHOTS = 4096      # calibration shots per circuit at which cal_w is measured (Var_cal = w / s)
MC_REPS = 500             # Monte-Carlo refits behind var_fixed (u) and cal_w of the as-run result (±6 %)


@register
class Learning(QEMMethod):
    key = "learning"
    label = "Learning-based (CDR / feature regression)"
    family = "learning"
    tex = r"\hat O=b+\mathbf c\cdot\mathbf x_{\rm noisy},\qquad (b,\mathbf c)=\arg\min\sum_i\bigl(y_i-b-\mathbf c\cdot\mathbf x_i\bigr)^2+\alpha\,\Omega(\mathbf c)"
    why = (r"Near-Clifford training circuits share the target's noise but have answers a classical computer can "
           r"check. The regression learns how the noise distorts the answer and undoes it on the target. "
           r"It fails where the noise depends on the rotation angles the training circuits removed. "
           r"The lasso ($\Omega=\|\mathbf c\|_1$) keeps few features, which keeps $\gamma=\|\mathbf c\|_1$ small.")
    params = (
        Param("features", "features", "choice", "cdr", ("cdr", "fr"),
              help="cdr: the noisy ⟨O⟩ only; fr: feature regression (FR, this project's heuristic) — plus ⟨Zi⟩, "
                   "⟨ZiZj⟩ of the same shots. "
                   "(Neighbor-informed learning of Chen et al. is the separate estimator “Neighbor-informed "
                   "learning”.)"),
        Param("model", "regression", "choice", "ols", ("ols", "ridge", "lasso")),
        Param("alpha", r"penalty $\alpha$", "float", 1e-3, lo=0.0, hi=1.0, step=1e-4),
        Param("n_train", "training circuits", "int", 16, lo=4, hi=4096, step=1),
        Param("keep", "non-Clifford gates kept", "int", 0, lo=0, hi=64, step=1,
              help="Rotations left at their true angle in every training circuit (more = training circuits closer "
                   "to the target). With the adaptive strategy this is the starting value."),
        Param("strategy", "training-set strategy", "choice", "adaptive", ("adaptive", "fixed"),
              help="adaptive: diversify the Clifford sampling and keep more non-Clifford gates until the training "
                   "values vary; fixed: exactly the settings above."),
        Param("min_spread", "min. spread of training values", "float", 0.05, lo=0.0, hi=1.0, step=0.01,
              help="Required standard deviation of the ideal training values, as a fraction of the observable's "
                   "range max(o) − min(o)."),
        Param("train_shots", "shots per training circuit", "int", 1024, lo=64, hi=65536, step=64,
              help="Training shots are part of the method's total budget (training circuits × this)."),
    )

    def min_cost(self, ctx, features="cdr", model="ols", alpha=1e-3, n_train=16, keep=0, train_shots=1024,
                 strategy="adaptive", min_spread=0.05) -> dict:
        if not non_clifford_slots(ctx.spec):       # a Clifford circuit: learning is not applied, nothing is trained
            return dict(training=0, target_circuits=1, note="Clifford circuit: no training circuits")
        return dict(training=int(n_train) * int(train_shots), target_circuits=1,
                    note="%d training circuits × %d shots" % (int(n_train), int(train_shots)))

    def short(self, p):
        extra = []
        if int(p.get("keep", 0)):
            extra.append("keep %d" % int(p["keep"]))
        if p.get("strategy", "adaptive") == "fixed":
            extra.append("fixed set")
        return "%s (%s)" % ("CDR" if p["features"] == "cdr" else "FR", ", ".join(
            [{"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso"}[p["model"]]] + extra))

    def run(self, ctx, features="cdr", model="ols", alpha=1e-3, n_train=16, keep=0, train_shots=1024,
            strategy="adaptive", min_spread=0.05) -> MethodResult:
        n = ctx.n
        o = ctx.obs.o
        F, names = feature_matrix(n, o, features)
        target = ctx.run()
        slots = non_clifford_slots(ctx.spec)
        if not slots:
            mean = ctx.expval(target)
            return MethodResult(mean, ctx.var_shot(target), 1, gamma=ctx.readout_gamma(),
                                **cal_fields(ctx, readout_fun(ctx, target)),
                                implemented=[("Implemented circuit", display_circuit(ctx))],
                                notes=["The circuit is Clifford: its ideal value is classically computable, and "
                                       "training circuits would equal the target. Learning is not applied."],
                                sampler=lambda N, B, rng: ctx.sample_expvals(target, N, B, rng))
        span = float(np.ptp(o)) or 1.0
        need = float(min_spread) * span
        keep_eff = min(int(keep), max(len(slots) - 1, 0))     # keeping every slot would reproduce the target itself
        train_specs, y, plan, tried = choose_training_set(ctx, slots, int(n_train), keep_eff, strategy, need)
        spread = float(np.std(y))
        notes = []
        fit_diag = {}                              # solver diagnostics of the as-run fit (recorded)
        if keep_eff != int(keep):
            notes.append("“non-Clifford gates kept” = %d exceeds what this circuit allows (%d non-Clifford gates; at "
                         "least one must be replaced), so %d was used." % (int(keep), len(slots), keep_eff))
        if np.ptp(y) < 1e-9:
            # known from classical simulation alone: no training circuit needs to run, and nothing is fitted
            warning = ("The training data cannot train the model: every training circuit has the same ideal value "
                       "%.4g, for all %d training-set strategies tried (up to “%s”). Learning is not applied; the value "
                       "shown is the unmitigated one, with its own uncertainty and cost. Choose an observable whose "
                       "training values vary, or use another method." % (y[0], len(tried), tried[-1]["text"]))
            return MethodResult(
                ctx.expval(target), ctx.var_shot(target), 1, gamma=ctx.readout_gamma(),
                **cal_fields(ctx, readout_fun(ctx, target)),
                implemented=[("Target circuit", display_circuit(ctx)),
                             ("Training circuit 1 of %d (not run)" % n_train, display_circuit(ctx, train_specs[0]))],
                details=dict(x_train=np.zeros(0), y_train=y, x_target=float("nan"), coef=np.zeros(1), intercept=0.0,
                             names=names, n_slots=len(slots), rank=0, n_features=int(F.shape[1]), degenerate=True,
                             warning=warning, spread=spread, spread_needed=need, keep_used=plan["keep"],
                             sampling=plan["text"], stages=tried),
                notes=notes, sampler=lambda N, B, rng: ctx.sample_expvals(target, N, B, rng))
        tr_runs = [ctx.run(s, role="training") for s in train_specs]
        feats = lambda rr: ctx.mitigate(ctx.measured(rr)) @ F
        X = np.array([feats(r) for r in tr_runs])
        xt = feats(target)
        T_, k = X.shape
        if len(tried) > 1 or plan["keep"] != keep_eff:
            notes.append("Training set: %s (tried %s)." % (plan["text"], "; ".join(
                "%s → spread %.3g" % (t["text"], t["spread"]) for t in tried)))
        Xc = X - X.mean(0)
        # per-shot covariance of each training circuit's feature vector (readout mitigation folded in linearly)
        Fe = ctx.mitigation_matrix().T @ F
        tr_meas = [ctx.measured(r) for r in tr_runs]
        covs = []
        for f_ in tr_meas:
            pf = np.clip(f_, 0, None)
            pf = pf / max(pf.sum(), 1e-300)
            mu = pf @ Fe
            covs.append(Fe.T @ (pf[:, None] * Fe) - np.outer(mu, mu))
        var_max = max(max(float(np.max(np.diag(c_))) for c_ in covs), 0.0)

        def rank_at(shots):
            """Feature directions the training circuits spread along by more than the shot noise of ``shots`` shots
            per training circuit can resolve: singular values above the noise edge σ(√n_train + √n_features)."""
            sig_ = float(np.sqrt(var_max / max(int(shots), 1)))
            tol_ = max(1e-9, sig_ * (np.sqrt(T_) + np.sqrt(k)))
            return int(np.linalg.matrix_rank(Xc, tol=tol_)) if T_ > 1 else 0

        rank = rank_at(train_shots)
        degenerate = bool(rank == 0)
        warning = ""
        p = np.clip(ctx.measured(target), 0, None)
        tg_meas = ctx.measured(target)
        raw_value, raw_var1 = ctx.expval(target), ctx.var_shot(target)
        if degenerate:
            coef, b = np.zeros(k), float(np.mean(y))
            mean = raw_value
            warning = ("The training data cannot train the model: the noisy training features vary by less than the "
                       "shot noise of %d shots per training circuit can resolve (no usable direction). Learning is "
                       "not applied; the value shown is the unmitigated one. Raise “non-Clifford gates kept” or the "
                       "training shots, choose another observable, or use another method." % int(train_shots))
        else:
            coef, b = fit(X, y, model, alpha, fit_diag)
            mean = float(b + xt @ coef)
            if not fit_diag.get("converged", True):
                notes.append("The %s solver did not converge within its iteration limit on the training data; the "
                             "estimate uses its last iterate." % model)
            if spread < need:
                warning = ("Weak training signal: the ideal training values spread by only %.3g (target %.3g), so the "
                           "fitted slope is poorly determined." % (spread, need))
        if rank < k:
            # is the target inside the span the training data explored?
            Q = np.linalg.svd(Xc, full_matrices=False)[2][:rank] if rank else np.zeros((0, k))
            d = xt - X.mean(0)
            resid = float(np.linalg.norm(d - Q.T @ (Q @ d))) / max(float(np.linalg.norm(d)), 1e-12)
            notes.append("The training circuits explore only %d of %d feature directions above shot noise%s." % (
                rank, k, "; the target lies outside them (%.0f %% of its offset), so the fit extrapolates"
                % (100 * resid) if resid > 0.05 else ""))

        def var1_of(c):
            """Per-shot variance of c·features on the target (readout mitigation folded in linearly)."""
            g_ = Fe @ c
            return max(float(p @ g_ ** 2 - (p @ g_) ** 2), 0.0)
        var1 = var1_of(coef) if not degenerate else raw_var1

        def refit(mit):
            """The whole learning estimate with another readout mitigation (training features refitted too)."""
            if degenerate:
                return float(o @ mit(tg_meas))
            Xm = np.array([mit(f) @ F for f in tr_meas])
            cm, bm = fit(Xm, y, model, alpha)
            return float(bm + (mit(tg_meas) @ F) @ cm)

        # readout mitigation of sampled frequencies: linear (none, tensored, TREX) -> one matrix product for all
        # replicates; M3 restricts to the observed bitstrings (nonlinear) and is applied replicate by replicate. A
        # sampled calibration θ̂ is applied through its (tensored) inverse, as Context._mitigator does.
        linear_ro = ctx.recipe.readout in ("none", "tensored", "trex")
        FE_lin = Fe if linear_ro else None
        p_tg = p / max(p.sum(), 1e-300)

        def FE_theta(t):
            """(2ⁿ, k) per-shot feature values with the mitigation of calibration θ (all k columns in one pass)."""
            return ctx.o_eff_theta(t, F)

        def features_of(freqs):
            """(R, k) feature rows of R sampled frequency vectors with this recipe's mitigation."""
            freqs = np.asarray(freqs)
            if linear_ro:
                return freqs @ FE_lin
            return np.array([ctx.mitigate(f) @ F for f in freqs])

        def features_theta(freq_list, ths):
            """[(R, k) feature rows of each (R, 2ⁿ) frequency array], replicate r mitigated with θ̂_r: one feature map
            per replicate, shared by every circuit of that replicate."""
            R_ = len(ths)
            out = [np.empty((R_, k)) for _ in freq_list]
            for r, t in enumerate(ths):
                FE = FE_theta(t)
                for i, fr in enumerate(freq_list):
                    out[i][r] = fr[r] @ FE
            return out

        def bad_rows(thetas, R):
            if thetas is None:
                return np.zeros(int(R), bool)
            return np.array([not ctx.theta_invertible(t) for t in thetas])

        def sampler(N, B, rng, theta=None):
            """B complete experiments with N target shots: training features sampled with train_shots shots per
            training circuit, the target with N, the regression refitted on every replicate; with ``theta``
            (B calibration draws θ̂) every circuit of replicate b is mitigated with θ̂_b."""
            th = ctx.theta_rows(theta, B)
            bad = bad_rows(th, B)
            ths = None if th is None else [t if not b_ else ctx.cal_theta() for t, b_ in zip(th, bad)]
            if degenerate:
                fr = ctx.sample_freqs(target, N, B, rng)
                if ths is None:
                    est = (fr @ ctx.o_eff()) if linear_ro else np.array([o @ ctx.mitigate(f) for f in fr])
                else:
                    est = np.array([f @ ctx.o_eff_theta(t) for f, t in zip(fr, ths)])
                est = np.asarray(est, float)
                est[bad] = np.nan
                return est
            trF = [ctx.sample_freqs(r, train_shots, B, rng) for r in tr_runs]          # each (B, 2ⁿ)
            tgF = ctx.sample_freqs(target, N, B, rng)
            if ths is None:
                trX = [features_of(f_) for f_ in trF]
                tgX = features_of(tgF)
            else:
                *trX, tgX = features_theta(trF + [tgF], ths)
            est = np.empty(int(B))
            for bi in range(int(B)):
                if bi % 16 == 0:
                    check()                             # cooperative time limit
                Xb = np.array([trX[i][bi] for i in range(T_)])
                cb, bb = fit(Xb, y, model, alpha)
                est[bi] = bb + tgX[bi] @ cb
            est[bad] = np.nan
            return est

        def train_resampler(s_tr, R, rng, theta=None):
            """(estimates, per-shot target variances): R refits on training features sampled with s_tr shots per
            training circuit, each applied to the exact (infinite-shot) target features, and the per-shot variance
            of that fit's target estimator, var1(c) = Var_x[c·f(x)]. By the law of total variance an experiment with
            N target shots has Var = E_c[var1(c)]/N + Var_c[estimate] exactly (no first-order cross term) when the
            readout mitigation is linear (none, tensored, TREX): given c the estimate is then a linear function of
            the target's sample frequencies. M3 is nonlinear in sampled data (it solves on the observed bitstrings),
            so for M3 the decomposition holds only for large N (the target variance is first order).
            The protocol's fallback is applied at s_tr: if the training data cannot resolve a feature direction at
            that many shots, the estimate is the unmitigated one. With ``theta`` (R calibration draws θ̂) the training
            and target features of replicate r are mitigated with θ̂_r (the random draws are the same as without, so
            the two calls pair up)."""
            th = ctx.theta_rows(theta, R)
            bad = bad_rows(th, R)
            ths = None if th is None else [t if not b_ else ctx.cal_theta() for t, b_ in zip(th, bad)]
            if rank_at(s_tr) == 0:
                if ths is None:
                    est = np.full(int(R), float(raw_value))
                else:
                    est = np.array([p_tg @ ctx.o_eff_theta(t) for t in ths])
                est[bad] = np.nan
                return est, np.full(int(R), float(raw_var1))
            trF = [ctx.sample_freqs(r, int(s_tr), int(R), rng) for r in tr_runs]
            if ths is None:
                trX, xts = [features_of(f_) for f_ in trF], None
            else:
                ptg = np.broadcast_to(p_tg, (int(R), len(p_tg)))
                *trX, xts = features_theta(trF + [ptg], ths)                 # exact target features per θ̂
            est, v1 = np.empty(int(R)), np.empty(int(R))
            for bi in range(int(R)):
                if bi % 16 == 0:
                    check()
                Xb = np.array([trX[i][bi] for i in range(T_)])
                cb, bb = fit(Xb, y, model, alpha)
                est[bi] = bb + (xt if xts is None else xts[bi]) @ cb
                v1[bi] = var1_of(cb)
            est[bad] = np.nan
            return est, v1

        # protocol moments at the protocol's training shots, by Monte Carlo (fixed seeds; qemx/stats.py): the mean of the
        # refitted estimates, their variance u and the mean conditional per-shot target variance — not the moments of
        # the reference fit to exact training features
        seed64 = int(ctx.seed) % (2 ** 63)
        mom = training_moments(train_resampler, int(train_shots), np.random.default_rng([seed64, 0x4C45, 1]))
        u_train = mom.u
        contract = protocol_contract(ctx, mom, mean, var1, notes, degenerate)
        mean, var1 = contract.pop("mean"), contract.pop("var1")
        if not degenerate:
            notes.append("Shot noise of the training data (%d × %d shots) adds a variance %.3g that more target shots "
                         "cannot remove (σ floor %.3g; %d Monte-Carlo refits)." % (T_, int(train_shots), u_train,
                                                                                   np.sqrt(u_train), mom.repeats))
        cal = cal_fields(ctx, refit)
        if cal:
            cal["cal_w"] = calibration_w(ctx, train_resampler, int(train_shots), MC_REPS, seed64)
            contract["flags"] = tuple(contract["flags"]) + ("calibration-mean-first-order",)

        return MethodResult(
            **contract,
            mean=mean, var1=var1, var_fixed=u_train, circuits=1, train_circuits=int(len(tr_runs)),
            fixed_shots=int(len(tr_runs)) * int(train_shots), **cal, joint_calibration=True,
            gamma=(float(np.abs(coef).sum()) if not degenerate else 1.0) * ctx.readout_gamma(),
            implemented=[("Target circuit", display_circuit(ctx)),
                         ("Training circuit 1 of %d (Clifford angles)" % n_train, display_circuit(ctx, train_specs[0]))],
            details=dict(x_train=X[:, 0], y_train=y, x_target=float(xt[0]), coef=coef, intercept=b, names=names,
                         X=X, xt=xt, n_slots=len(slots), rank=rank, n_features=int(k), degenerate=degenerate,
                         warning=warning, spread=spread, spread_needed=need, keep_used=plan["keep"],
                         sampling=plan["text"], stages=tried, cal_w_ref_shots=CAL_REF_SHOTS,
                         fit_diagnostics=fit_diag),
            notes=notes, sampler=sampler, train_resampler=train_resampler)


def calibration_w(ctx, train_resampler, s_tr: int, R: int, seed64: int, s_ref: int = CAL_REF_SHOTS) -> float:
    """Calibration coefficient w of a learning method by Monte Carlo: R calibrations θ̂ drawn with s_ref shots per
    calibration circuit, each paired with one draw of training data at s_tr shots; w = s_ref·Var(est(θ̂) − est(θ))
    over the pairs (the same training data with the exact calibration). Non-invertible draws are dropped."""
    th = ctx.sample_cal_theta(int(s_ref), int(R), np.random.default_rng([seed64, 0x4341, 2]))
    if th is None:
        return 0.0
    e0, _ = train_resampler(int(s_tr), int(R), np.random.default_rng([seed64, 0x4341, 1]))
    e1, _ = train_resampler(int(s_tr), int(R), np.random.default_rng([seed64, 0x4341, 1]), theta=th)
    d = np.asarray(e1, float) - np.asarray(e0, float)
    d = d[np.isfinite(d)]
    return float(s_ref * np.var(d, ddof=1)) if len(d) > 1 else 0.0


def choose_training_set(ctx, slots, n_train: int, keep: int, strategy: str, need: float) -> tuple:
    """(training specs, their ideal values, chosen plan, plans tried). Uses only ideal (classical) values.

    Stages, closest to the target first: for k = keep, keep+1, …: near-Clifford angles (σ = 1), then uniformly drawn
    Clifford angles; then uniformly redrawn rotations everywhere; finally the same with random basis-state inputs. The first stage whose training values have a
    standard deviation ≥ `need` wins; otherwise the stage with the largest spread is used."""
    stages = [dict(keep=keep, sigma=1.0, all=False, flip=False)]
    if strategy == "adaptive":
        kmax = max(len(slots) - 1, keep)
        stages = []
        for k in range(keep, kmax + 1):
            stages += [dict(keep=k, sigma=1.0, all=False, flip=False), dict(keep=k, sigma=None, all=False, flip=False)]
            if len(stages) >= 12:                   # a dozen stages are plenty; the last one is always tried
                break
        stages += [dict(keep=0, sigma=None, all=True, flip=False), dict(keep=0, sigma=None, all=True, flip=True)]
    tried, best = [], None
    for i, stg in enumerate(stages):
        rng = np.random.default_rng(ctx.seed + 101 + 7919 * i)
        specs = [clifford_replace(ctx.spec, rng, keep=stg["keep"], sigma=stg["sigma"], resample_all=stg["all"],
                                  flip_inputs=stg["flip"]) for _ in range(n_train)]
        y = np.array([ctx.ideal_value(s) for s in specs])
        text = ("all rotations redrawn as random Clifford angles" + (", random basis-state inputs" if stg["flip"] else "")
                if stg["all"] else
                "%d non-Clifford gate%s kept, %s Clifford angles" % (stg["keep"], "" if stg["keep"] == 1 else "s",
                                                                     "near" if stg["sigma"] else "uniform"))
        rec = dict(stg, text=text, spread=float(np.std(y)))
        tried.append(rec)
        if best is None or rec["spread"] > best[3]["spread"] + 1e-12:
            best = (specs, y, rec, rec)
        if rec["spread"] >= need and np.ptp(y) > 1e-9:
            return specs, y, rec, tried
    specs, y, rec, _ = best
    return specs, y, rec, tried
