r"""The unmitigated estimator: the reference every method is compared with."""
from __future__ import annotations

from .base import MethodResult, QEMMethod, register
from ..transforms import dd_display, twirl_instance


def display_circuit(ctx, spec=None):
    """What the device runs under this recipe's transforms (one sampled twirl, DD pulses drawn)."""
    s = spec or ctx.spec
    if ctx.recipe.twirl != "none":
        s = twirl_instance(s, seed=ctx.seed)
    if ctx.recipe.dd != "none":
        s = dd_display(s, ctx.recipe.dd)
    return s


def cal_fields(ctx, fun) -> dict:
    """The readout calibration's accounting for a MethodResult: its circuits, and the sensitivity of the estimate
    to the calibration's estimated flip probabilities (``fun(mitigate)`` recomputes the estimate)."""
    c = ctx.cal_circuits()
    if not c:
        return {}
    g, th = ctx.cal_gradient(fun)
    return dict(cal_circuits=c, cal_grad=g, cal_theta=th, cal_fun=fun)


def readout_fun(ctx, rr, o=None):
    """fun(mitigate) for a linear estimate o · M f of one run."""
    o = ctx.obs.o if o is None else o
    f = ctx.measured(rr)
    return lambda mit: float(o @ mit(f))


@register
class Raw(QEMMethod):
    key = "raw"
    label = "Unmitigated"
    family = "raw"
    tex = r"\hat O=\sum_x \hat p(x)\,o(x)"
    why = (r"The noisy device returns $\tilde O=\mathrm{Tr}[\mathcal E(\rho)\,O]$. Shots only shrink the "
           r"statistical error $\sqrt{\mathrm{Var}(o)/N}$; the bias $\tilde O-O$ stays.")
    params = ()

    def short(self, p):
        return "Unmitigated"

    def run(self, ctx, **params) -> MethodResult:
        rr = ctx.run()
        return MethodResult(
            mean=ctx.expval(rr), var1=ctx.var_shot(rr), circuits=1, gamma=ctx.readout_gamma(),
            **cal_fields(ctx, readout_fun(ctx, rr)),
            implemented=[("Implemented circuit", display_circuit(ctx))],
            sampler=lambda N, B, rng: ctx.sample_expvals(rr, N, B, rng))
