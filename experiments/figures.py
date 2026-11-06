"""Paper figures, read from results/*.csv."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import dgp  # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.environ.get("RESULTS_DIR", os.path.join(BASE, "results"))
FIG = os.path.join(BASE, "paper", "figures")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def fig_reactive():
    d = pd.read_csv(os.path.join(RES, "s1.csv"))
    g = d.groupby(["controls", "kappa", "sigma_v"]).theta.agg(["mean", "std"]).reset_index()
    sv = np.linspace(0.2, 2.1, 200)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    colors = {0.0: "C0", 0.25: "C1", 0.5: "C2", 1.0: "C3"}
    for k, c in colors.items():
        ax.plot(sv, dgp.THETA + dgp.reactive_bias(k, sv), color=c, lw=1.2)
        s = g[(g.controls == "X") & (g.kappa == k)]
        ax.errorbar(
            s.sigma_v,
            s["mean"],
            yerr=s["std"],
            fmt="o",
            color=c,
            ms=4,
            capsize=2,
            label=rf"$\kappa = {k:g}$, forecast not adjusted for",
        )
    s = g[g.controls == "X+F"].groupby("sigma_v")["mean"].mean()
    ax.plot(s.index, s.values, "k^", ms=5, label=r"forecast adjusted for (all $\kappa$)")
    ax.fill_between(
        sv,
        dgp.THETA - 1 / (2 * sv),
        dgp.THETA + 1 / (2 * sv),
        color="0.85",
        zorder=0,
        label=r"worst-case bound $\theta \pm |b| s_W / (2\sigma_v)$",
    )
    ax.axhline(dgp.THETA, color="k", lw=0.8, ls="--")
    ax.set_ylim(-2.1, 1.6)
    ax.set_xlabel(r"Exogenous price variation $\sigma_v$")
    ax.set_ylabel(r"Estimate $\hat\theta$ (true $\theta = 0.5$)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "reactive.png"), dpi=200)
    plt.close(fig)


def fig_coarse():
    d = pd.read_csv(os.path.join(RES, "s2.csv"))
    d = d[d.learner == "spline"]
    g = (
        d.groupby(["calendar", "sigma_v"])
        .agg(m=("theta", "mean"), s=("theta", "std"), ov=("overlap", "mean"))
        .reset_index()
    )
    sv = np.linspace(0.07, 0.72, 200)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axes[0]
    w = dgp.VAR_WITHIN_MONTH
    ax.fill_between(
        sv,
        dgp.THETA,
        dgp.THETA + dgp.GAMMA * np.sqrt(w) / (2 * sv),
        color="0.85",
        zorder=0,
        label="worst-case bound",
    )
    ax.plot(
        sv,
        dgp.THETA + dgp.coarse_calendar_bias(sv),
        color="C3",
        lw=1.2,
        label="Proposition 1, month indicators",
    )
    for cal, c, lab in (("month", "C3", "month indicators"), ("exact", "C0", "exact calendar")):
        s = g[g.calendar == cal]
        ax.errorbar(s.sigma_v, s.m, yerr=s.s, fmt="o", color=c, ms=4, capsize=2, label=lab)
    ax.axhline(dgp.THETA, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"Exogenous price variation $\sigma_v$")
    ax.set_ylabel(r"Estimate $\hat\theta$ (true $\theta = 0.5$)")
    ax.legend(fontsize=7)
    ax = axes[1]
    for cal, c, lab in (("month", "C3", "month indicators"), ("exact", "C0", "exact calendar")):
        s = g[g.calendar == cal]
        ax.plot(s.sigma_v, s.ov, "o-", color=c, ms=4, label=lab)
    ax.plot(
        sv,
        sv**2 / (dgp.VAR_SEASON_PART + sv**2),
        "k--",
        lw=0.8,
        label=r"exogenous share $\sigma_v^2 / \mathrm{Var}(D)$",
    )
    ax.set_xlabel(r"Exogenous price variation $\sigma_v$")
    ax.set_ylabel(r"Estimated overlap $\hat\omega$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "coarse_calendar.png"), dpi=200)
    plt.close(fig)


def fig_overlap_decomposition():
    from analysis import SCENARIO_LABELS, scenario_shares

    d = pd.read_csv(os.path.join(RES, "s3.csv"))
    shares = scenario_shares()
    labs = list(SCENARIO_LABELS)
    short = {
        "identified": "Identified",
        "weak_overlap": "Weak\noverlap",
        "coarse_calendar": "Coarse\ncalendar",
        "reactive_unlogged": "Reactive,\nnot logged",
        "reactive_forecast_logged": "Reactive,\nlogged",
    }
    names = [short[s] for s in labs]
    wv = np.array([shares[s][0] for s in labs])
    ww = np.array([shares[s][1] for s in labs])
    est = np.array([d[d.scenario == s].overlap.mean() for s in labs])
    bias = np.array([d[d.scenario == s].theta.mean() - dgp.THETA for s in labs])
    sd = np.array([d[d.scenario == s].theta.std() for s in labs])
    x = np.arange(len(labs))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.bar(x, wv, color="C0", label=r"exogenous share $\omega_V$")
    ax.bar(
        x, ww, bottom=wv, color="C3", hatch="//", alpha=0.8, label=r"confounded share $\omega_W$"
    )
    ax.plot(x, est, "kD", ms=5, label=r"estimated overlap $\hat\omega$")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("Share of price variance")
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    ax.barh(
        x[::-1], bias, xerr=sd, color=["C0" if abs(b) < 0.05 else "C3" for b in bias], capsize=2
    )
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(x[::-1])
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(r"Bias of $\hat\theta$ (true $\theta = 0.5$)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "overlap_decomposition.png"), dpi=200)
    plt.close(fig)


def fig_coverage():
    d = pd.read_csv(os.path.join(RES, "s2.csv"))
    g = d.groupby(["calendar", "learner", "sigma_v"]).covers.agg(["mean", "size"]).reset_index()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    styles = {
        ("exact", "spline"): ("C0", "o-"),
        ("exact", "boosting"): ("C0", "s--"),
        ("month", "spline"): ("C3", "o-"),
        ("month", "boosting"): ("C3", "s--"),
    }
    for (cal, lrn), (c, st) in styles.items():
        s = g[(g.calendar == cal) & (g.learner == lrn)]
        err = 1.96 * np.sqrt(s["mean"] * (1 - s["mean"]) / s["size"])
        ax.errorbar(
            s.sigma_v,
            s["mean"],
            yerr=err,
            fmt=st,
            color=c,
            ms=4,
            capsize=2,
            label=f"{'exact calendar' if cal == 'exact' else 'month indicators'}, {lrn}",
        )
    ax.axhline(0.95, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"Exogenous price variation $\sigma_v$")
    ax.set_ylabel("Coverage of 95% intervals")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "coverage.png"), dpi=200)
    plt.close(fig)


def fig_bias_map():
    ak = np.linspace(0.0, 2.0, 201)
    sv = np.linspace(0.1, 2.0, 191)
    A, S = np.meshgrid(ak, sv)
    bias = -A / (A**2 + S**2)
    ov = (A**2 + S**2) / (dgp.VAR_H + A**2 + S**2)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    im = ax.pcolormesh(A, S, bias, cmap="Reds_r", vmin=-2.5, vmax=0, shading="auto")
    cs = ax.contour(A, S, ov, levels=[0.1, 0.2, 0.4, 0.6, 0.8], colors="k", linewidths=0.7)
    ax.clabel(cs, fmt=lambda v: f"overlap {v:.1f}", fontsize=7)
    ax.plot(
        ak[ak <= 2.0],
        ak[ak <= 2.0],
        color="C2",
        lw=1.2,
        ls=":",
        label=r"worst case $a\,s_W = \sigma_v$",
    )
    fig.colorbar(im, ax=ax, label=r"bias $\theta^\ast - \theta$ ($b = -1$, $s_W = 1$)")
    ax.set_xlabel(r"Reaction to unobserved difficulty $a\,s_W$")
    ax.set_ylabel(r"Exogenous price variation $\sigma_v$")
    ax.set_ylim(0.1, 2.0)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bias_map.png"), dpi=200)
    plt.close(fig)


def fig_application_data():
    import application as A

    if not os.path.exists(A.DATA):
        print("data/zone_hour_2023-03.parquet not found; keeping application_data.png")
        return
    g = A.load()
    dres = A.demean(g.pay_per_mile, [g.zone, g.how])
    bins = pd.qcut(g.surprise, 20)
    bx = g.surprise.groupby(bins, observed=True).mean()
    by = dres.groupby(bins, observed=True).mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    ax = axes[0]
    ax.plot(bx, by, "o", color="C0", ms=4)
    slope = np.polyfit(g.surprise, dres, 1)[0]
    xx = np.linspace(bx.min(), bx.max(), 10)
    ax.plot(xx, slope * xx, color="C3", lw=1, label=f"slope {slope:.2f} \\$/mile per log unit")
    ax.set_xlabel("Demand surprise (log trips, net of zone and hour of week)")
    ax.set_ylabel("Residual pay per mile (\\$)")
    ax.legend(fontsize=7)
    ax = axes[1]
    prof = g.groupby("how").agg(pay=("pay_per_mile", "mean"), q=("logq", "mean"))
    ax.plot(prof.index, prof.pay, color="C0", lw=1)
    ax.set_xlabel("Hour of week (0 = Monday 00:00)")
    ax.set_ylabel("Mean pay per mile (\\$)", color="C0")
    ax2 = ax.twinx()
    ax2.plot(prof.index, prof.q, color="C1", lw=1)
    ax2.set_ylabel("Mean log trips", color="C1")
    ax.set_xticks(range(0, 169, 24))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "application_data.png"), dpi=200)
    plt.close(fig)


def fig_application():
    path = os.path.join(RES, "application.csv")
    if not os.path.exists(path):
        print("application results not found; skipping application figures")
        return
    import application as A

    d = pd.read_csv(path)
    pred = pd.read_csv(os.path.join(RES, "application_predicted.csv")).set_index("adjustment")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    ax = axes[0]
    order = list(A.ADJUSTMENTS)
    labels = ["zone,\nhour of week", "zone, hour of week,\ndemand", "zone, day part,\ndemand"]
    main = d[d.b == A.B_MAIN]
    for i, adj in enumerate(order):
        x = main[main.adjustment == adj].theta
        ax.errorbar(
            i,
            x.mean(),
            yerr=x.std(),
            fmt="o",
            color="C0",
            capsize=3,
            label="DML estimate" if i == 0 else None,
        )
        p = pred.loc[adj]
        ax.plot(
            i,
            A.THETA + A.B_MAIN * p.c_w + A.GAMMA * p.c_s,
            "x",
            color="C3",
            ms=7,
            label="Proposition 1 (plug-in)" if i == 0 else None,
        )
    ax.axhline(A.THETA, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"Estimate $\hat\theta$ (true $\theta = 0.5$)")
    ax.legend(fontsize=7)
    ax = axes[1]
    adj = order[0]
    s = d[d.adjustment == adj].groupby("b").theta.agg(["mean", "std"]).reset_index()
    ax.errorbar(
        s.b,
        s["mean"] - A.THETA,
        yerr=s["std"],
        fmt="o",
        color="C0",
        capsize=3,
        label="DML bias, zone and hour of week",
    )
    bb = np.linspace(-2.2, 1.2, 10)
    ax.plot(bb, bb * pred.loc[adj].c_w, color="C3", lw=1, label="Proposition 1 (plug-in)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"Outcome loading on demand surprise $b$")
    ax.set_ylabel("Bias")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "application_estimates.png"), dpi=200)
    plt.close(fig)


def fig_nuisance_error():
    from analysis import S4_NAMES, study4_data

    g = study4_data()
    if g is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    ax = axes[0]
    base = g[(g.n == 8000) & (g.sigma_v == 0.5)]
    omega_v = 0.25 / (2.0 + 0.25)  # sigma_v^2 / Var(D), Var h = 2 with the interaction
    ww = np.linspace(omega_v, 0.62, 100)
    ax.plot(
        ww,
        -(1 - omega_v / ww),
        color="C3",
        lw=1.2,
        label=r"Proposition 1: $-(1 - \omega_V / \hat\omega)$",
    )
    markers = {
        "spline": "o",
        "spline_interaction": "s",
        "random_forest": "^",
        "boosting": "D",
        "boosting_tuned": "v",
    }
    colors = {
        "spline": "C3",
        "spline_interaction": "C4",
        "random_forest": "C2",
        "boosting": "C0",
        "boosting_tuned": "C1",
    }
    for r in base.itertuples():
        ax.errorbar(
            r.ov,
            r.bias,
            yerr=r.sd,
            fmt=markers[r.learner],
            ms=6,
            capsize=2,
            color=colors[r.learner],
            ecolor=colors[r.learner],
            label=S4_NAMES[r.learner].replace("$X_3 X_4$", "X3*X4"),
        )
    ax.axvline(omega_v, color="0.5", ls=":", lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"Estimated overlap $\hat\omega$ (true $\omega_V$ dotted)")
    ax.set_ylabel(r"Bias of $\hat\theta$")
    ax.legend(fontsize=6.5, loc="lower left")
    ax = axes[1]
    for lrn, c, lab in (
        ("boosting", "C0", "Study 4, boosting defaults"),
        ("boosting_tuned", "C1", "Study 4, boosting tuned"),
    ):
        x = g[(g.learner == lrn) & (g.sigma_v == 0.5)].sort_values("n")
        ax.plot(x.n, x.bias_sd.abs(), "o-", color=c, label=lab)
    sub = os.path.join(RES, "application_subsample.csv")
    if os.path.exists(sub):
        q = (
            pd.read_csv(sub)
            .groupby("frac")
            .agg(n=("n", "first"), m=("theta", "mean"), s=("theta", "std"))
            .reset_index()
        )
        ax.plot(q.n, (q.m - 0.5).abs() / q.s, "s--", color="C2", label="NYC, demand adjusted")
    ax.set_xscale("log")
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel(r"|bias| / standard deviation")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "nuisance_error.png"), dpi=200)
    plt.close(fig)


def fig_delivery():
    path = os.path.join(RES, "delivery.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    order = ["zone + hour of week + week", "+ logged forecast inputs", "+ logged forecast"]
    labels = ["zone, hour of week,\nweek", "+ logged\nforecast inputs", "+ logged\nforecast"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for kappa, c, lab, off in (
        (0.0, "C0", "forecast uses logged inputs only", -0.08),
        (1.0, "C3", "forecast also uses real-time supply", 0.08),
    ):
        x = (
            d[d.kappa == kappa]
            .groupby("adjustment")
            .agg(
                m=("theta", "mean"),
                s=("theta", "std"),
                p=("predicted_bias", "mean"),
                ov=("overlap", "mean"),
                exo=("exogenous_share", "mean"),
            )
            .reindex(order)
        )
        xs = np.arange(3) + off
        axes[0].errorbar(xs, x.m, yerr=x.s, fmt="o", color=c, capsize=3, label=lab)
        axes[0].plot(xs, 0.5 + x.p, "x", color="k", ms=6)
        axes[1].plot(xs, x.ov, "o-", color=c, label=lab)
    axes[0].plot([], [], "x", color="k", label="Proposition 1")
    axes[0].axhline(0.5, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel(r"Estimate $\hat\theta$ (true $\theta = 0.5$)")
    axes[1].axhline(
        d.exogenous_share.mean(), color="0.4", lw=0.8, ls=":", label=r"exogenous share $\omega_V$"
    )
    axes[1].set_ylabel(r"Estimated overlap $\hat\omega$")
    for ax in axes:
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=7)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "delivery.png"), dpi=200)
    plt.close(fig)


def fig_diagnostic():
    path = os.path.join(RES, "diagnostic.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    rng = np.random.default_rng(0)
    d = d.groupby("case", group_keys=False).apply(
        lambda x: x.sample(min(len(x), 150), random_state=0)
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for conf, c, lab in (
        (False, "C0", "no confounded variation in the coarse residual"),
        (True, "C3", "confounded variation in the coarse residual"),
    ):
        x = d[d.confounded == conf]
        jitter = rng.normal(0, 0.004, len(x))
        ax.scatter(
            100 * (x.absorbed + jitter),
            x.z.abs().clip(lower=0.01),
            s=6,
            alpha=0.4,
            color=c,
            label=lab,
        )
    ax.axvline(5, color="k", lw=0.8, ls="--")
    ax.axhline(2, color="k", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("Share of coarse residual price variance absorbed by the refinement (%)")
    ax.set_ylabel(r"$|z|$, shift in the estimate (standard errors)")
    ax.text(38, 50, "flagged region", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "diagnostic.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_reactive()
    fig_coarse()
    fig_overlap_decomposition()
    fig_coverage()
    fig_bias_map()
    fig_application_data()
    fig_application()
    fig_nuisance_error()
    fig_delivery()
    fig_diagnostic()
    print("figures written to", FIG)
