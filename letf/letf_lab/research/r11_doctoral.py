"""R11 — Riesgo de estimación, regímenes, comparación de modelos y colas.

A. Riesgo de estimación en L*. Con rebalanceo diario y utilidad logarítmica,
   E[g(L)] = r + L(μ − f − s) − ER − ½L²σ² es lineal en μ, así que la
   incertidumbre sobre μ entra solo por la media a posteriori. Pero usar el
   L* estimado, L̂ = (μ̂ − f − s)/σ² (el L* aproximado), tiene un coste. Con σ²
   conocida, rentabilidades i.i.d. y μ̂ la media de N años de datos pasados,
   μ̂ ~ N(μ, σ²/N) y
       E[g(c·L̂)] − (r − ER) = c·SR² − ½c²(SR² + 1/N),   SR = (μ − f − s)/σ,
   que se maximiza en c* = SR²/(SR² + 1/N) (Kan y Zhou 2007, caso de
   covarianza conocida). c* depende del SR verdadero; como E[SR̂²] = SR² + 1/N,
   c*_insesgado = max(0, 1 − 1/(N·SR̂²)). Se compara con dos priors escépticos
   sobre la prima neta μ − f − s, en niveles (N(5%, 2,5%²)) y sobre el ratio de
   Sharpe (N(0,30; 0,15²)), con la probabilidad a posteriori de que el 3x crezca
   menos que las letras y con el L robusto de R10.
B. Regímenes: modelo de Markov de dos estados (HMM gaussiano, Hamilton 1989)
   sobre rentabilidades semanales en exceso de las letras, reestimado cada
   cinco años con ventana creciente y filtrado en tiempo real. L* por régimen
   y L* ex ante: (i) μ y σ² por régimen, con la varianza de dentro de cada
   régimen (con reset diario el crecimiento es lineal en μ); (ii) solo σ² por
   régimen. Contraste fuera de muestra de las medias por régimen.
C. Previsión de la varianza a 21 días: Model Confidence Set (Hansen, Lunde y
   Nason 2011) con pérdidas QLIKE y MSE sobre siete modelos, fuera de muestra, y
   con la pérdida exacta de crecimiento de la ecuación central con una prima
   constante, ½·SR²·(ρ − 1)² con ρ = realizada/prevista (la de Patton con
   b = −3), que ordena los modelos por el crecimiento que se pierde.
   El HAR logarítmico lleva la corrección de Duan en tiempo real
   (r10.har_var_forecast); el GARCH(1,1) se reestima cada cinco años sin datos
   futuros. Diagnóstico del sesgo de cada modelo, variante con todos los
   modelos calibrados en tiempo real y sensibilidad a la semilla.
D. Reality Check de White (2000) y test SPA de Hansen (2005) sobre las 15
   reglas de R10 con orden al cierre siguiente (el mismo cierre, como
   sensibilidad), rango entre semillas y potencia: error típico, efecto mínimo
   detectable y umbral del Reality Check.
E. Colas: picos sobre umbral con distribución de Pareto generalizada, IC por
   bootstrap de años naturales, estimador de momentos (Dekkers, Einmahl y de
   Haan 1989), sensibilidad al umbral e índice extremal por rachas (Smith y
   Weissman 1994); el estimador de Hill, solo si la cola es de tipo Pareto.
   Probabilidad de una caída diaria que liquide un 3x.
F. Error de la aproximación de segundo orden de la rentabilidad diaria del
   propio 3x (no la ecuación central de R10 §A): la cota de la Proposición,
   que se cumple por construcción, la reducción del error con los términos de
   tercer y cuarto orden y la descomposición del término cúbico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto, norm

from letf_lab import kelly, stats
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research import r10_predictability as r10

INDICES = r10.INDICES
SEED = 11
B = 500             # IC de ξ con excesos i.i.d. (la comparación de siempre)
B_MCS = 1000
B_TEST = 2000       # Reality Check, SPA y potencia
B_SEEDS = 1000      # remuestreos de cada semilla de la sensibilidad
N_SEEDS = 10
B_TAILS = 1000      # bootstrap de años naturales de las colas
SEED_MCS, SEED_TEST = 1100, 1200
SEED_MCS_G = 1150   # MCS de la pérdida de crecimiento (generador propio: no mueve el resto)
PRIOR_MEAN, PRIOR_SD = 0.05, 0.025        # prior en niveles sobre μ − f − s
PRIOR_SR_MEAN, PRIOR_SR_SD = 0.30, 0.15   # prior sobre el ratio de Sharpe (μ − f − s)/σ
REFIT_WEEKS = 260
MIN_WEEKS = 520
TAIL_Q = 0.95
TAIL_QS = (0.90, 0.95, 0.975, 0.99)
TAIL_K = 0.01       # fracción de la muestra en Hill y en el estimador de momentos
RUNS_R = 5          # sesiones sin exceso que separan dos rachas (índice extremal)
# sin «|»: rompería la tabla markdown
P_WORST = "P_GPD(peor ≥ esa caída, con θ)"       # 1 − (1 − p)^{nθ}
P_WORST_IND = "P_GPD(peor ≥ esa caída, θ = 1)"   # días independientes
TEXT_PER = ["GPD", "IC 90% (años)", "Hill", "normal"]  # columnas de texto de los periodos
MODELS = ["HAR (niveles)", "HAR (log)", "GARCH(1,1)", "EWMA λ=0,94", "RV 21 días", "RV 252 días",
          "media histórica"]


# ───────────────────────────── A. riesgo de estimación ─────────────────────────────

def posterior(mu: float, var: float, N: float, prior_mean: float | None = None,
              prior_sd: float | None = None) -> tuple[float, float]:
    """Media y desviación típica a posteriori de la prima con μ̂ ~ N(μ, var/N).
    Sin prior (prior_sd = None), el prior plano: (μ̂, √(var/N))."""
    if prior_sd is None:
        return mu, float(np.sqrt(var / N))
    prec = 1 / prior_sd ** 2 + N / var
    return (prior_mean / prior_sd ** 2 + mu * N / var) / prec, float(np.sqrt(1 / prec))


def p_below(threshold: float, m: float, sd: float) -> float:
    """P(prima < umbral) con la prima ~ N(m, sd²)."""
    return float(norm.cdf((threshold - m) / sd))


def estimation_risk(d: dict) -> dict:
    df = pd.concat([d["u"], d["rf"], d["f"]], axis=1, keys=["u", "r", "f"]).dropna()
    N = len(df) / 252
    s, er = d["spread"], d["er"]
    mu = (df.u - df.f).mean() * 252 - s  # prima neta μ − f − s, aritmética
    var = df.u.var(ddof=0) * 252
    sig = np.sqrt(var)
    sr = mu / sig
    L_hat = kelly.lstar_approx(df.u, df.f, s)  # = mu/var
    c_star = sr ** 2 / (sr ** 2 + 1 / N)
    c_unb = max(0.0, 1 - 1 / (N * sr ** 2))
    m_lv, sd_lv = posterior(mu, var, N, PRIOR_MEAN, PRIOR_SD)
    m_sr, sd_sr = posterior(mu, var, N, PRIOR_SR_MEAN * sig, PRIOR_SR_SD * sig)
    thr = (er + 4.5 * var) / 3  # g(3x) < r  ⇔  μ − f − s < (ER + 4,5σ²)/3
    return {"años": N, "SR": sr, kelly.LSTAR_APROX: L_hat, "c*": c_star, "c*·L̂": c_star * L_hat,
            "c*_insesgado": c_unb, "c*_insesgado·L̂": c_unb * L_hat,
            # pérdida esperada de crecimiento por usar L̂ en vez de L*: ½σ²·Var(L̂) = ½/N
            "pérdida_plug_in": 0.5 / N, "SE(μ)": np.sqrt(var / N),
            "μ_post": m_lv, "peso_datos": (N / var) / (1 / PRIOR_SD ** 2 + N / var),
            "L*_bayes": m_lv / var,
            "μ_post_SR": m_sr, "peso_datos_SR": (N / var) / (1 / (PRIOR_SR_SD * sig) ** 2 + N / var),
            "L*_bayes_SR": m_sr / var,
            "umbral_prima": thr,
            "P_post(g3x<letras)_plano": p_below(thr, *posterior(mu, var, N)),
            "P_post(g3x<letras)_niveles": p_below(thr, m_lv, sd_lv),
            "P_post(g3x<letras)_SR": p_below(thr, m_sr, sd_sr)}


# ───────────────────────────── B. regímenes ─────────────────────────────

def weekly(d: dict) -> pd.DataFrame:
    df = pd.concat([np.log1p(d["u"]), np.log1p(d["rf"]), np.log1p(d["f"])], axis=1,
                   keys=["u", "r", "f"]).dropna()
    w = df.resample("W-FRI").sum()
    w = w[df.resample("W-FRI").size() > 0]
    return w.assign(x=w.u - w.r, x_f=w.u - w.f)


def regime_lstar(pred, mu_w, sd_w, fr_w, s: float) -> tuple[np.ndarray, np.ndarray]:
    """L* de la variante (i) con parámetros semanales (log) por régimen.

    Numerador: 52·Σp_k(μ_k + ½σ_k²) − 52·(f − r)_semanal − s (μ es el exceso
    sobre las letras; se pasa a la financiación). Devuelve el L* con la varianza
    de dentro de cada régimen, 52·Σp_kσ_k², y, como sensibilidad, con la de la
    mezcla, 52·(Σp_k(σ_k² + μ_k²) − (Σp_kμ_k)²)."""
    pred, mu_w, sd_w = (np.asarray(a, float) for a in (pred, mu_w, sd_w))
    num = (pred * (mu_w + 0.5 * sd_w ** 2)).sum(axis=-1) * 52 - 52 * np.asarray(fr_w, float) - s
    v_in = (pred * sd_w ** 2).sum(axis=-1) * 52
    m = (pred * mu_w).sum(axis=-1)
    v_mix = ((pred * (sd_w ** 2 + mu_w ** 2)).sum(axis=-1) - m ** 2) * 52
    return num / v_in, num / v_mix


def regimes(d: dict) -> dict:
    w = weekly(d)
    x = w.x.to_numpy()
    full = stats.hmm_fit(x)
    s = d["spread"]
    fr_day = (d["f"] - d["rf"]).mean() * 252
    reg = pd.DataFrame({
        "μ_anual": (full["mu"] + 0.5 * full["sd"] ** 2) * 52,
        "σ_anual": full["sd"] * np.sqrt(52),
        "duración_semanas": 1 / (1 - np.diag(full["P"])),
        "prob_incondicional": full["smoothed"].mean(axis=0),
    }, index=["calma", "crisis"])
    reg["L*_régimen"] = (reg.μ_anual - fr_day - s) / reg.σ_anual ** 2
    # filtrado en tiempo real con reestimación periódica (parámetros semanales)
    n = len(x)
    pred = np.full((n, 2), np.nan)
    mu_w = np.full((n, 2), np.nan)
    sd_w = np.full((n, 2), np.nan)
    P_t = np.full((n, 2, 2), np.nan)
    params = None
    for t in range(MIN_WEEKS, n):
        if params is None or (t - MIN_WEEKS) % REFIT_WEEKS == 0:
            params = stats.hmm_fit(x[:t])
            probs = stats.hmm_filter(x, params)
        pred[t] = probs[t - 1]  # probabilidad para la semana t con datos hasta t−1
        mu_w[t], sd_w[t], P_t[t] = params["mu"], params["sd"], params["P"]
    p = pd.DataFrame(pred, index=w.index, columns=["calma", "crisis"])
    fr_w = (w.f - w.r).shift(1).expanding().mean().to_numpy()  # (f − r) semanal hasta t−1
    ex_in, ex_mix = regime_lstar(pred, mu_w, sd_w, fr_w, s)
    # (ii) solo σ² cambia con el régimen; μ − f es la media creciente hasta t−1
    xf = w.x_f.shift(1)
    mu_bar = (xf.expanding(MIN_WEEKS).mean() + 0.5 * xf.expanding(MIN_WEEKS).var()) * 52
    v_only = (pred * sd_w ** 2).sum(axis=1) * 52
    ex_sigma = pd.Series((mu_bar.values - s) / v_only, index=w.index)
    # «hoy»: previsión para la semana siguiente con datos hasta la última semana
    nxt = probs[-1]
    fr_now = (w.f - w.r).mean()
    today_in, today_mix = regime_lstar(nxt, params["mu"], params["sd"], fr_now, s)
    mu_now = (w.x_f.mean() + 0.5 * w.x_f.var()) * 52
    today_sigma = (mu_now - s) / ((nxt * params["sd"] ** 2).sum() * 52)
    # el mismo modelo con la varianza media prevista para las 52 semanas siguientes (horizonte de un
    # año, el del HAR anual): p_k = p₁·P^(k−1) y la media de p_k; numerador y denominador son lineales en p
    pk, acc = np.asarray(nxt, float), np.zeros(2)
    for _ in range(52):
        acc += pk
        pk = pk @ params["P"]
    p52 = acc / 52
    today_in52, _ = regime_lstar(p52, params["mu"], params["sd"], fr_now, s)
    today_sigma52 = (mu_now - s) / ((p52 * params["sd"] ** 2).sum() * 52)
    # validación: volatilidad realizada de las 4 semanas siguientes según el régimen previsto
    fwd_vol = w.x.rolling(4).std().shift(-4) * np.sqrt(52)
    crisis = p.crisis > 0.5
    calm = (p.crisis <= 0.5) & p.crisis.notna()
    val = {"vol_4s_si_calma": fwd_vol[calm].mean(), "vol_4s_si_crisis": fwd_vol[crisis].mean(),
           "semanas_crisis_prevista": crisis[p.crisis.notna()].mean(),
           "exceso_anual_si_calma": w.x[calm].mean() * 52,
           "exceso_anual_si_crisis": w.x[crisis].mean() * 52,
           "semanas_negativas_en_crisis_suavizada": float((full["smoothed"][:, 1][x < 0] > 0.5).mean())}
    oos = regime_mean_oos(w, pred, mu_w, P_t)
    return {"tabla": reg, "prob": p, "ex_ante": pd.Series(ex_in, index=w.index),
            "ex_ante_mezcla": pd.Series(ex_mix, index=w.index), "ex_ante_sigma": ex_sigma,
            "hoy": {"semana": w.index[-1].date(), "(i) μ y σ² por régimen": float(today_in),
                    "(i) con la varianza de la mezcla": float(today_mix),
                    "(ii) solo σ² por régimen": float(today_sigma),
                    "(i) a 52 semanas": float(today_in52), "(ii) a 52 semanas": float(today_sigma52),
                    "P(crisis) media a 52 semanas": float(p52[1])},
            "w": w, "validación": val, "oos": oos}


def regime_mean_oos(w: pd.DataFrame, pred, mu_w, P_t) -> dict:
    """¿Predicen las medias por régimen? Previsión de x (exceso semanal log sobre
    las letras) con Σ pred·μ a 1 semana y Σ_k (pred·P^k)·μ a 4 semanas, frente a
    la media creciente de x hasta t−1 (×4). R² fuera de muestra y Clark-West:
    a 1 semana sin solapamiento (0 retardos); a 4, hac_overlap(4)."""
    x = w.x
    bench = x.shift(1).expanding(MIN_WEEKS).mean()
    f1 = pd.Series((pred * mu_w).sum(axis=1), index=w.index).where(~np.isnan(pred[:, 0]))
    f4 = np.full(len(x), np.nan)
    for t in np.flatnonzero(~np.isnan(pred[:, 0])):
        p, acc = pred[t], 0.0
        for _ in range(4):
            acc += p @ mu_w[t]
            p = p @ P_t[t]
        f4[t] = acc
    y4 = x.rolling(4).sum().shift(-3)
    out = {}
    for name, y, f, b, kw in (("1 semana", x, f1, bench, {"lags": 0}),
                               ("4 semanas", y4, pd.Series(f4, index=w.index), 4 * bench,
                                stats.hac_overlap(4))):
        m = pd.concat([y, f, b], axis=1, keys=["y", "f", "b"]).dropna()
        cw = stats.clark_west_fit(m.y.values, m.b.values, m.f.values, **kw)
        out[name] = {"semanas": len(m), "desde": m.index[0].date(),
                     "R2_OOS": stats.oos_r2(m.y.values, m.f.values, m.b.values),
                     "CW_t": float(cw.t[0]), "CW_p": float(norm.sf(cw.t[0])),
                     "CW_estimador": cw.estimator}
    return out


# ───────────────────────────── C. previsión de varianza ─────────────────────────────

def variance_forecasts_full(d: dict, h: int = 21) -> pd.DataFrame:
    """Las siete previsiones de la varianza de (t, t+h] en toda la muestra, más
    el HAR logarítmico sin corregir (exp de la previsión del log, la media geométrica) y
    la realizada."""
    u = d["u"].dropna()
    rv = (u ** 2).rolling(h).sum().shift(-h) * 252 / h  # varianza realizada en (t, t+h]
    rv_d = u ** 2 * 252
    X = pd.DataFrame({"d": rv_d, "w": rv_d.rolling(5).mean(), "m": rv_d.rolling(21).mean()})
    f = {}
    f["HAR (niveles)"] = stats.expanding_forecast(rv, X, h=h, min_obs=r10.MIN_OBS).modelo.clip(lower=1e-4)
    hv = r10.har_var_forecast(rv, u, h)
    f["HAR (log)"] = hv.nivel
    # GARCH(1,1) reestimado cada 5 años con ventana creciente
    g = pd.Series(np.nan, index=u.index)
    r_arr = u.to_numpy()
    for start in range(r10.MIN_OBS, len(u), 252 * 5):
        p = stats.garch11_fit(r_arr[:start])
        h_next = stats.garch11_filter(r_arr - r_arr[:start].mean(), p)
        seg = slice(start, min(start + 252 * 5, len(u)))
        g.iloc[seg] = stats.garch11_horizon(h_next[seg], p, h) * 252
    f["GARCH(1,1)"] = g
    f["EWMA λ=0,94"] = rv_d.ewm(alpha=0.06, adjust=False).mean()
    f["RV 21 días"] = X.m
    f["RV 252 días"] = rv_d.rolling(252).mean()
    f["media histórica"] = rv_d.expanding(r10.MIN_OBS).mean()
    F = pd.DataFrame(f)
    F["HAR (log) sin corregir"] = hv.nivel_raw
    F["realizada"] = rv
    return F


def variance_forecasts(d: dict, h: int = 21, full: pd.DataFrame | None = None) -> pd.DataFrame:
    """Los siete modelos del MCS y la realizada, en la ventana en que existen todos."""
    F = variance_forecasts_full(d, h) if full is None else full
    first = F[MODELS].apply(lambda c: c.first_valid_index()).max()
    return F.loc[F.index >= first, MODELS + ["realizada"]].dropna()


def bias_table(F: pd.DataFrame, full: pd.DataFrame, h: int = 21) -> pd.DataFrame:
    """Sesgo de cada modelo en la ventana del MCS: cociente realizada/prevista y
    QLIKE, con la previsión tal cual y calibrada en tiempo real (la misma
    corrección multiplicativa para todos: media creciente del cociente con los
    objetivos ya observados, r10.duan_oos). El HAR (log) calibrado es el HAR
    (log) sin corregir con esa corrección, es decir, el de Duan."""
    y = F.realizada
    cal = {}
    for m in MODELS:
        base = full["HAR (log) sin corregir"] if m == "HAR (log)" else full[m]
        cal[m] = base * r10.duan_oos(full.realizada, np.log(base), h)
    C = pd.DataFrame(cal).reindex(F.index)
    ok = C.notna().all(axis=1)
    rows = {}
    for m in MODELS + ["HAR (log) sin corregir"]:
        f = full[m].reindex(F.index)
        ratio = y / f
        rows[m] = {"cociente_medio": ratio.mean(), "cociente_mediano": ratio.median(),
                   "QLIKE": float(stats.qlike(y.values, f.values).mean()),
                   "QLIKE_calibrado": float(stats.qlike(y[ok].values, C.loc[ok, m].values).mean())
                   if m in C else np.nan}
    out = pd.DataFrame(rows).T
    out.attrs.update(dias_calibrado=int(ok.sum()), desde_calibrado=F.index[ok][0].date())
    return out


growth_loss = r10.growth_loss   # pérdida exacta de crecimiento (vive en r10: §B y §C la usan)


def premium_const(d: dict) -> float:
    """Prima neta constante μ − f − s' de toda la muestra (la de §A)."""
    return r10.premium_const(d)


def mcs(F: pd.DataFrame, rng, B: int = B_MCS, premium: float | None = None,
        rng_g=None) -> pd.DataFrame:
    """MCS con QLIKE y MSE y, si se da la prima, con la pérdida exacta de crecimiento
    (growth_loss, en pb anuales). rng_g: generador propio de esa tercera pérdida, para no
    mover el de las otras dos ni el del resto del informe (por defecto, el mismo rng
    después de QLIKE y MSE)."""
    y = F.realizada.values
    models = F.drop(columns="realizada")
    ql = pd.DataFrame({c: stats.qlike(y, models[c].values) for c in models}, index=F.index)
    mse = pd.DataFrame({c: (y - models[c].values) ** 2 for c in models}, index=F.index)
    a = stats.model_confidence_set(ql, 63, B, rng).add_suffix("_QLIKE")
    b = stats.model_confidence_set(mse * 100, 63, B, rng).add_suffix("_MSE")
    out = a.join(b)
    if premium is not None:
        gl = pd.DataFrame({c: growth_loss(y, models[c].values, premium) * 1e4 for c in models},
                          index=F.index)
        c = stats.model_confidence_set(gl, 63, B, rng if rng_g is None else rng_g).add_suffix("_crec")
        out = out.join(c.rename(columns={"pérdida_media_crec": "pérdida_crec_pb"}))
    for k, col in (("QLIKE", "pérdida_media_QLIKE"), ("MSE", "pérdida_media_MSE"), ("crec", "pérdida_crec_pb")):
        if col in out:
            out[f"puesto_{k}"] = out[col].rank(method="min").astype(int)
    return out


def mcs_seeds(F: pd.DataFrame, n: int = N_SEEDS, B: int = B_MCS, seed0: int = SEED_MCS,
              premium: float | None = None) -> pd.DataFrame:
    """Nº de semillas (de n) en que cada modelo queda dentro del MCS, por pérdida."""
    cols = ["en_MCS_QLIKE", "en_MCS_MSE"] + (["en_MCS_crec"] if premium is not None else [])
    cnt = None
    for k in range(n):
        m = mcs(F, np.random.default_rng(seed0 + k), B, premium)[cols].astype(int)
        cnt = m if cnt is None else cnt + m
    return cnt.rename(columns={c: f"{c}_semillas" for c in cols})


def best_by_loss(tab: pd.DataFrame) -> dict:
    """Modelo con menor pérdida media para cada pérdida de la tabla del MCS."""
    out = {"QLIKE": tab["pérdida_media_QLIKE"].idxmin(), "MSE": tab["pérdida_media_MSE"].idxmin()}
    if "pérdida_crec_pb" in tab:
        out["crecimiento"] = tab["pérdida_crec_pb"].idxmin()
    return out


# ───────────────────────────── D. reglas: RC, SPA y potencia ─────────────────────────────

def rc_spa(D: pd.DataFrame, rng, B: int = B_TEST) -> dict:
    spa = stats.spa_test(D.values, 63, B, rng)
    rc = stats.reality_check(D.values, 63, B, rng)
    return {"p_RC": rc["p_RC"], "p_SPA_c": spa["p_SPA_c"], "p_SPA_l": spa["p_SPA_l"],
            "t_SPA": spa["t_SPA"], "mejor_regla": r10.RULE_LABELS.get(D.columns[rc["mejor"]], D.columns[rc["mejor"]]),
            "Δ_mejor": float(D.values.mean(axis=0)[rc["mejor"]] * 252)}


def power_table(D: pd.DataFrame, rng, B: int = B_TEST, block: float = 63,
                ref: str | None = r10.LRS_RULE) -> dict:
    """Potencia del contraste con remuestreo estacionario de las medias diarias.

    SE_k = sd bootstrap de la media de la regla k ×252. MDE_80_una =
    (z_0,95 + z_0,80)·SE: efecto mínimo detectable con potencia del 80% y
    α = 5% unilateral para UNA regla preespecificada. MDE_80_familia =
    (q95_max_t + z_0,80)·SE de la mejor regla, con q95_max_t el cuantil 95% del
    máximo de las t centradas de la familia: supone que la alternativa solo
    mueve la mejor regla. umbral_RC_5%: cuantil 95% del máximo de las medias
    centradas (puntos anuales). reglas_recentradas_SPA: reglas con
    √T·d̄/ω ≤ −√(2·log log T), las únicas que el SPA consistente recentra."""
    dv = np.asarray(D, float)
    T, K = dv.shape
    dbar = dv.mean(axis=0)
    m = np.array([dv[stats.stationary_indices(T, block, rng)].mean(axis=0) for _ in range(B)])
    sd = m.std(axis=0, ddof=1)
    z = norm.ppf(0.95) + norm.ppf(0.80)
    q95 = float(np.quantile(((m - dbar) / sd).max(axis=1), 0.95))
    best = int(dbar.argmax())
    mde_fam = (q95 + norm.ppf(0.80)) * sd * 252          # regla a regla: depende de su error típico
    out = {"Δ_mejor": dbar[best] * 252, "SE_mejor": sd[best] * 252, "t_mejor": dbar[best] / sd[best],
           "MDE_80_una": z * sd[best] * 252, "q95_max_t": q95,
           "MDE_80_familia": mde_fam[best],
           "MDE_80_familia_mín": float(mde_fam.min()), "MDE_80_familia_máx": float(mde_fam.max()),
           "umbral_RC_5%": float(np.quantile((m - dbar).max(axis=1), 0.95) * 252),
           "reglas_recentradas_SPA": int((dbar / sd <= -np.sqrt(2 * np.log(np.log(T)))).sum())}
    if ref is not None and isinstance(D, pd.DataFrame) and ref in D.columns:
        j = list(D.columns).index(ref)
        out.update({"Δ_media200": dbar[j] * 252, "SE_media200": sd[j] * 252,
                    "MDE_80_una_media200": z * sd[j] * 252})
    return out


def same_close_split(d: dict, rule: str = r10.LRS_RULE, split: str = "1990-01-01") -> dict:
    """Ventaja de ejecutar una regla en el mismo cierre (lag = 1) frente al cierre siguiente
    (lag = 2), en crecimiento log anual, antes y después de `split`, y lo que aporta el
    19-10-1987 (solo si cae en la muestra)."""
    w = r10.rules(d)[rule]
    diff = (r10.rule_returns(d, w, 1) - r10.rule_returns(d, w, 2)).dropna()
    start = d["X"]["tendencia"].first_valid_index()
    diff = diff.loc[start:]
    pre, post = diff.loc[:pd.Timestamp(split) - pd.Timedelta(days=1)], diff.loc[split:]
    crash = pd.Timestamp("1987-10-19")
    c = float(diff.get(crash, np.nan))
    return {"desde": diff.index[0].date(), "Δ_total": diff.mean() * 252,
            f"Δ_hasta_{int(split[:4]) - 1}": pre.mean() * 252 if len(pre) else np.nan,
            f"Δ_desde_{split[:4]}": post.mean() * 252,
            "19-10-1987_log": c,
            f"Δ_hasta_{int(split[:4]) - 1}_sin_19-10-1987": (pre.drop(crash, errors="ignore").mean() * 252
                                                             if len(pre) else np.nan)}


# ───────────────────────────── E. colas ─────────────────────────────

def period_years(p: float) -> float:
    """Periodo de retorno en años de un suceso diario de probabilidad p (∞ si p = 0)."""
    return np.inf if p <= 0 else 1 / (p * 252)


def years_es(y: float, log10: float | None = None) -> str:
    """Periodo legible con su unidad: «0,4 años», «1.998 años», «4,4 × 10⁹ años»,
    «nunca (cola acotada)». Un decimal por debajo de 10 años; a partir de un millón,
    notación científica con la mantisa redondeada antes de fijar el exponente
    (9,97·10⁹ → «1,0 × 10¹⁰»). Con log10 se da el logaritmo decimal del periodo
    (para la normal, que desborda)."""
    if log10 is None:
        if np.isinf(y):
            return "nunca (cola acotada)"
        if np.isnan(y):
            return "no aplicable"
        log10 = np.log10(y) if y > 0 else -np.inf
    if log10 < 6:
        v = 10 ** log10
        if round(v, 1) < 10:
            return f"{num_es(v, 1)} años"
        if round(v) < 1e6:
            return f"{num_es(v, 0)} años"
    e = int(np.floor(log10))
    m = round(10 ** (log10 - e), 1)
    if m >= 10:
        m, e = m / 10, e + 1
    sup = str(e).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
    return f"{num_es(m, 1)} × 10{sup} años"


def moment_estimator(losses: np.ndarray, k: int) -> float:
    """Estimador de momentos de Dekkers, Einmahl y de Haan (1989) del índice de
    cola ξ con las k mayores pérdidas; válido para cualquier signo de ξ."""
    s = np.sort(np.asarray(losses, float)[~np.isnan(losses)])[::-1]
    s = s[s > 0]
    lg = np.log(s[:k]) - np.log(s[k])
    m1, m2 = lg.mean(), (lg ** 2).mean()
    return float(m1 + 1 - 0.5 / (1 - m1 ** 2 / m2))


def extremal_index_runs(exceed, r: int = RUNS_R) -> float:
    """Índice extremal por rachas (Smith y Weissman 1994): nº de rachas / nº de
    excesos, con dos excesos en rachas distintas si les separan al menos r
    sesiones sin exceso. ≈ 1 sin agrupamiento; 1/θ es el tamaño medio de racha."""
    idx = np.flatnonzero(np.asarray(exceed, bool))
    if not len(idx):
        return np.nan
    return float((1 + (np.diff(idx) > r).sum()) / len(idx))


def p_max_at_least(p: float, n: int, theta: float = 1.0) -> float:
    """P(el peor de n días supera un nivel de probabilidad diaria p) = 1 − (1 − p)^{nθ}.

    θ = 1: días independientes. Con extremos agrupados (índice extremal θ < 1),
    P(M_n ≤ x) ≈ F(x)^{nθ} (Leadbetter 1983): n días valen como nθ independientes."""
    return float(1 - (1 - p) ** (n * theta))


def period_ci(pb: np.ndarray, level: float = 0.90) -> tuple[float, float, float]:
    """IC de un periodo de retorno a partir de las probabilidades bootstrap pb:
    se toman los cuantiles de la PROBABILIDAD, que es finita, y se invierten
    (una probabilidad 0, cola acotada, es un periodo infinito). Devuelve (bajo,
    alto, fracción de remuestreos con cola acotada, pb ≤ 0)."""
    pb = np.asarray(pb, float)
    a = (1 - level) / 2
    lo_p, hi_p = np.quantile(pb, [a, 1 - a])
    return period_years(hi_p), period_years(lo_p), float((pb <= 0).mean())


def year_bootstrap(u: pd.Series, rng, B: int = B_TAILS, q: float = TAIL_Q,
                   xs=(1 / 3, 1 / 2)) -> dict:
    """Bootstrap de años naturales: se remuestrean años con reposición, se
    reestima el umbral (cuantil q) y se reajusta la GPD. Conserva el
    agrupamiento de los extremos dentro de cada año."""
    groups = [-g.to_numpy() for _, g in u.groupby(u.index.year)]
    xis, probs, fails = [], [], 0
    for _ in range(B):
        sample = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        try:
            fit = stats.gpd_tail(sample, q)
            pr = [stats.gpd_exceedance_prob(fit, x) for x in xs]
        except Exception:  # noqa: BLE001 — ajuste fallido: se cuenta
            fails += 1
            continue
        xis.append(fit["xi"])
        probs.append(pr)
    return {"xi": np.array(xis), "p": np.array(probs), "fallidos": fails}


def tails(d: dict, rng) -> dict:
    u = d["u"].dropna()
    losses = -u.values
    n = len(losses)
    fit = stats.gpd_tail(losses, TAIL_Q)
    exc = losses[losses > fit["u"]]
    xis_iid = [genpareto.fit(rng.choice(exc, len(exc)) - fit["u"], floc=0)[0] for _ in range(B)]
    yb = year_bootstrap(u, rng)
    ci_y = np.percentile(yb["xi"], [5, 95])
    k = int(TAIL_K * n)
    xi_h = stats.hill(losses, k)
    xi_m = moment_estimator(losses, k)
    # criterio añadido en la revisión, cuando ya se conocían los valores de Hill: Hill solo si
    # la cola es de tipo Pareto según el IC 90% de ξ por años (límite inferior > 0) Y el
    # estimador de momentos (ξ > 0). El ξ de Hill se da siempre; su periodo, solo si pasa
    hill_ok = bool(ci_y[0] > 0 and xi_m > 0)
    theta = extremal_index_runs(losses > fit["u"])
    xk = np.sort(losses)[::-1][k]
    sd = u.std()
    main = {"n_días": n, "umbral_q95": fit["u"], "n_excesos": fit["n_exc"], "ξ_GPD": fit["xi"],
            "ξ_IC90_iid_bajo": np.percentile(xis_iid, 5), "ξ_IC90_iid_alto": np.percentile(xis_iid, 95),
            "ξ_IC90_años_bajo": ci_y[0], "ξ_IC90_años_alto": ci_y[1],
            "ξ_momentos": xi_m, "ξ_Hill": xi_h, "Hill_válido": hill_ok,
            "θ_rachas": theta, "remuestreos_fallidos": yb["fallidos"]}
    per = {}
    for j, L in enumerate((3, 2)):
        x = 1 / L
        p = stats.gpd_exceedance_prob(fit, x)
        ci_lo, ci_hi, bounded = period_ci(yb["p"][:, j])
        per_gpd = period_years(p)
        per_hill = period_years(k / n * (x / xk) ** (-1 / xi_h)) if hill_ok else np.nan
        logsf = norm.logsf(x / sd)
        log10_norm = -(logsf / np.log(10)) - np.log10(252)
        per[L] = {"caída_que_lo_liquida": -x, "periodo_GPD_años": per_gpd,
                  "IC90_bajo_años": ci_lo, "IC90_alto_años": ci_hi,
                  "fracción_cola_acotada": bounded,
                  "periodo_en_episodios_años": per_gpd / theta,
                  "periodo_Hill_años": per_hill, "log10_periodo_normal": log10_norm,
                  "log10(P_GPD/P_normal)": (np.log(p) - logsf) / np.log(10) if p > 0 else -np.inf,
                  "GPD": years_es(per_gpd),
                  "IC 90% (años)": f"{years_es(ci_lo)} – {years_es(ci_hi)}",
                  "Hill": years_es(per_hill) if hill_ok else "no aplicable",
                  "normal": years_es(np.nan, log10_norm)}
    thr = {}
    for q in TAIL_QS:
        fq = stats.gpd_tail(losses, q)
        pq = period_years(stats.gpd_exceedance_prob(fq, 1 / 3))
        thr[q] = {"umbral": fq["u"], "n_excesos": fq["n_exc"], "ξ": fq["xi"],
                  "periodo_3x_años": pq, "periodo_3x": years_es(pq)}
    wd = u.idxmin()
    pw = stats.gpd_exceedance_prob(fit, -u.min())
    worst = {"fecha": wd.date(), "caída_índice": u.min(), "3x_ese_día": float(d["x3"].get(wd, np.nan)),
             "periodo_GPD_años": period_years(pw), "periodo_GPD": years_es(period_years(pw)),
             P_WORST: p_max_at_least(pw, n, theta), P_WORST_IND: p_max_at_least(pw, n)}
    return {"main": main, "per": per, "thr": thr, "worst": worst, "fit": fit, "losses": losses}


# ───────────────────────────── F. aproximación ─────────────────────────────

def cubic_split(u: pd.Series, L: float = 3.0) -> dict:
    """Término cúbico anual del fondo L·u en pb, L³/3·252·E[u³], y su reparto:
    E[u³] = E[(u − μ)³] (asimetría) + 3μσ² + μ³ (media), con momentos poblacionales."""
    u = np.asarray(u, float)
    u = u[~np.isnan(u)]
    mu = u.mean()
    c = u - mu
    k = L ** 3 / 3 * 252 * 1e4
    return {"término_cúbico_3x_pb": k * (u ** 3).mean(),
            "de_la_asimetría_pb": k * (c ** 3).mean(),
            "de_la_media_pb": k * (3 * mu * (c ** 2).mean() + mu ** 3)}


def approximation(d: dict, h: int = 756) -> dict:
    x = d["x3"].dropna().clip(lower=-0.999)
    ws = lambda s: r10.window_sum(s, h) * 252 / h  # noqa: E731
    exact = ws(np.log1p(x))
    g2 = ws(x - x ** 2 / 2)
    g4 = ws(x - x ** 2 / 2 + x ** 3 / 3 - x ** 4 / 4)
    cubic = ws(x ** 3 / 3)
    bound = ws(np.abs(x) ** 3 / (3 * (1 - np.abs(x).clip(upper=0.99))))
    df = pd.DataFrame({"exacto": exact, "g2": g2, "g4": g4, "cúbico": cubic, "cota": bound}).dropna()
    e2, e4 = df.exacto - df.g2, df.exacto - df.g4
    u = d["u"].dropna()
    rm2, rm4 = np.sqrt((e2 ** 2).mean()), np.sqrt((e4 ** 2).mean())
    return {"df": df,
            "RMSE_2º_orden": rm2, "máx_error_2º": e2.abs().max(),
            "RMSE_4º_orden": rm4, "máx_error_4º": e4.abs().max(),
            "reducción_error_cuadrático": 1 - (e4 ** 2).sum() / (e2 ** 2).sum(),
            "reducción_RMSE": 1 - rm4 / rm2,
            "cota/error (medias)": df.cota.mean() / e2.abs().mean(),
            "asimetría_diaria": float(((u - u.mean()) ** 3).mean() / u.std() ** 3),
            "curtosis_diaria": float(((u - u.mean()) ** 4).mean() / u.std() ** 4),
            **cubic_split(u),
            "término_cuártico_3x_pb": -81 * (u ** 4).mean() / 4 * 252 * 1e4}


# ───────────────────────────── figuras ─────────────────────────────

def fig_regimes(reg: dict, rep: Report):
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 10.5), sharex=False)
    for ax, (name, r) in zip(axes, reg.items()):
        w = r["w"]
        px = np.exp(w.u.cumsum())
        ax.semilogy(px, color="k", lw=0.8)
        ax2 = ax.twinx()
        ax2.fill_between(r["prob"].index, 0, r["prob"].crisis.fillna(0), color="C3", alpha=0.3, step="pre")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("P(crisis)", color="C3")
        ax.set(title=r10.fig_name(name), ylabel="Índice (log)")
    fig.suptitle("Índice (negro) y probabilidad de régimen de crisis en tiempo real (rojo)\n"
                 "según un modelo de Markov de dos estados", fontsize=14)
    comma_fig(fig)
    rep.fig(fig, "regimenes", "Probabilidad en tiempo real del régimen de crisis según un modelo de cambio de régimen")


def fig_tails(tl: dict, rep: Report):
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 11.6))
    for ax, (name, t) in zip(axes, tl.items()):
        l = np.sort(t["losses"][t["losses"] > 0])[::-1]  # noqa: E741
        emp = np.arange(1, len(l) + 1) / len(t["losses"])
        ax.loglog(l * 100, emp, ".", ms=3, color="0.4", label="datos")
        grid = np.linspace(t["fit"]["u"] * 1.01, 0.6, 200)
        ax.loglog(grid * 100, [stats.gpd_exceedance_prob(t["fit"], g) for g in grid], color="C3",
                  label="ajuste de Pareto generalizada")
        sd = np.std(t["losses"])
        ax.loglog(grid * 100, norm.sf(grid / sd), "--", color="C0", label="normal")
        for L, c in ((3, "r"), (2, "m")):
            ax.axvline(100 / L, color=c, ls=":", lw=1)
            # arriba y a la izquierda de su línea: ni la línea ni las curvas atraviesan el texto
            ax.text(100 / L * 0.96, 0.5, f"liquida un {L}x", color=c, fontsize=11, rotation=90,
                    va="top", ha="right")
        ax.set(title=f"{r10.fig_name(name)} (ξ = {num_es(t['fit']['xi'], 2)})",
               xlabel="Caída diaria del índice (%)", ylim=(1e-9, 1), xlim=(1, 70),
               ylabel="P(caída diaria > x)")
        ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    h, lab = axes[0].get_legend_handles_labels()
    legend_band(fig, h, lab, ncol=3, markerscale=3)
    fig.suptitle("Cola izquierda diaria: datos, Pareto generalizada y normal")
    rep.fig(fig, "colas", "Probabilidad de caídas diarias extremas: datos, ajuste de Pareto generalizada y normal")


def fig_approx(ap: dict, rep: Report):
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 10.5))
    for ax, (name, a) in zip(axes, ap.items()):
        df = a["df"]
        ax.plot((df.exacto - df.g2) * 1e4, lw=0.8, color="C3", label="error de 2.º orden")
        ax.plot((df.exacto - df.g4) * 1e4, lw=0.8, color="C2", label="error de 4.º orden")
        ax.plot(df.cúbico * 1e4, lw=0.8, ls="--", color="k", label="término cúbico del 3x")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set(title=r10.fig_name(name), ylabel="Puntos básicos al año")
        ax.legend(loc="best")
    fig.suptitle("Error de la aproximación de segundo orden en ventanas de 3 años (3x)")
    comma_fig(fig)
    rep.fig(fig, "aproximacion", "Error de la aproximación de segundo orden y el término cúbico")


# ───────────────────────────── informe ─────────────────────────────

def run():
    rep = r10.Report1("r11_doctoral", "R11 — Riesgo de estimación, regímenes, modelos y colas")
    rep.p(__doc__.split("\n", 2)[2])
    rng = np.random.default_rng(SEED)
    D = {name: r10.load(t) for name, t in INDICES}

    rep.h("A. Riesgo de estimación en el apalancamiento de Kelly")
    rep.table(pd.DataFrame({n: estimation_risk(d) for n, d in D.items()}).T,
              pct={"pérdida_plug_in", "μ_post", "μ_post_SR", "SE(μ)", "peso_datos", "peso_datos_SR",
                   "umbral_prima", "P_post(g3x<letras)_plano", "P_post(g3x<letras)_niveles",
                   "P_post(g3x<letras)_SR"}, digits=3)
    rep.p(f"Prima: μ − f − s', la media aritmética de u − f ×252 menos s' (neta de la financiación "
          f"interbancaria, no de las letras). {kelly.LSTAR_APROX} = L̂ = prima/σ² con toda la historia. "
          "c* = SR²/(SR² + 1/N) es la fracción del Kelly estimado que maximiza el crecimiento "
          "esperado teniendo en cuenta el error de μ̂; depende del SR verdadero, y la columna c* "
          "lo sustituye por el SR̂ muestral (plug-in), que la sesga al alza porque "
          "E[SR̂²] = SR² + 1/N. c*_insesgado corrige ese sesgo con el estimador insesgado de SR², "
          "SR̂² − 1/N: c*_insesgado = max(0, 1 − 1/(N·SR̂²)). pérdida_plug_in = ½/N es el crecimiento "
          "anual esperado que se pierde por usar L̂ en lugar del L* verdadero. L*_bayes: media a "
          f"posteriori de la prima con un prior N({pct_es(PRIOR_MEAN, 0)}; {pct_es(PRIOR_SD)}²) en "
          f"niveles; L*_bayes_SR, con un prior común sobre el ratio de Sharpe, "
          f"SR ~ N({num_es(PRIOR_SR_MEAN, 2)}; {num_es(PRIOR_SR_SD, 2)}²), es decir, una prima a "
          "priori N(0,30σ̂; (0,15σ̂)²). Los dos priors son elecciones del investigador. peso_datos: "
          "peso de la media muestral en la media a posteriori. P_post: probabilidad a posteriori de "
          "que el 3x crezca menos que las letras, g(3) < r ⇔ prima < umbral_prima = (ER + 4,5σ²)/3, "
          "con prior plano, en niveles y sobre el Sharpe. Supuestos: σ² conocida, rentabilidades "
          "i.i.d. y μ̂ estimada con datos pasados.")

    rep.h("B. Regímenes (modelo de Markov de dos estados, rentabilidades semanales en exceso)")
    reg, oos, today = {}, {}, {}
    for name, d in D.items():
        r = reg[name] = regimes(d)
        rep.h(name, 3)
        rep.table(r["tabla"], pct={"μ_anual", "σ_anual", "prob_incondicional"}, digits=2)
        v = r["validación"]
        rep.p(f"Validación fuera de muestra: volatilidad realizada de las 4 semanas siguientes "
              f"**{pct_es(v['vol_4s_si_calma'])}** cuando el filtro prevé calma y "
              f"**{pct_es(v['vol_4s_si_crisis'])}** cuando prevé crisis "
              f"({pct_es(v['semanas_crisis_prevista'], 0)} de las semanas). Exceso anual realizado "
              f"sobre las letras: {pct_es(v['exceso_anual_si_calma'])} en semanas con calma prevista "
              f"y {pct_es(v['exceso_anual_si_crisis'])} con crisis prevista. De las semanas con "
              f"exceso negativo, {pct_es(v['semanas_negativas_en_crisis_suavizada'], 0)} tienen una "
              "probabilidad suavizada de crisis mayor que 0,5.")
        oos.update({(name, k): x for k, x in r["oos"].items()})
        today[name] = {**r["hoy"],
                       "(i) mediana histórica": r["ex_ante"].median(),
                       "(ii) mediana histórica": r["ex_ante_sigma"].median()}
    rep.p("μ_anual: media del exceso semanal sobre las letras en cada régimen, (μ + ½σ²)·52. "
          "L*_régimen = (μ_anual − media(f − r) − s')/σ²_anual con toda la muestra.")
    rep.h("L* ex ante por régimen: hoy y mediana histórica", 3)
    rep.table(pd.DataFrame(today).T, digits=2)
    rep.p("(i) μ y σ² por régimen: (52·Σp_k(μ_k + ½σ_k²) − 52·(f − r)_semanal − s')/(52·Σp_kσ_k²), "
          "con p la probabilidad prevista de cada régimen y la varianza de dentro de cada uno: en un "
          "fondo de reset diario el crecimiento es lineal en μ, así que la dispersión de μ entre "
          "regímenes no es riesgo. «con la varianza de la mezcla» añade esa dispersión, como "
          "sensibilidad. (ii) solo σ² por régimen: μ − f con la media creciente del exceso semanal "
          "sobre f. hoy: previsión para la semana siguiente a la última semana de datos "
          "(semana), con las probabilidades filtradas hasta esa semana. «a 52 semanas» (añadido tras "
          "una revisión externa): el mismo modelo con la probabilidad media de cada régimen en las 52 "
          "semanas siguientes, p̄ = (1/52)·Σₖ p₁·Pᵏ⁻¹, es decir, con la varianza media prevista para "
          "el año siguiente, el horizonte del HAR anual de R10 §E; «P(crisis) media a 52 semanas» es "
          "esa probabilidad media de crisis. La diferencia entre la columna de la semana siguiente y la "
          "de 52 semanas es la parte del horizonte; la que queda frente al HAR es de modelo (varianza "
          "semanal frente a diaria, corrección de Duan y prima).")
    rep.h("¿Predicen las medias por régimen? Contraste fuera de muestra", 3)
    ot = pd.DataFrame(oos).T
    ot.index.names = ["índice", "horizonte"]
    rep.table(ot, pct={"R2_OOS"}, digits=3)
    rep.p("Previsión del exceso semanal log sobre las letras con las medias por régimen y las "
          "probabilidades previstas (a 4 semanas, con la matriz de transición), frente a la media "
          "creciente del exceso hasta la semana anterior. Clark-West unilateral: a 1 semana sin "
          "retardos; a 4 semanas, con objetivos solapados, Hansen-Hodrick con 3 retardos (recurso "
          "Newey-West 8; CW_estimador dice cuál se usó).")
    fig_regimes(reg, rep)

    rep.h("C. Previsión de la varianza a 21 días: Model Confidence Set")
    best, prem = {}, {}
    for k, (name, d) in enumerate(D.items()):
        full = variance_forecasts_full(d)
        F = variance_forecasts(d, full=full)
        prem[name] = premium_const(d)
        rep.h(f"{name} ({F.index[0].date()} → {F.index[-1].date()}, {num_es(len(F), 0)} días)", 3)
        tab = mcs(F, rng, premium=prem[name], rng_g=np.random.default_rng(SEED_MCS_G + k))
        tab = tab.join(mcs_seeds(F, premium=prem[name]))
        best[name] = best_by_loss(tab)
        rep.table(tab, digits=3)
        bt = bias_table(F, full)
        rep.p(f"Diagnóstico del sesgo en la misma ventana (QLIKE_calibrado: {num_es(bt.attrs['dias_calibrado'], 0)} "
              f"días desde {bt.attrs['desde_calibrado']}):")
        rep.table(bt.sort_values("QLIKE_calibrado"), digits=3)
    rep.p(f"Horizonte de 21 sesiones. en_MCS: el modelo pertenece al conjunto de mejores modelos al "
          f"90% (B = {num_es(B_MCS, 0)}, bloque de 63 sesiones); en_MCS_*_semillas: en cuántas de {N_SEEDS} "
          "semillas adicionales queda dentro. HAR (log): exponencial de la previsión del logaritmo "
          "por el factor de Duan calculado en tiempo real (r10.har_var_forecast); la ventana común "
          "empieza cuando existe ese factor, así que el MCS cambia también por la ventana. "
          "«HAR (log) sin corregir» (exp de la previsión del logaritmo: la media geométrica, que coincide "
          "con la mediana solo si el error del logaritmo es simétrico y queda por debajo de la media) no entra en el "
          "MCS. cociente_medio y cociente_mediano: realizada/prevista. QLIKE_calibrado: la misma "
          "corrección multiplicativa en tiempo real para todos los modelos (media creciente del "
          "cociente con los objetivos ya observados), para ver si el orden depende de corregir solo "
          "el HAR logarítmico. GARCH(1,1) reestimado cada cinco años sin datos futuros. MSE y QLIKE "
          "son ambas robustas al ruido de usar la varianza realizada como proxy (Patton 2011); MSE "
          "da más peso a los episodios de volatilidad extrema.")
    rep.p("*_crec: la pérdida exacta de crecimiento de la ecuación central (r11.growth_loss). Con una "
          "prima constante m = μ − f − s', fijar el apalancamiento con la varianza prevista σ̂² en vez de "
          "la realizada σ² cuesta g(L*) − g(m/σ̂²) = ½·SR²·(ρ − 1)² de crecimiento log anual, con "
          "ρ = σ²/σ̂² y SR² = m²/σ²: proporcional a (ρ − 1)²/σ², que es la pérdida robusta de Patton "
          "(2011) con b = −3, así que es tan válida como QLIKE y MSE para ordenar modelos con una varianza "
          "realizada ruidosa. pérdida_crec_pb: su media en puntos básicos de crecimiento anual frente al "
          "apalancamiento que usaría la varianza realizada de cada mes (una referencia que ningún modelo "
          "alcanza, así que los niveles son grandes y lo que informa es el orden), con m la prima de toda "
          "la muestra (§A: "
          + "; ".join(f"{n}, {pct_es(m, 1)}" for n, m in prem.items())
          + "); el orden y el MCS no dependen de m. QLIKE ≈ ½(ρ − 1)² solo para errores pequeños: mide la "
          "fracción del crecimiento adicional que se pierde, pesa igual todos los periodos y castiga sin "
          "límite pasarse en la varianza prevista (ρ → 0), mientras que la pérdida de crecimiento pesa "
          "más la calma (SR² = m²/σ²), crece con el cuadrado de ρ − 1 al quedarse corto (ρ > 1) y, al "
          "pasarse, cuesta como mucho ½·SR². Modelo con menor pérdida media, por pérdida: "
          + "; ".join(f"{n}: QLIKE, {b['QLIKE']}; MSE, {b['MSE']}; crecimiento, {b['crecimiento']}"
                      for n, b in best.items()) + ".")

    rep.h("D. Reality Check y test SPA sobre las 15 reglas de R10")
    rows, seeds, powr, lrs = {}, {}, {}, {}
    for name, d in D.items():
        Dd, _, _ = r10.rule_diffs(d)
        rows[(name, "cierre siguiente (base)")] = {**rc_spa(Dd, rng), "días": len(Dd),
                                                   "desde": Dd.index[0].date()}
        D1, _, _ = r10.rule_diffs(d, lag=1)
        rows[(name, "mismo cierre (lag = 1)")] = {**rc_spa(D1, rng), "días": len(D1),
                                                  "desde": D1.index[0].date()}
        ps = pd.DataFrame([rc_spa(Dd, np.random.default_rng(SEED_TEST + k), B_SEEDS)
                           for k in range(N_SEEDS)])
        seeds[name] = {f"{c} ({q})": getattr(ps[c], q)() for c in ("p_RC", "p_SPA_c", "p_SPA_l")
                       for q in ("min", "max")}
        powr[name] = power_table(Dd, rng)
        lrs[name] = r10.rule_table(d).loc[r10.LRS_RULE, ["Δ_crec_anual", "p_nominal", "p_Holm", "q_BH",
                                                          "tiempo_dentro"]].to_dict()
    tt = pd.DataFrame(rows).T
    tt.index.names = ["índice", "ejecución"]
    rep.table(tt, pct={"Δ_mejor"}, digits=3)
    rep.p(f"B = {num_es(B_TEST, 0)} remuestreos estacionarios con bloque de 63 sesiones. La matriz de "
          "diferencias es la de R10 §C (r10.rule_diffs): regla menos mantener el 3x, orden al cierre "
          "siguiente; «mismo cierre» ejecuta en el cierre de la señal. El SPA estudentiza y recentra "
          "las reglas claramente peores que mantener el 3x, así que es menos sensible que el Reality "
          "Check a la inclusión de reglas malas, pero las que no recentra le siguen restando potencia.")
    rep.h(f"Rango entre {N_SEEDS} semillas (B = {num_es(B_SEEDS, 0)}, cierre siguiente)", 3)
    rep.table(pd.DataFrame(seeds).T, digits=3)
    rep.h("Potencia: error típico y efecto mínimo detectable (puntos anuales)", 3)
    rep.table(pd.DataFrame(powr).T, pct={"Δ_mejor", "SE_mejor", "MDE_80_una", "MDE_80_familia",
                                         "MDE_80_familia_mín", "MDE_80_familia_máx", "umbral_RC_5%", "Δ_media200", "SE_media200",
                                         "MDE_80_una_media200"}, digits=2)
    rep.p("SE: desviación típica bootstrap de la diferencia media anual. MDE_80_una = "
          "(z₀,₉₅ + z₀,₈₀)·SE = 2,486·SE: la mejora mínima que una regla preespecificada detectaría "
          "con potencia del 80% y α = 5% unilateral. MDE_80_familia = (q95_max_t + z₀,₈₀)·SE de la "
          "mejor regla, con q95_max_t el cuantil 95% del máximo de las t centradas de las 15: "
          "supone que la alternativa solo mueve la mejor regla. En un contraste estudentizado el umbral "
          "es proporcional al error típico de cada regla: MDE_80_familia_mín y _máx dan el rango entre "
          "las 15 reglas, (q95_max_t + z₀,₈₀)·SE_k (las que casi siempre están dentro del 3x tienen "
          "menos error típico y bastaría menos). umbral_RC_5%: la mejora anual que "
          "la mejor regla necesitaría para que el Reality Check rechazara al 5%. "
          "reglas_recentradas_SPA: reglas con t ≤ −√(2·log log T), las únicas que el SPA recentra. "
          "*_media200: la regla «dentro solo sobre la media de 200 sesiones» (la señal de la LRS), "
          "preespecificada.")
    rep.h("El mismo cierre antes y después de 1990 (regla «dentro solo sobre la media de 200 sesiones»)", 3)
    rep.table(pd.DataFrame({n: same_close_split(d) for n, d in D.items()}).T,
              pct={"Δ_total", "Δ_hasta_1989", "Δ_desde_1990", "19-10-1987_log",
                   "Δ_hasta_1989_sin_19-10-1987"}, digits=3)
    rep.p("Añadido tras una revisión externa. Crecimiento log anual de ejecutar la regla en el mismo "
          "cierre de la señal menos el de ejecutarla al cierre siguiente (la base), en toda la muestra y "
          "antes y después de 1990; «19-10-1987_log» es la diferencia de ese día (el mismo cierre estaba "
          "fuera y el siguiente, dentro). Hasta 1989 el S&P 500 diario tiene autocorrelación positiva, la "
          "huella de precios desfasados que no se podían negociar (R15): una ventaja del mismo cierre "
          "concentrada antes de 1990 es compatible con ese efecto y no sería operable.")
    rep.h("La regla «dentro solo sobre la media de 200 sesiones» (de la tabla de R10 §C)", 3)
    rep.table(pd.DataFrame(lrs).T, pct={"Δ_crec_anual", "tiempo_dentro"}, digits=3)
    rep.p("Misma fuente que R10 §C (r10.rule_table, con su propia semilla): p nominal, Holm y "
          "Benjamini-Hochberg sobre las 15 reglas.")

    rep.h("E. Colas: probabilidad de una caída diaria que liquide el fondo")
    tl = {name: tails(d, rng) for name, d in D.items()}
    rep.table(pd.DataFrame({n: t["main"] for n, t in tl.items()}).T, pct={"umbral_q95"}, digits=3)
    rep.p(f"GPD: picos sobre el cuantil {pct_es(TAIL_Q, 0)} de las pérdidas diarias. ξ_IC90_iid "
          "(bajo, alto): "
          f"remuestreo de los excesos como si fueran independientes, con el umbral fijo (B = {num_es(B, 0)}); "
          f"ξ_IC90_años: bootstrap de años naturales, que reestima el umbral y conserva el "
          f"agrupamiento de los extremos (B = {num_es(B_TAILS, 0)}). ξ_momentos: estimador de Dekkers, Einmahl "
          f"y de Haan (1989) con k = {pct_es(TAIL_K, 0)} de la muestra, válido para cualquier signo "
          "de ξ. ξ_Hill: estimador de Hill con el mismo k, que solo es consistente si ξ > 0. "
          "Hill_válido: el límite inferior del IC 90% de ξ por años y el estimador de momentos son "
          "positivos; si no, el periodo de Hill no se da (el ξ de Hill sí). Es un criterio añadido en "
          "la revisión, cuando ya se conocían los valores de Hill, no fijado de antemano. "
          f"θ_rachas: índice extremal por rachas (r = {RUNS_R} sesiones) en el umbral q95.")
    rep.h("Periodo de retorno de la caída que liquida un 3x (−33,3%) y un 2x (−50%)", 3)
    pt = pd.DataFrame({(n, f"{L}x"): v for n, t in tl.items() for L, v in t["per"].items()}).T
    pt.index.names = ["índice", "fondo"]
    rep.table(pt.drop(columns=TEXT_PER), pct={"caída_que_lo_liquida", "fracción_cola_acotada"}, digits=1)
    rep.p("Periodos en años de sesiones (252 al año). IC 90%: cuantiles de la probabilidad de "
          "excedencia en el bootstrap por años, invertidos (una probabilidad 0, cola acotada, es un "
          "periodo infinito); fracción_cola_acotada: remuestreos en que la GPD ajustada hace "
          "imposible esa caída. periodo_en_episodios: periodo GPD / θ, el de un episodio de "
          "pérdidas extremas agrupadas. normal: con la desviación típica diaria del índice; "
          "log10(P_GPD/P_normal): órdenes de magnitud que la normal subestima. Los mismos "
          "periodos en texto, con el formato del artículo (coma decimal y punto de miles; las "
          "tablas numéricas de este informe usan punto decimal):")
    rep.table(pt[TEXT_PER])
    rep.h("Sensibilidad al umbral (caída del 33,3%)", 3)
    ht = pd.DataFrame({(n, q): v for n, t in tl.items() for q, v in t["thr"].items()}).T
    ht.index.names = ["índice", "cuantil"]
    rep.table(ht.drop(columns="periodo_3x"), pct={"umbral"}, digits=3)
    rep.p("periodo_3x_años en texto, con el formato del artículo:")
    rep.table(ht[["periodo_3x"]])
    rep.h("El peor día de cada índice", 3)
    wt = pd.DataFrame({n: t["worst"] for n, t in tl.items()}).T
    rep.table(wt.drop(columns="periodo_GPD"),
              pct={"caída_índice", "3x_ese_día", P_WORST, P_WORST_IND}, digits=1)
    rep.p("3x_ese_día: la rentabilidad del 3x de la base ese día (la réplica antes del lanzamiento). "
          f"{P_WORST} = 1 − (1 − p)^(nθ), con p la probabilidad diaria de la GPD, n los días de la "
          "muestra y θ el índice extremal por rachas de la primera tabla (θ_rachas, estimado en "
          "el umbral q95): con los extremos agrupados, el peor de n días se comporta como el de "
          f"nθ días independientes (Leadbetter 1983). {P_WORST_IND} supone días independientes "
          "y da una probabilidad mayor. Si no es pequeña, el peor día es compatible con el ajuste. "
          "periodo_GPD_años en texto: "
          + "; ".join(f"{n}, {t['worst']['periodo_GPD']}" for n, t in tl.items())
          + ". Un 3x queda a cero si el índice cae un 33,3% en un día (un 2x, un 50%). Desde 1998 una "
          "caída del 30% del Dow Jones, y desde 2013 una del 20% del S&P 500, cierran la sesión en todo "
          "el mercado de acciones de EE. UU., también en las acciones del Nasdaq-100 y de "
          "semiconductores. Eso trunca en la práctica la cola de cierre a cierre, la que cuenta para un "
          "fondo de reset diario, pero no impide que la caída siga en la sesión siguiente. El disparador "
          "es el Dow Jones hasta 2013 y el S&P 500 después, no el Nasdaq-100 ni los semiconductores: "
          "estos solo podrían caer un 33,3% en un día si lo hicieran antes de que el índice de "
          "referencia llegara a su umbral.")
    fig_tails(tl, rep)

    rep.h("F. Precisión de la aproximación de segundo orden (rentabilidad diaria del 3x, ventanas de 3 años)")
    ap = {name: approximation(d) for name, d in D.items()}
    rep.table(pd.DataFrame({n: {k: v for k, v in a.items() if k != "df"} for n, a in ap.items()}).T,
              pct={"RMSE_2º_orden", "máx_error_2º", "RMSE_4º_orden", "máx_error_4º",
                   "reducción_error_cuadrático", "reducción_RMSE"}, digits=2)
    rep.p("Es la expansión de log(1 + x) en la rentabilidad diaria x del propio 3x, no la ecuación "
          "central de R10 §A (allí, la columna de momentos). Proposición: para |x| < 1, "
          "|log(1+x) − x + x²/2| ≤ |x|³ / (3(1 − |x|)); vale día a día y, por la desigualdad "
          "triangular, el error agregado de cada ventana queda por debajo de la suma de las cotas "
          "diarias por construcción. cota/error (medias): cociente entre la cota media y el error "
          "absoluto medio, como dato informativo. reducción_error_cuadrático = 1 − SSE(4.º)/SSE(2.º) "
          "y reducción_RMSE = 1 − RMSE(4.º)/RMSE(2.º). Los términos cúbico (+L³·E[u³]/3) y cuártico "
          "(−L⁴·E[u⁴]/4) usan los momentos brutos diarios del índice (en pb al año); el cúbico se "
          "reparte en la parte de la asimetría, 27/3·252·E[(u − μ)³], y la de la media, "
          "27/3·252·(3μσ² + μ³).")
    fig_approx(ap, rep)
    rep.save()


if __name__ == "__main__":
    run()
